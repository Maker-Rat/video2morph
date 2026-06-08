#!/usr/bin/env python3
"""Live-ish video -> SMPL -> optional GMR -> morph student -> target robot viewer."""

import argparse
import atexit
import contextlib
import csv
import os
import pickle
import select
import sys
import termios
import time
import tty
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from loguru import logger
from tqdm import tqdm

from video_to_g1_gmr import (
    SmplxFrameAdapter,
    configure_backbone_trt_for_image_size,
    suppress_output,
    sync_cuda,
)
from video_to_gmr_smpl_npz import (
    RootTranslationMapper,
    choose_person,
    parse_root_correction,
    root_orient_from_output,
    smooth_pose_body,
    smooth_rotvec,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROBOT_ASSET_ROOT = REPO_ROOT / "assets" / "robots"


def _to_wxyz(quat_xyzw):
    q = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


def _wxyz_to_xyzw(quat_wxyz):
    q = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)


def _rotvec_to_wxyz(rotvec):
    quat_xyzw = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64).reshape(3)).as_quat()
    return _to_wxyz(quat_xyzw.astype(np.float32))


def _unique_output_path(path):
    path = Path(path).expanduser()
    if not path.exists():
        return path
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = path.suffix
    stem = path.stem if suffix else path.name
    for idx in range(1000):
        extra = "" if idx == 0 else f"_{idx:02d}"
        candidate = path.parent / f"{stem}_{stamp}{extra}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a unique output path for {path}")


def _get_non_free_joint_qpos_addrs(model):
    out = []
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        out.append(int(model.jnt_qposadr[j]))
    return out


def _local_robot_xml_path(spec, morph_root):
    spec_xml = spec.source_xml
    if spec_xml.is_absolute():
        try:
            rel = spec_xml.relative_to(morph_root)
        except ValueError:
            rel = Path("assets") / "robots" / spec_xml.name
    else:
        rel = spec_xml

    parts = rel.parts
    if "robots" in parts:
        rel = Path(*parts[parts.index("robots") + 1:])
    local = LOCAL_ROBOT_ASSET_ROOT / rel
    if local.exists():
        return local.resolve()

    fallback = spec_xml if spec_xml.is_absolute() else (morph_root / spec_xml)
    return fallback.resolve()


def _floorless_xml_path(xml_path):
    xml_path = Path(xml_path).expanduser().resolve()
    out_path = xml_path.with_name(f"{xml_path.stem}_floorless{xml_path.suffix}")
    if out_path.exists() and out_path.stat().st_mtime >= xml_path.stat().st_mtime:
        return out_path

    tree = ET.parse(xml_path)
    root = tree.getroot()
    removed = 0
    for worldbody in root.findall("worldbody"):
        for geom in list(worldbody.findall("geom")):
            name = str(geom.get("name", "")).lower()
            gtype = str(geom.get("type", "")).lower()
            material = str(geom.get("material", "")).lower()
            if gtype == "plane" or "floor" in name or "ground" in name or "ground" in material:
                worldbody.remove(geom)
                removed += 1

    if removed == 0:
        return xml_path
    tree.write(out_path, encoding="utf-8", xml_declaration=False)
    return out_path.resolve()


def _load_viewer_model(spec, morph_root, *, floorless=True):
    xml_path = _local_robot_xml_path(spec, morph_root)
    if floorless:
        xml_path = _floorless_xml_path(xml_path)
    return mujoco.MjModel.from_xml_path(str(xml_path)), xml_path


def _extract_yaw_xyzw(quat_xyzw):
    x, y, z, w = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _extract_yaw_wxyz(quat_wxyz):
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _yaw_only_quat_xyzw(quat_xyzw):
    yaw = _extract_yaw_xyzw(quat_xyzw)
    half = 0.5 * yaw
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float32)


def _world_vel_to_local(lin_vel_world, yaw):
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    vx, vy, vz = np.asarray(lin_vel_world, dtype=np.float32).reshape(3)
    return np.array([c * vx + s * vy, -s * vx + c * vy, vz], dtype=np.float32)


def _quat_xyzw_to_wxyz(quat_xyzw):
    q = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


def _yaw_rate_xyzw(prev_quat, curr_quat, dt):
    prev_yaw = _extract_yaw_xyzw(prev_quat)
    curr_yaw = _extract_yaw_xyzw(curr_quat)
    dyaw = np.arctan2(np.sin(curr_yaw - prev_yaw), np.cos(curr_yaw - prev_yaw))
    return float(dyaw / max(float(dt), 1e-8))


def _body_ang_vel_xyzw(prev_quat, curr_quat, dt):
    prev_rot = Rotation.from_quat(np.asarray(prev_quat, dtype=np.float64).reshape(4))
    curr_rot = Rotation.from_quat(np.asarray(curr_quat, dtype=np.float64).reshape(4))
    rel = prev_rot.inv() * curr_rot
    return (rel.as_rotvec() / max(float(dt), 1e-8)).astype(np.float32)


def _integrate_body_ang_vel_xyzw(quat_xyzw, body_ang_vel, dt):
    rot = Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64).reshape(4))
    delta = Rotation.from_rotvec(np.asarray(body_ang_vel, dtype=np.float64).reshape(3) * max(float(dt), 1e-8))
    return (rot * delta).as_quat().astype(np.float32)


def _make_ref_frame(
    dof_pos,
    root_pos,
    root_rot_xyzw,
    prev_root_pos,
    prev_root_rot_xyzw,
    dt,
    quat_convention,
    num_joints,
):
    num_joints = int(num_joints)
    dof = np.asarray(dof_pos, dtype=np.float32).reshape(-1)[:num_joints]
    if dof.shape[0] < num_joints:
        dof = np.pad(dof, (0, num_joints - dof.shape[0])).astype(np.float32)

    root_pos = np.asarray(root_pos, dtype=np.float32).reshape(3)
    root_rot_xyzw = np.asarray(root_rot_xyzw, dtype=np.float32).reshape(4)
    if prev_root_pos is None or prev_root_rot_xyzw is None:
        lin_vel_local = np.zeros(3, dtype=np.float32)
        ang_vel_local = np.zeros(3, dtype=np.float32)
    else:
        lin_vel_world = (root_pos - np.asarray(prev_root_pos, dtype=np.float32).reshape(3)) / max(float(dt), 1e-8)
        yaw = _extract_yaw_xyzw(root_rot_xyzw)
        lin_vel_local = _world_vel_to_local(lin_vel_world, yaw)
        ang_vel_local = _body_ang_vel_xyzw(prev_root_rot_xyzw, root_rot_xyzw, dt)

    quat = root_rot_xyzw if quat_convention == "xyzw" else _quat_xyzw_to_wxyz(root_rot_xyzw)
    return np.concatenate([dof, lin_vel_local, ang_vel_local, quat.astype(np.float32)]).astype(np.float32)



class SmplPacketPublisher:
    DEFAULT_OFFSETS = (-2, -1, 0, 1)

    def __init__(self, endpoint, fps, offsets=None):
        import zmq

        self.endpoint = endpoint
        self.fps = float(fps)
        self.offsets = tuple(int(x) for x in (offsets if offsets is not None else self.DEFAULT_OFFSETS))
        if len(self.offsets) == 0:
            raise ValueError(f"Invalid SMPL publish offsets: {self.offsets}")
        self.latency_frames = max(0, max(self.offsets))
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(endpoint)
        self.frames = []
        self.seq = 0

    def append_and_publish(self, smpl_feat, world_vel=None):
        smpl_feat = np.asarray(smpl_feat, dtype=np.float32).reshape(69)
        pose_body = smpl_feat[:63].astype(np.float32)
        root_motion = np.array(
            [
                smpl_feat[65],
                smpl_feat[63],
                0.0 if world_vel is None else np.asarray(world_vel, dtype=np.float32).reshape(3)[2],
                smpl_feat[67],
            ],
            dtype=np.float32,
        )
        self.frames.append(
            {
                "smpl_features": smpl_feat.copy(),
                "pose_body": pose_body.copy(),
                "root_motion": root_motion.copy(),
            }
        )

        anchor = len(self.frames) - 1 - self.latency_frames
        if anchor < 0 or (anchor + min(self.offsets)) < 0:
            return None
        if (anchor + max(self.offsets)) >= len(self.frames):
            return None

        idxs = [anchor + offset for offset in self.offsets]
        pose_context = np.stack([self.frames[i]["pose_body"] for i in idxs], axis=0).astype(np.float32)
        features_context = np.stack([self.frames[i]["smpl_features"] for i in idxs], axis=0).astype(np.float32)
        root_motion_context = np.stack([self.frames[i]["root_motion"] for i in idxs], axis=0).astype(np.float32)
        packet = {
            "version": 1,
            "stream": "smpl_context",
            "seq": int(self.seq),
            "timestamp": float(time.time()),
            "fps": self.fps,
            "latency_frames": int(self.latency_frames),
            "context_offsets": list(self.offsets),
            "anchor_index": int(anchor),
            "pose_body_shape": [int(len(self.offsets)), 63],
            "smpl_features_shape": [int(len(self.offsets)), 69],
            "root_motion_shape": [int(len(self.offsets)), 4],
            "pose_body_context": pose_context.tolist(),
            "smpl_features_context": features_context.tolist(),
            "root_motion_context": root_motion_context.tolist(),
            "root_motion_command": self.frames[anchor]["root_motion"].tolist(),
            "valid": True,
        }
        self.socket.send_pyobj(packet)
        self.seq += 1
        if len(self.frames) > 128:
            keep = max(64, self.latency_frames + abs(min(self.offsets)) + max(self.offsets) + 8)
            self.frames = self.frames[-keep:]
        return packet

    def reset(self):
        self.frames.clear()
        self.seq = 0

    def close(self):
        self.socket.close(linger=0)


class RefPacketPublisher:
    DEFAULT_OFFSETS = (0, 1, 2, 5, 10)

    def __init__(self, endpoint, robot, fps, quat_convention="xyzw", offsets=None, num_joints=12, motion_mode=0):
        import zmq

        self.endpoint = endpoint
        self.robot = robot
        self.fps = float(fps)
        self.dt = 1.0 / max(self.fps, 1e-8)
        self.quat_convention = str(quat_convention)
        self.num_joints = int(num_joints)
        if self.num_joints <= 0:
            raise ValueError(f"Invalid publish ref joint count: {self.num_joints}")
        self.frame_dim = self.num_joints + 10
        self.motion_mode = int(motion_mode)
        self.offsets = tuple(int(x) for x in (offsets if offsets is not None else self.DEFAULT_OFFSETS))
        if len(self.offsets) == 0 or min(self.offsets) < 0:
            raise ValueError(f"Invalid publish ref offsets: {self.offsets}")
        self.latency_frames = max(self.offsets)
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(endpoint)
        self.frames = []
        self.seq = 0

    def set_motion_mode(self, motion_mode):
        self.motion_mode = int(motion_mode)

    def append_and_publish(
        self,
        dof_pos,
        root_pos,
        root_rot_xyzw,
        motion_mode=None,
        gripper_score=None,
        gripper_target_angle=None,
        gripper_bin=None,
    ):
        if motion_mode is not None:
            self.set_motion_mode(motion_mode)
        prev = self.frames[-1] if self.frames else None
        ref = _make_ref_frame(
            dof_pos=dof_pos,
            root_pos=root_pos,
            root_rot_xyzw=root_rot_xyzw,
            prev_root_pos=None if prev is None else prev["root_pos"],
            prev_root_rot_xyzw=None if prev is None else prev["root_rot"],
            dt=self.dt,
            quat_convention=self.quat_convention,
            num_joints=self.num_joints,
        )
        self.frames.append(
            {
                "ref": ref,
                "root_pos": np.asarray(root_pos, dtype=np.float32).reshape(3).copy(),
                "root_rot": np.asarray(root_rot_xyzw, dtype=np.float32).reshape(4).copy(),
            }
        )
        if len(self.frames) <= self.latency_frames:
            return None

        anchor = len(self.frames) - 1 - self.latency_frames
        refs = np.stack(
            [self.frames[anchor + offset]["ref"] for offset in self.offsets],
            axis=0,
        ).astype(np.float32)
        packet = {
            "version": 1,
            "seq": int(self.seq),
            "timestamp": float(time.time()),
            "robot": self.robot,
            "fps": self.fps,
            "latency_frames": int(self.latency_frames),
            "ref_offsets": list(self.offsets),
            "ref_shape": [int(len(self.offsets)), int(self.frame_dim)],
            "joint_count": int(self.num_joints),
            "frame_dim": int(self.frame_dim),
            "refs_dtype": "float32",
            "quat_convention": self.quat_convention,
            "motion_mode": int(self.motion_mode),
            # Keep this as plain Python data so subscribers in different
            # NumPy versions/envs do not fail unpickling numpy._core.
            "refs": refs.tolist(),
            "valid": True,
        }
        if gripper_score is not None:
            packet["gripper_score"] = float(gripper_score)
        if gripper_target_angle is not None:
            packet["gripper_target_angle"] = float(gripper_target_angle)
            packet["gripper_target"] = float(gripper_target_angle)
        if gripper_bin is not None:
            packet["gripper_bin"] = int(gripper_bin)
        self.socket.send_pyobj(packet)
        self.seq += 1
        if len(self.frames) > 64:
            self.frames = self.frames[-32:]
        return packet

    def reset(self):
        self.frames.clear()
        self.seq = 0

    def close(self):
        self.socket.close(linger=0)


def _hand_gripper_score(hand_landmarks):
    fingers = ((8, 5), (12, 9), (16, 13), (20, 17))
    wrist = np.array(
        [hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z],
        dtype=np.float32,
    )
    scores = []
    for tip_idx, mcp_idx in fingers:
        tip = np.array(
            [hand_landmarks[tip_idx].x, hand_landmarks[tip_idx].y, hand_landmarks[tip_idx].z],
            dtype=np.float32,
        )
        mcp = np.array(
            [hand_landmarks[mcp_idx].x, hand_landmarks[mcp_idx].y, hand_landmarks[mcp_idx].z],
            dtype=np.float32,
        )
        scores.append(np.linalg.norm(tip - wrist) / (np.linalg.norm(mcp - wrist) + 1.0e-6))
    raw = float(np.mean(scores))
    return float(np.clip((raw - 1.0) / 1.5, 0.0, 1.0))


class MediaPipeGripperEstimator:
    """Small hand-open score estimator for optional gripper packet fields."""

    def __init__(
        self,
        model_path,
        *,
        alpha=0.2,
        closed_score=0.08,
        open_score=0.65,
        bins=5,
        closed_angle=-0.05,
        open_angle=-1.15,
        hand="right",
        missing_timeout=0.0,
        debug=False,
    ):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

        model_path = Path(model_path).expanduser()
        if not model_path.is_absolute():
            model_path = REPO_ROOT / model_path
        if not model_path.is_file():
            raise FileNotFoundError(f"MediaPipe hand landmarker model not found: {model_path}")

        self.mp = mp
        self.HandLandmarker = HandLandmarker
        self.options = HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = HandLandmarker.create_from_options(self.options)
        self.alpha = float(alpha)
        self.closed_score = float(closed_score)
        self.open_score = float(open_score)
        self.bins = max(2, int(bins))
        self.closed_angle = float(closed_angle)
        self.open_angle = float(open_angle)
        self.hand = str(hand).lower()
        if self.hand not in {"right", "left", "any"}:
            raise ValueError("--gripper-hand must be right, left, or any.")
        self.missing_timeout = float(missing_timeout)
        self.debug = bool(debug)
        self.score = self.open_score
        self.last_seen_time = 0.0
        self._last_timestamp_ms = 0

    def close(self):
        self.landmarker.close()

    def _score_to_bin_and_target(self, score):
        normalized = (float(score) - self.closed_score) / max(self.open_score - self.closed_score, 1.0e-6)
        normalized = float(np.clip(normalized, 0.0, 1.0))
        bin_idx = int(round(normalized * (self.bins - 1)))
        quantized = bin_idx / float(self.bins - 1)
        target = (1.0 - quantized) * self.closed_angle + quantized * self.open_angle
        return bin_idx, float(target)

    def _select_hand_landmarks(self, result):
        if not result.hand_landmarks:
            return None
        if self.hand == "any" or not getattr(result, "handedness", None):
            return result.hand_landmarks[0]
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            label = ""
            if handedness:
                label = str(handedness[0].category_name).lower()
            if label == self.hand:
                return landmarks
        return None

    def update(self, frame_bgr):
        now = time.monotonic()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(int(now * 1000.0), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_landmarks = self._select_hand_landmarks(result)
        if hand_landmarks is not None:
            raw_score = _hand_gripper_score(hand_landmarks)
            self.score = self.alpha * raw_score + (1.0 - self.alpha) * self.score
            self.last_seen_time = now
        elif (now - self.last_seen_time) > self.missing_timeout:
            self.score = self.open_score
        bin_idx, target = self._score_to_bin_and_target(self.score)
        return {
            "score": float(self.score),
            "target_angle": float(target),
            "bin": int(bin_idx),
            "valid": hand_landmarks is not None,
        }


class CameraLocalRootIntegrator:
    """Integrate camera translation deltas as local planar robot motion."""

    def __init__(self, root_height=0.9, scale=1.0, yaw_offset_deg=0.0, include_z=False, frame="heading"):
        self.root_height = root_height
        self.scale = float(scale)
        self.yaw_offset = float(np.deg2rad(yaw_offset_deg))
        self.include_z = bool(include_z)
        self.frame = str(frame)
        self.reset()

    def reset(self):
        self.prev_cam = None
        self.root_pos = np.array(
            [0.0, 0.0, float(self.root_height if self.root_height is not None else 0.0)],
            dtype=np.float32,
        )
        self.last_debug = None

    def step(self, pred_cam_t, root_rot_wxyz):
        cam = np.asarray(pred_cam_t, dtype=np.float32).reshape(3)
        if self.prev_cam is None:
            self.prev_cam = cam.copy()
            self.last_debug = {
                "cam_x": float(cam[0]), "cam_y": float(cam[1]), "cam_z": float(cam[2]),
                "cam_dx": 0.0, "cam_dy": 0.0, "cam_dz": 0.0,
                "local_dx": 0.0, "local_dy": 0.0,
                "heading_yaw": 0.0,
                "world_dx": 0.0, "world_dy": 0.0, "world_dz": 0.0,
            }
            return self.root_pos.copy()

        delta = (cam - self.prev_cam) * self.scale
        self.prev_cam = cam.copy()

        # Camera x is image-right; camera z is depth/forward.
        # Student/GMR local convention is x-forward, y-left.
        local_dx = float(delta[2])
        local_dy = float(-delta[0])

        if abs(self.yaw_offset) > 1e-8:
            co = float(np.cos(self.yaw_offset))
            so = float(np.sin(self.yaw_offset))
            local_dx, local_dy = (
                co * local_dx - so * local_dy,
                so * local_dx + co * local_dy,
            )

        yaw = 0.0 if self.frame == "fixed" else _extract_yaw_wxyz(root_rot_wxyz)
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        world_delta = np.array(
            [
                c * local_dx - s * local_dy,
                s * local_dx + c * local_dy,
                float(delta[1]) if self.include_z else 0.0,
            ],
            dtype=np.float32,
        )
        self.root_pos = self.root_pos + world_delta
        if self.root_height is not None and not self.include_z:
            self.root_pos[2] = float(self.root_height)
        self.last_debug = {
            "cam_x": float(cam[0]), "cam_y": float(cam[1]), "cam_z": float(cam[2]),
            "cam_dx": float(delta[0]), "cam_dy": float(delta[1]), "cam_dz": float(delta[2]),
            "local_dx": float(local_dx), "local_dy": float(local_dy),
            "heading_yaw": float(yaw),
            "world_dx": float(world_delta[0]), "world_dy": float(world_delta[1]), "world_dz": float(world_delta[2]),
        }
        return self.root_pos.copy()


class StreamingSmplFeatureBuilder:
    """Causal builder for morph's 69D SMPL feature layout."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_trans = None
        self.prev_root_rot = None
        self.last_world_vel = np.zeros(3, dtype=np.float32)

    def step(self, pose_body, root_orient, trans, dt):
        pose_body = np.asarray(pose_body, dtype=np.float32).reshape(63)
        root_orient = np.asarray(root_orient, dtype=np.float32).reshape(3)
        trans = np.asarray(trans, dtype=np.float32).reshape(3)
        dt = max(float(dt), 1e-8)
        curr_rot = Rotation.from_rotvec(root_orient.astype(np.float64))
        if self.prev_trans is None or self.prev_root_rot is None:
            lin_vel_world = np.zeros(3, dtype=np.float32)
            lin_vel_local = np.zeros(3, dtype=np.float32)
            ang_vel_local = np.zeros(3, dtype=np.float32)
        else:
            lin_vel_world = (trans - self.prev_trans) / dt
            lin_vel_local = curr_rot.inv().apply(lin_vel_world.astype(np.float64)).astype(np.float32)
            rel = self.prev_root_rot.inv() * curr_rot
            ang_vel_local = (rel.as_rotvec() / dt).astype(np.float32)
        self.last_world_vel = lin_vel_world.astype(np.float32)
        self.prev_trans = trans.copy()
        self.prev_root_rot = curr_rot
        return np.concatenate([pose_body, lin_vel_local, ang_vel_local]).astype(np.float32)


class LiveStudentSMPLDirect:
    def __init__(self, args, src_fps):
        morph_root = Path(args.morph_root).expanduser().resolve()
        morph_src = morph_root / "src"
        if str(morph_src) not in sys.path:
            sys.path.insert(0, str(morph_src))
        from csmt.models.student_rt import StudentRT
        from csmt.pipelines.infer_student_smpl import _load_smpl_norm_stats, _resolve_stats_path
        from csmt.pipelines.infer_teacher import InferenceStats
        from csmt.robots.registry import load_robot_spec
        from csmt.tasks.registry import resolve_task_config
        from csmt.utils.smpl_features import SMPL_INPUT_DIM, root_motion_4d_from_smpl_features
        self.device = torch.device(args.student_device if torch.cuda.is_available() else "cpu")
        self.fps = float(src_fps)
        self.dt = 1.0 / max(self.fps, 1e-8)
        self.root_motion_mode = str(args.root_motion_mode)
        self.root_blend_alpha = float(np.clip(args.root_blend_alpha, 0.0, 1.0))
        self.heading_mode = str(args.heading_mode)
        self.smpl_input_linear_vel_mode = str(args.smpl_input_linear_vel_mode)
        self.smpl_input_angular_vel_mode = str(args.smpl_input_angular_vel_mode)
        self.smpl_low_std_threshold = float(args.smpl_low_std_threshold)
        self.smpl_root_map = str(args.smpl_root_map)
        self.root_motion_4d_from_smpl_features = root_motion_4d_from_smpl_features
        processed_root = Path(args.processed_dir).expanduser().resolve() if args.processed_dir else (morph_root / "data" / "processed").resolve()
        resolved = resolve_task_config(morph_root, args.task_family, args.pair_id)
        self.src_robot_id = "smpl"
        self.dst_robot_id = resolved.dst_robot
        dst_spec = load_robot_spec(morph_root / "configs" / "robots" / f"{self.dst_robot_id}.yaml")
        self.dst_stats = InferenceStats(str(_resolve_stats_path(self.dst_robot_id, processed_root)), njoints=dst_spec.njoints, nbodies=dst_spec.nbodies)
        self.dst_start_height = float(args.dst_start_height) if args.dst_start_height is not None else float(dst_spec.nominal_base_height if dst_spec.nominal_base_height is not None else 0.28)
        ckpt_path = Path(args.student_ckpt).expanduser().resolve()
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        cfg = ckpt.get("config", {})
        self.src_dim = int(cfg.get("src_dim", SMPL_INPUT_DIM))
        self.dst_dim = int(cfg.get("dst_dim", 0))
        self.hist_len = int(cfg.get("hist_len", 24))
        self.prev_len = int(cfg.get("prev_len", 2))
        if self.src_dim != SMPL_INPUT_DIM:
            raise ValueError(f"SMPL-direct student expects src_dim={SMPL_INPUT_DIM}, checkpoint has {self.src_dim}. Use a student_smpl checkpoint.")
        if self.dst_dim <= 0:
            raise RuntimeError("Invalid SMPL student checkpoint config: missing dst_dim")
        self.smpl_mean, self.smpl_std, self.smpl_stats_origin = _load_smpl_norm_stats(cfg, self.src_dim, ckpt_path, args.smpl_stats)
        self.smpl_low_std_mask = (
            self.smpl_std < self.smpl_low_std_threshold
            if self.smpl_low_std_threshold > 0.0
            else np.zeros_like(self.smpl_std, dtype=bool)
        )
        self.model = StudentRT(src_dim=self.src_dim, dst_dim=self.dst_dim, hist_len=self.hist_len, prev_len=self.prev_len, conv_channels=int(cfg.get("conv_channels", 192)), gru_hidden=int(cfg.get("gru_hidden", 384)), conv_kernel=int(cfg.get("conv_kernel", 3)), conv_dropout=float(cfg.get("conv_dropout", 0.1)), use_attn=bool(cfg.get("use_attn", True)), attn_heads=int(cfg.get("attn_heads", 4)), attn_dropout=float(cfg.get("attn_dropout", 0.1)), predict_residual=bool(cfg.get("predict_residual", False))).to(self.device)
        self.model.load_state_dict(ckpt.get("model", ckpt))
        self.model.eval()
        self.dst_root_start = int(self.dst_stats.njoints)
        self.dst_mean = self.dst_stats.mean.detach().cpu().numpy().astype(np.float32)
        self.dst_std = self.dst_stats.std.detach().cpu().numpy().astype(np.float32)
        self.dst_mean_root = self.dst_mean[self.dst_root_start:self.dst_root_start + 4]
        self.dst_std_root = self.dst_std[self.dst_root_start:self.dst_root_start + 4]
        self.reset_state()

    def reset_state(self):
        self.src_hist = deque(maxlen=self.hist_len)
        self.prev_out = deque(maxlen=max(1, self.prev_len))
        self.initialized = False
        self.yaw = 0.0
        self.root_pos = np.array([0.0, 0.0, self.dst_start_height], dtype=np.float32)
        self.last_debug = {}

    def step(self, smpl_feat, dt=None, source_yaw=None, smpl_world_vel=None):
        dt = self.dt if dt is None else max(float(dt), 1e-8)
        smpl_feat_raw = np.asarray(smpl_feat, dtype=np.float32).reshape(-1)
        smpl_feat = smpl_feat_raw.copy()
        if smpl_feat.shape[0] != self.src_dim:
            raise ValueError(f"SMPL-direct student expects src_dim={self.src_dim}, got {smpl_feat.shape[0]}")
        if self.smpl_input_linear_vel_mode == "zero":
            smpl_feat[63:66] = 0.0
        if self.smpl_input_angular_vel_mode == "zero":
            smpl_feat[66:69] = 0.0
        clamped_count = 0
        if np.any(self.smpl_low_std_mask):
            clamped_count = int(np.sum(self.smpl_low_std_mask))
            smpl_feat[self.smpl_low_std_mask] = self.smpl_mean[self.smpl_low_std_mask]
        smpl_feat_norm = ((smpl_feat - self.smpl_mean) / (self.smpl_std + 1e-8)).astype(np.float32)
        if not self.initialized:
            if self.heading_mode == "integrate" and source_yaw is not None:
                # Match morph SMPL offline inference, which uses the first SMPL root yaw as yaw_init.
                self.yaw = float(source_yaw)
            for _ in range(self.hist_len):
                self.src_hist.append(smpl_feat_norm.copy())
            zero_dst = np.zeros((self.dst_dim,), dtype=np.float32)
            for _ in range(max(1, self.prev_len)):
                self.prev_out.append(zero_dst.copy())
            self.initialized = True
        self.src_hist.append(smpl_feat_norm)
        x_hist = torch.from_numpy(np.stack(self.src_hist, axis=0)).unsqueeze(0).to(self.device)
        if self.prev_len > 0:
            y_prev = torch.from_numpy(np.stack(list(self.prev_out)[-self.prev_len:], axis=0)).unsqueeze(0).to(self.device)
        else:
            y_prev = torch.zeros(1, 0, self.dst_dim, device=self.device)
        with torch.no_grad():
            y_hat, _ = self.model(x_hist, y_prev)
        y_np = y_hat[0].detach().cpu().numpy().astype(np.float32)
        if self.root_motion_mode != "student":
            src_root_phys = self.root_motion_4d_from_smpl_features(smpl_feat.reshape(1, -1))[0]
            if self.smpl_root_map == "world_z" and smpl_world_vel is not None:
                src_root_phys[2] = float(np.asarray(smpl_world_vel, dtype=np.float32).reshape(3)[2])
            pred_root_phys = y_np[self.dst_root_start:self.dst_root_start + 4] * self.dst_std_root + self.dst_mean_root
            out_root_phys = src_root_phys if self.root_motion_mode == "source" else (1.0 - self.root_blend_alpha) * pred_root_phys + self.root_blend_alpha * src_root_phys
            y_np[self.dst_root_start:self.dst_root_start + 4] = (out_root_phys - self.dst_mean_root) / (self.dst_std_root + 1e-8)
        self.prev_out.append(y_np)
        y_phys = y_np * self.dst_std + self.dst_mean
        nj = int(self.dst_stats.njoints)
        dst_dof = y_phys[:nj].astype(np.float32)
        lin_vel_local = y_phys[nj:nj + 3].astype(np.float32)
        yaw_rate = float(y_phys[nj + 3])
        if self.heading_mode == "source" and source_yaw is not None:
            self.yaw = float(source_yaw)
        else:
            self.yaw = float(self.yaw + yaw_rate * dt)
        c = float(np.cos(self.yaw)); ss = float(np.sin(self.yaw))
        world_vel = np.array([c * lin_vel_local[0] - ss * lin_vel_local[1], ss * lin_vel_local[0] + c * lin_vel_local[1], lin_vel_local[2]], dtype=np.float32)
        self.root_pos = self.root_pos + world_vel * dt
        half_yaw = self.yaw * 0.5
        dst_root_rot = np.array([0.0, 0.0, np.sin(half_yaw), np.cos(half_yaw)], dtype=np.float32)
        self.last_debug = {
            "smpl_raw_lin_x": float(smpl_feat_raw[63]),
            "smpl_raw_lin_y": float(smpl_feat_raw[64]),
            "smpl_raw_lin_z": float(smpl_feat_raw[65]),
            "smpl_raw_ang_x": float(smpl_feat_raw[66]),
            "smpl_raw_ang_y": float(smpl_feat_raw[67]),
            "smpl_raw_ang_z": float(smpl_feat_raw[68]),
            "smpl_world_vz": float(np.asarray(smpl_world_vel, dtype=np.float32).reshape(3)[2]) if smpl_world_vel is not None else 0.0,
            "smpl_lin_x": float(smpl_feat[63]),
            "smpl_lin_y": float(smpl_feat[64]),
            "smpl_lin_z": float(smpl_feat[65]),
            "smpl_ang_x": float(smpl_feat[66]),
            "smpl_ang_y": float(smpl_feat[67]),
            "smpl_ang_z": float(smpl_feat[68]),
            "smpl_z_abs_max": float(np.max(np.abs(smpl_feat_norm))),
            "smpl_z_rms": float(np.sqrt(np.mean(np.square(smpl_feat_norm)))),
            "smpl_low_std_clamped": float(clamped_count),
            "pred_root_vx": float(lin_vel_local[0]),
            "pred_root_vy": float(lin_vel_local[1]),
            "pred_root_vz": float(lin_vel_local[2]),
            "pred_yaw_rate": float(yaw_rate),
        }
        return dst_dof, self.root_pos.copy(), dst_root_rot


class FinalTargetSmoother:
    def __init__(self, joint_alpha=0.0, root_alpha=0.0):
        self.joint_alpha = float(joint_alpha)
        self.root_alpha = float(root_alpha)
        self.reset()

    def reset(self):
        self.prev_dof = None
        self.prev_root_pos = None
        self.prev_root_rot = None

    @staticmethod
    def _smooth_quat_xyzw(prev_quat, curr_quat, alpha):
        if prev_quat is None or alpha <= 0.0:
            return np.asarray(curr_quat, dtype=np.float32).reshape(4)
        if alpha >= 1.0:
            return np.asarray(prev_quat, dtype=np.float32).reshape(4)
        prev_rot = Rotation.from_quat(np.asarray(prev_quat, dtype=np.float64).reshape(4))
        curr_rot = Rotation.from_quat(np.asarray(curr_quat, dtype=np.float64).reshape(4))
        delta = prev_rot.inv() * curr_rot
        smoothed = prev_rot * Rotation.from_rotvec((1.0 - alpha) * delta.as_rotvec())
        return smoothed.as_quat().astype(np.float32)

    def step(self, dof, root_pos, root_rot):
        dof = np.asarray(dof, dtype=np.float32).reshape(-1)
        root_pos = np.asarray(root_pos, dtype=np.float32).reshape(3)
        root_rot = np.asarray(root_rot, dtype=np.float32).reshape(4)

        if self.prev_dof is None or self.joint_alpha <= 0.0:
            out_dof = dof.copy()
        elif self.joint_alpha >= 1.0:
            out_dof = self.prev_dof.copy()
        else:
            out_dof = (self.joint_alpha * self.prev_dof + (1.0 - self.joint_alpha) * dof).astype(np.float32)

        if self.prev_root_pos is None or self.root_alpha <= 0.0:
            out_root_pos = root_pos.copy()
            out_root_rot = root_rot.copy()
        elif self.root_alpha >= 1.0:
            out_root_pos = self.prev_root_pos.copy()
            out_root_rot = self.prev_root_rot.copy()
        else:
            out_root_pos = (self.root_alpha * self.prev_root_pos + (1.0 - self.root_alpha) * root_pos).astype(np.float32)
            out_root_rot = self._smooth_quat_xyzw(self.prev_root_rot, root_rot, self.root_alpha)

        self.prev_dof = out_dof.copy()
        self.prev_root_pos = out_root_pos.copy()
        self.prev_root_rot = out_root_rot.copy()
        return out_dof, out_root_pos, out_root_rot


def _open_video_capture(video_arg):
    """Open a file path or a webcam index like '0'."""
    text = str(video_arg)
    if text.isdigit():
        source = int(text)
        is_camera = True
    else:
        source = text
        is_camera = False
    return cv2.VideoCapture(source), is_camera


class SplitRenderViewer:
    def __init__(
        self,
        src_model,
        src_data,
        dst_model,
        dst_data,
        width=640,
        height=480,
        left_width=0,
        render_width=640,
        render_height=480,
        show_camera=False,
    ):
        self.src_model = src_model
        self.src_data = src_data
        self.dst_model = dst_model
        self.dst_data = dst_data
        self.width = int(width)
        self.height = int(height)
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self.left_width = int(left_width) if int(left_width) > 0 else max(320, self.width // 2)
        self.show_camera = bool(show_camera)
        self.small_height = self.height // 2
        self.render_small_height = self.render_height // 2
        self.has_src = src_model is not None and src_data is not None
        if self.show_camera and self.has_src:
            self.window_name = "camera + source | Student target"
        elif self.show_camera:
            self.window_name = "camera | Student target"
        elif self.has_src:
            self.window_name = "source | Student target"
        else:
            self.window_name = "Student target"
        self.running = True
        src_height = self.render_small_height if self.show_camera else self.render_height
        src_width = min(self.left_width, self.render_width) if self.show_camera else self.render_width
        self.src_renderer = (
            mujoco.Renderer(src_model, height=src_height, width=src_width)
            if self.has_src else None
        )
        self.dst_renderer = mujoco.Renderer(dst_model, height=self.render_height, width=self.render_width)
        self.src_cam = self._make_camera(azimuth=-45)
        self.dst_cam = self._make_camera(azimuth=45)

    @staticmethod
    def _make_camera(azimuth):
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.distance = 2.8
        cam.azimuth = float(azimuth)
        cam.elevation = -20
        return cam

    def __enter__(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        window_width = self.width + (self.left_width if (self.show_camera or self.has_src) else 0)
        cv2.resizeWindow(self.window_name, window_width, self.height)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def is_running(self):
        return self.running

    def _render_one(self, renderer, data, cam, title):
        if data.model.nbody > 1:
            cam.lookat[:] = data.xpos[1]
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if title == "target" and data.model.nq >= 7:
            yaw = _extract_yaw_wxyz(data.qpos[3:7])
            img = self._draw_heading_overlay(img, yaw, cam.azimuth)
        cv2.putText(
            img,
            title,
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return img

    @staticmethod
    def _draw_heading_overlay(img, yaw, cam_azimuth_deg):
        # Small top-down compass: arrow is robot forward projected into the fixed viewer camera.
        h, w = img.shape[:2]
        center = (w - 72, 72)
        radius = 42
        cv2.circle(img, center, radius, (40, 40, 40), -1, cv2.LINE_AA)
        cv2.circle(img, center, radius, (210, 210, 210), 1, cv2.LINE_AA)
        cv2.line(img, (center[0] - radius + 8, center[1]), (center[0] + radius - 8, center[1]), (90, 90, 90), 1)
        cv2.line(img, (center[0], center[1] - radius + 8), (center[0], center[1] + radius - 8), (90, 90, 90), 1)

        screen_ang = float(yaw) - np.deg2rad(float(cam_azimuth_deg))
        end = (
            int(round(center[0] + np.cos(screen_ang) * (radius - 10))),
            int(round(center[1] - np.sin(screen_ang) * (radius - 10))),
        )
        cv2.arrowedLine(img, center, end, (0, 80, 255), 3, cv2.LINE_AA, tipLength=0.35)
        yaw_deg = (np.rad2deg(float(yaw)) + 180.0) % 360.0 - 180.0
        cv2.putText(
            img,
            f"{yaw_deg:+.0f}",
            (center[0] - 28, center[1] + radius + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return img

    @staticmethod
    def _fit_panel(img, width, height):
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        if img is None or img.size == 0:
            return panel
        h, w = img.shape[:2]
        scale = min(width / max(w, 1), height / max(h, 1))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x0 = (width - new_w) // 2
        y0 = (height - new_h) // 2
        panel[y0:y0 + new_h, x0:x0 + new_w] = resized
        return panel

    def _camera_panel(self, frame_bgr):
        panel_height = self.small_height if self.has_src else self.height
        if frame_bgr is None:
            img = np.zeros((panel_height, self.left_width, 3), dtype=np.uint8)
        else:
            img = self._fit_panel(frame_bgr, self.left_width, panel_height)
        cv2.putText(
            img,
            "camera",
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return img

    def sync(self, frame_bgr=None):
        src_img = None
        if self.has_src:
            src_img = self._render_one(self.src_renderer, self.src_data, self.src_cam, "source")
        dst_img = self._render_one(self.dst_renderer, self.dst_data, self.dst_cam, "target")
        if self.show_camera:
            if self.has_src:
                left_col = np.concatenate([self._camera_panel(frame_bgr), src_img], axis=0)
            else:
                left_col = self._camera_panel(frame_bgr)
            left_col = cv2.resize(
                left_col,
                (self.left_width, self.height),
                interpolation=cv2.INTER_LINEAR,
            )
        elif self.has_src:
            left_col = cv2.resize(src_img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        else:
            left_col = None
        dst_img = cv2.resize(dst_img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        combined = np.concatenate([left_col, dst_img], axis=1) if left_col is not None else dst_img
        cv2.imshow(self.window_name, combined)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            self.running = False
        try:
            visible = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            visible = 0
        if visible < 1:
            self.running = False

    def close(self):
        self.running = False
        try:
            if self.src_renderer is not None:
                self.src_renderer.close()
            self.dst_renderer.close()
        except Exception:
            pass
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass


class NullViewer:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    @staticmethod
    def is_running():
        return True

    @staticmethod
    def sync(*_args, **_kwargs):
        return None


class LiveStudentRT:
    def __init__(self, args, src_fps):
        morph_root = Path(args.morph_root).expanduser().resolve()
        morph_src = morph_root / "src"
        if str(morph_src) not in sys.path:
            sys.path.insert(0, str(morph_src))

        from csmt.models.student_rt import StudentRT
        from csmt.pipelines.infer_student_rt import InferenceStats, _resolve_stats_path
        from csmt.robots.registry import load_robot_spec
        from csmt.tasks.registry import resolve_task_config

        self.morph_root = morph_root
        self.device = torch.device(args.student_device if torch.cuda.is_available() else "cpu")
        self.fps = float(src_fps)
        self.dt = 1.0 / max(self.fps, 1e-8)
        self.root_motion_mode = args.root_motion_mode
        self.root_blend_alpha = float(np.clip(args.root_blend_alpha, 0.0, 1.0))
        self.heading_mode = args.heading_mode

        processed_root = (
            Path(args.processed_dir).expanduser().resolve()
            if args.processed_dir
            else (morph_root / "data" / "processed").resolve()
        )
        resolved = resolve_task_config(morph_root, args.task_family, args.pair_id)
        self.src_robot_id = resolved.src_robot
        self.dst_robot_id = resolved.dst_robot

        src_spec = load_robot_spec(morph_root / "configs" / "robots" / f"{self.src_robot_id}.yaml")
        dst_spec = load_robot_spec(morph_root / "configs" / "robots" / f"{self.dst_robot_id}.yaml")
        self.src_stats = InferenceStats(
            str(_resolve_stats_path(self.src_robot_id, processed_root)),
            njoints=src_spec.njoints,
            nbodies=src_spec.nbodies,
        )
        self.dst_stats = InferenceStats(
            str(_resolve_stats_path(self.dst_robot_id, processed_root)),
            njoints=dst_spec.njoints,
            nbodies=dst_spec.nbodies,
        )
        self.dst_start_height = (
            float(args.dst_start_height)
            if args.dst_start_height is not None
            else float(dst_spec.nominal_base_height if dst_spec.nominal_base_height is not None else 0.28)
        )

        ckpt = torch.load(args.student_ckpt, map_location="cpu")
        cfg = ckpt.get("config", {})
        self.src_dim = int(cfg.get("src_dim", 0))
        self.dst_dim = int(cfg.get("dst_dim", 0))
        self.hist_len = int(cfg.get("hist_len", 24))
        self.prev_len = int(cfg.get("prev_len", 2))
        if self.src_dim <= 0 or self.dst_dim <= 0:
            raise RuntimeError("Invalid student checkpoint config: missing src_dim/dst_dim")

        self.model = StudentRT(
            src_dim=self.src_dim,
            dst_dim=self.dst_dim,
            hist_len=self.hist_len,
            prev_len=self.prev_len,
            conv_channels=int(cfg.get("conv_channels", 128)),
            gru_hidden=int(cfg.get("gru_hidden", 256)),
            conv_kernel=int(cfg.get("conv_kernel", 3)),
            conv_dropout=float(cfg.get("conv_dropout", 0.1)),
            use_attn=bool(cfg.get("use_attn", False)),
            attn_heads=int(cfg.get("attn_heads", 4)),
            attn_dropout=float(cfg.get("attn_dropout", 0.1)),
            predict_residual=bool(cfg.get("predict_residual", False)),
        ).to(self.device)
        self.model.load_state_dict(ckpt.get("model", ckpt))
        self.model.eval()

        self.src_root_start = int(self.src_stats.njoints)
        self.dst_root_start = int(self.dst_stats.njoints)
        self.src_root_dim = int(getattr(self.src_stats, "root_dim", 3 + int(getattr(self.src_stats, "root_ang_dim", 1))))
        self.dst_root_dim = int(getattr(self.dst_stats, "root_dim", 3 + int(getattr(self.dst_stats, "root_ang_dim", 1))))
        self.src_root_ang_dim = int(getattr(self.src_stats, "root_ang_dim", max(1, self.src_root_dim - 3)))
        self.dst_root_ang_dim = int(getattr(self.dst_stats, "root_ang_dim", max(1, self.dst_root_dim - 3)))
        self.src_mean = self.src_stats.mean.detach().cpu().numpy().astype(np.float32)
        self.src_std = self.src_stats.std.detach().cpu().numpy().astype(np.float32)
        self.dst_mean = self.dst_stats.mean.detach().cpu().numpy().astype(np.float32)
        self.dst_std = self.dst_stats.std.detach().cpu().numpy().astype(np.float32)
        self.src_mean_root = self.src_mean[self.src_root_start:self.src_root_start + self.src_root_dim]
        self.src_std_root = self.src_std[self.src_root_start:self.src_root_start + self.src_root_dim]
        self.dst_mean_root = self.dst_mean[self.dst_root_start:self.dst_root_start + self.dst_root_dim]
        self.dst_std_root = self.dst_std[self.dst_root_start:self.dst_root_start + self.dst_root_dim]

        self.reset_state()

    def reset_state(self):
        self.src_hist = deque(maxlen=self.hist_len)
        self.prev_out = deque(maxlen=max(1, self.prev_len))
        self.initialized = False
        self.prev_root_pos = None
        self.prev_yaw = None
        self.prev_root_rot_xyzw = None
        self.yaw = None
        self.root_pos = np.array([0.0, 0.0, self.dst_start_height], dtype=np.float32)
        self.root_rot_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def _src_feature(self, dof_pos, root_pos, root_rot_xyzw, dt):
        dof_pos = np.asarray(dof_pos, dtype=np.float32).reshape(-1)
        root_pos = np.asarray(root_pos, dtype=np.float32).reshape(3)
        root_rot_xyzw = np.asarray(root_rot_xyzw, dtype=np.float32).reshape(4)
        yaw = _extract_yaw_xyzw(root_rot_xyzw)
        dt = max(float(dt), 1e-8)

        if self.prev_root_pos is None:
            lin_vel_world = np.zeros(3, dtype=np.float32)
            root_ang_rate = np.zeros((self.src_root_ang_dim,), dtype=np.float32)
        else:
            lin_vel_world = (root_pos - self.prev_root_pos) / dt
            if self.src_root_ang_dim >= 3:
                root_ang_rate = _body_ang_vel_xyzw(self.prev_root_rot_xyzw, root_rot_xyzw, dt)
                if self.src_root_ang_dim > 3:
                    root_ang_rate = np.pad(root_ang_rate, (0, self.src_root_ang_dim - 3)).astype(np.float32)
                elif self.src_root_ang_dim < 3:
                    root_ang_rate = root_ang_rate[:self.src_root_ang_dim].astype(np.float32)
            else:
                yaw_diff = np.arctan2(np.sin(yaw - self.prev_yaw), np.cos(yaw - self.prev_yaw))
                root_ang_rate = np.array([float(yaw_diff / dt)], dtype=np.float32)

        self.prev_root_pos = root_pos.copy()
        self.prev_yaw = yaw
        self.prev_root_rot_xyzw = root_rot_xyzw.copy()
        lin_vel_local = _world_vel_to_local(np.clip(lin_vel_world, -10.0, 10.0), yaw)
        src_phys = np.concatenate([dof_pos, lin_vel_local, root_ang_rate])
        if src_phys.shape[0] != self.src_dim:
            raise ValueError(f"student expects src_dim={self.src_dim}, got {src_phys.shape[0]}")
        src_norm = ((src_phys - self.src_mean) / (self.src_std + 1e-8)).astype(np.float32)
        return src_norm, yaw

    def step(self, dof_pos, root_pos, root_rot_xyzw, dt=None):
        dt = self.dt if dt is None else max(float(dt), 1e-8)
        src_norm, src_yaw = self._src_feature(dof_pos, root_pos, root_rot_xyzw, dt)
        if not self.initialized:
            for _ in range(self.hist_len):
                self.src_hist.append(src_norm.copy())
            zero_dst = np.zeros((self.dst_dim,), dtype=np.float32)
            for _ in range(max(1, self.prev_len)):
                self.prev_out.append(zero_dst.copy())
            self.yaw = src_yaw
            self.initialized = True

        self.src_hist.append(src_norm)
        x_hist = torch.from_numpy(np.stack(self.src_hist, axis=0)).unsqueeze(0).to(self.device)
        if self.prev_len > 0:
            y_prev_np = np.stack(list(self.prev_out)[-self.prev_len:], axis=0)
            y_prev = torch.from_numpy(y_prev_np).unsqueeze(0).to(self.device)
        else:
            y_prev = torch.zeros(1, 0, self.dst_dim, device=self.device)

        with torch.no_grad():
            y_hat, _ = self.model(x_hist, y_prev)
        y_np = y_hat[0].detach().cpu().numpy().astype(np.float32)

        if self.root_motion_mode != "student":
            src_root_phys = src_norm[self.src_root_start:self.src_root_start + self.src_root_dim] * self.src_std_root + self.src_mean_root
            pred_root_phys = y_np[self.dst_root_start:self.dst_root_start + self.dst_root_dim] * self.dst_std_root + self.dst_mean_root
            if src_root_phys.shape[0] != pred_root_phys.shape[0]:
                n = min(src_root_phys.shape[0], pred_root_phys.shape[0])
                src_root_phys = src_root_phys[:n]
                pred_root_phys = pred_root_phys[:n]
            if self.root_motion_mode == "source":
                out_root_phys = src_root_phys
            else:
                out_root_phys = (1.0 - self.root_blend_alpha) * pred_root_phys + self.root_blend_alpha * src_root_phys
            y_np[self.dst_root_start:self.dst_root_start + out_root_phys.shape[0]] = (
                out_root_phys - self.dst_mean_root[:out_root_phys.shape[0]]
            ) / (self.dst_std_root[:out_root_phys.shape[0]] + 1e-8)

        self.prev_out.append(y_np)
        y_phys = y_np * self.dst_std + self.dst_mean
        nj = int(self.dst_stats.njoints)
        dst_dof = y_phys[:nj].astype(np.float32)
        lin_vel_local = y_phys[nj:nj + 3].astype(np.float32)
        dst_root_ang = y_phys[nj + 3:nj + 3 + self.dst_root_ang_dim].astype(np.float32)

        if self.heading_mode == "source":
            self.yaw = float(src_yaw)
            half_yaw = self.yaw * 0.5
            self.root_rot_xyzw = np.array([0.0, 0.0, np.sin(half_yaw), np.cos(half_yaw)], dtype=np.float32)
        else:
            if self.dst_root_ang_dim >= 3:
                self.root_rot_xyzw = _integrate_body_ang_vel_xyzw(self.root_rot_xyzw, dst_root_ang[:3], dt)
                self.yaw = _extract_yaw_xyzw(self.root_rot_xyzw)
            else:
                yaw_rate = float(dst_root_ang[-1])
                self.yaw = float(self.yaw + yaw_rate * dt)
                half_yaw = self.yaw * 0.5
                self.root_rot_xyzw = np.array([0.0, 0.0, np.sin(half_yaw), np.cos(half_yaw)], dtype=np.float32)
        c = float(np.cos(self.yaw))
        s = float(np.sin(self.yaw))
        world_vel = np.array(
            [
                c * lin_vel_local[0] - s * lin_vel_local[1],
                s * lin_vel_local[0] + c * lin_vel_local[1],
                lin_vel_local[2],
            ],
            dtype=np.float32,
        )
        self.root_pos = self.root_pos + world_vel * dt
        dst_root_rot = self.root_rot_xyzw.astype(np.float32)
        return dst_dof, self.root_pos.copy(), dst_root_rot




def _load_dual_yuna_config(path):
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Dual Yuna config not found: {config_path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("--dual-yuna-config requires PyYAML in the video2morph environment") from exc
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    modes = cfg.get("modes")
    if not modes:
        raise ValueError("Dual Yuna config must contain a non-empty 'modes' section")
    normalized = []
    if isinstance(modes, dict):
        items = modes.items()
    elif isinstance(modes, list):
        items = []
        for idx, entry in enumerate(modes):
            if not isinstance(entry, dict):
                raise ValueError("Each mode in dual Yuna config must be a mapping")
            items.append((entry.get("name", f"mode_{idx}"), entry))
    else:
        raise ValueError("Dual Yuna config 'modes' must be a mapping or list")
    for name, mode_cfg in items:
        if not isinstance(mode_cfg, dict):
            raise ValueError(f"Mode {name!r} must be a mapping")
        cfg_copy = dict(mode_cfg)
        cfg_copy.setdefault("name", str(name))
        if "motion_mode" not in cfg_copy:
            raise ValueError(f"Mode {name!r} is missing required 'motion_mode'")
        cfg_copy["motion_mode"] = int(cfg_copy["motion_mode"])
        if cfg_copy["motion_mode"] not in (0, 1):
            raise ValueError(f"Mode {name!r} has unsupported motion_mode={cfg_copy['motion_mode']}; expected 0 or 1")
        normalized.append(cfg_copy)
    normalized.sort(key=lambda item: item["motion_mode"])
    return {
        "path": config_path,
        "start_mode": cfg.get("start_mode"),
        "switch_key": cfg.get("switch_key"),
        "modes": normalized,
    }


class DualYunaStudentManager:
    """Holds the loco and loco-manip Yuna students and exposes the active one."""

    RESERVED_KEYS = {"name", "motion_mode", "description"}

    def __init__(self, args, src_fps, student_cls, config_path):
        self.config = _load_dual_yuna_config(config_path)
        self.mode_order = []
        self.students = {}
        self.mode_names = {}
        self.switch_key = self.config.get("switch_key")
        first = None
        for mode_cfg in self.config["modes"]:
            motion_mode = int(mode_cfg["motion_mode"])
            mode_args = argparse.Namespace(**vars(args))
            for key, value in mode_cfg.items():
                normalized_key = str(key).replace("-", "_")
                if normalized_key in self.RESERVED_KEYS:
                    continue
                if hasattr(mode_args, normalized_key):
                    setattr(mode_args, normalized_key, value)
                else:
                    logger.warning(f"Ignoring unknown dual Yuna mode config key {key!r} for mode {mode_cfg['name']!r}")
            logger.info(
                f"Loading Yuna mode {mode_cfg['name']!r} (motion_mode={motion_mode}, "
                f"task_family={mode_args.task_family}, pair_id={mode_args.pair_id}, ckpt={mode_args.student_ckpt})"
            )
            student = student_cls(mode_args, src_fps=src_fps)
            if first is None:
                first = student
                self.src_robot_id = student.src_robot_id
                self.dst_robot_id = student.dst_robot_id
                self.src_stats = getattr(student, "src_stats", None)
                self.dst_stats = student.dst_stats
            else:
                if student.dst_robot_id != self.dst_robot_id:
                    raise ValueError("Dual Yuna modes must target the same destination robot")
                if int(student.dst_stats.njoints) != int(self.dst_stats.njoints):
                    raise ValueError("Dual Yuna modes must have the same destination joint count")
                if getattr(student, "src_robot_id", self.src_robot_id) != self.src_robot_id:
                    raise ValueError("Dual Yuna modes must use the same source robot")
            self.mode_order.append(motion_mode)
            self.students[motion_mode] = student
            self.mode_names[motion_mode] = str(mode_cfg["name"])
        if 0 not in self.students:
            raise ValueError("Dual Yuna config must include loco motion_mode 0")
        start_mode_cfg = self.config.get("start_mode")
        if start_mode_cfg is None:
            self.motion_mode = 0
        elif isinstance(start_mode_cfg, str) and not start_mode_cfg.strip().lstrip("+-").isdigit():
            matches = [mode for mode, name in self.mode_names.items() if name == start_mode_cfg]
            if not matches:
                raise ValueError(f"Unknown dual Yuna start_mode {start_mode_cfg!r}")
            self.motion_mode = matches[0]
        else:
            self.motion_mode = int(start_mode_cfg)
        if self.motion_mode not in self.students:
            raise ValueError(f"Dual Yuna start_mode {self.motion_mode} is not configured")
        logger.info(
            f"Dual Yuna pipeline ready. Starting in mode {self.motion_mode} "
            f"({self.mode_names[self.motion_mode]})."
        )

    @property
    def active_student(self):
        return self.students[self.motion_mode]

    @property
    def active_mode_name(self):
        return self.mode_names[self.motion_mode]

    def __getattr__(self, name):
        return getattr(self.active_student, name)

    def reset_state(self):
        for student in self.students.values():
            student.reset_state()

    def switch_to_next_mode(self):
        idx = self.mode_order.index(self.motion_mode)
        self.motion_mode = self.mode_order[(idx + 1) % len(self.mode_order)]
        self.active_student.reset_state()
        logger.info(f"Switched active Yuna student to mode {self.motion_mode} ({self.active_mode_name})")
        return self.motion_mode

    def step(self, *args, **kwargs):
        return self.active_student.step(*args, **kwargs)


class TerminalModeSwitcher:
    def __init__(self, key, enabled):
        self.key = str(key or "").strip()
        self.enabled = bool(enabled and self.key and sys.stdin.isatty())
        self.fd = None
        self.old_termios = None
        if self.enabled:
            self.fd = sys.stdin.fileno()
            try:
                self.old_termios = termios.tcgetattr(self.fd)
                tty.setcbreak(self.fd)
                atexit.register(self.close)
                logger.info(f"Press '{self.key}' to toggle Yuna mode.")
            except (termios.error, OSError, ValueError) as exc:
                self.enabled = False
                self.fd = None
                self.old_termios = None
                logger.warning(f"Terminal mode switching disabled: {exc}")
        elif enabled and self.key:
            logger.info("Terminal mode switching disabled because stdin is not interactive.")

    def close(self):
        if self.fd is not None and self.old_termios is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_termios)
            except (termios.error, OSError, ValueError):
                pass
            self.old_termios = None

    def poll(self):
        if not self.enabled or self.fd is None:
            return False
        try:
            readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        except (OSError, ValueError):
            return False
        if not readable:
            return False
        try:
            chars = os.read(self.fd, 32).decode(errors="ignore")
        except (BlockingIOError, OSError, UnicodeDecodeError):
            return False
        return self.key in chars

def _parse_ref_offsets(spec):
    offsets = []
    for raw in str(spec).split(","):
        raw = raw.strip()
        if raw:
            offsets.append(int(raw))
    if len(offsets) == 0:
        raise argparse.ArgumentTypeError("expected comma-separated offsets, e.g. 0,1")
    if any(x < 0 for x in offsets):
        raise argparse.ArgumentTypeError("reference offsets must be non-negative")
    if any(b < a for a, b in zip(offsets, offsets[1:])):
        raise argparse.ArgumentTypeError("reference offsets must be sorted ascending")
    return tuple(offsets)


def _parse_smpl_offsets(spec):
    offsets = []
    for raw in str(spec).split(','):
        raw = raw.strip()
        if raw:
            offsets.append(int(raw))
    if len(offsets) == 0:
        raise argparse.ArgumentTypeError("expected comma-separated offsets, e.g. -2,-1,0,1")
    if any(b < a for a, b in zip(offsets, offsets[1:])):
        raise argparse.ArgumentTypeError("SMPL context offsets must be sorted ascending")
    return tuple(offsets)


def parse_args():
    p = argparse.ArgumentParser(description="Video -> GMR G1 -> morph student_rt -> live target robot viewer.")
    p.add_argument("--video", required=True, help="Input video path or webcam index, e.g. 0")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--loop", action="store_true", help="Restart the video when it reaches the end.")
    p.add_argument("--stride", type=int, default=1, help="Read every Nth source video frame.")
    p.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True,
                   help="If enabled, skip source video frames when processing falls behind.")
    p.add_argument("--display-fps", type=float, default=0.0,
                   help="Viewer pacing FPS. 0 uses video_fps/stride.")
    p.add_argument("--output-pkl", default=None, help="Optional output target robot PKL saved after a clean run.")
    p.add_argument(
        "--autosave-pkl",
        default="output/live_go2_autosave.pkl",
        help="Save produced target frames here on normal exit or Ctrl+C. Use '' to disable.",
    )
    p.add_argument("--timing-csv", default=None)
    p.add_argument(
        "--motion-debug-csv",
        default=None,
        help="Optional CSV with camera/root translation debug vectors for coordinate-frame tests.",
    )
    p.add_argument("--publish-zmq", default=None, help="Optional ZMQ PUB endpoint for Go2/Yuna ref packets, e.g. tcp://127.0.0.1:5555")
    p.add_argument("--publish-smpl-zmq", default=None, help="Optional ZMQ PUB endpoint for live SMPL context packets, e.g. tcp://127.0.0.1:5560")
    p.add_argument(
        "--publish-quat-convention",
        choices=["xyzw", "wxyz"],
        default="xyzw",
        help="Quaternion convention for the 4 base quat values in published reference frames.",
    )
    p.add_argument(
        "--publish-ref-offsets",
        type=_parse_ref_offsets,
        default=RefPacketPublisher.DEFAULT_OFFSETS,
        help="Comma-separated reference frame offsets to publish. Default: 0,1,2,5,10. Use 0,1 for current+next.",
    )
    p.add_argument(
        "--publish-gripper-mediapipe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add MediaPipe hand gripper_score and binned gripper_target_angle to ref packets.",
    )
    p.add_argument(
        "--gripper-model",
        default=str(REPO_ROOT / "hand_landmarker.task"),
        help="MediaPipe hand_landmarker.task path for --publish-gripper-mediapipe.",
    )
    p.add_argument("--gripper-smooth-alpha", type=float, default=0.2, help="EMA alpha for gripper score.")
    p.add_argument("--gripper-closed-score", type=float, default=0.08, help="Score mapped to closed gripper.")
    p.add_argument("--gripper-open-score", type=float, default=0.65, help="Score mapped to fully open gripper.")
    p.add_argument("--gripper-bins", type=int, default=5, help="Number of semi-continuous gripper target bins.")
    p.add_argument("--gripper-closed-angle", type=float, default=-0.05, help="Closed gripper target angle.")
    p.add_argument("--gripper-open-angle", type=float, default=-1.15, help="Open gripper target angle.")
    p.add_argument(
        "--gripper-hand",
        choices=["right", "left", "any"],
        default="right",
        help="Which MediaPipe handedness label controls the gripper. Use any if camera mirroring makes labels unreliable.",
    )
    p.add_argument(
        "--gripper-missing-timeout",
        type=float,
        default=0.0,
        help="Grace seconds without the selected hand before gripper score falls back open. Default 0 opens immediately.",
    )
    p.add_argument(
        "--publish-smpl-offsets",
        type=_parse_smpl_offsets,
        default=SmplPacketPublisher.DEFAULT_OFFSETS,
        help="Comma-separated SMPL context offsets to publish. Default: -2,-1,0,1.",
    )

    p.add_argument("--gmr-root", default="/home/psyduck/Ritwik/GMR")
    p.add_argument("--src-robot", default="unitree_g1")
    p.add_argument("--morph-root", default="/home/psyduck/Ritwik/morph")
    p.add_argument("--processed-dir", default=None)
    p.add_argument("--dual-yuna-config", default=None, help="Optional YAML config with two Yuna students: motion_mode 0 loco and motion_mode 1 loco-manip.")
    p.add_argument("--mode-switch-key", default="m", help="When --dual-yuna-config is active, press this key then Enter to toggle modes. Use '' to disable.")
    p.add_argument("--task-family", default="manipulation")
    p.add_argument("--pair-id", default="g1_to_go2_with_d1")
    p.add_argument("--student-ckpt", default="/home/psyduck/Ritwik/morph/runs/student_rt_g1_go2_d1_v6/best.pt")
    p.add_argument("--student-input-mode", choices=["gmr", "smpl"], default="gmr", help="gmr: FastSAM->SMPLX FK->GMR source robot->student_rt. smpl: FastSAM->SMPL features->student_smpl directly.")
    p.add_argument("--smpl-stats", default=None, help="Optional smpl_input_stats.npz for --student-input-mode smpl.")
    p.add_argument(
        "--smpl-low-std-threshold",
        type=float,
        default=1e-3,
        help=(
            "For --student-input-mode smpl, clamp SMPL channels with train std below this threshold "
            "to the train mean before normalization. Use 0 to disable."
        ),
    )
    p.add_argument(
        "--smpl-input-linear-vel-mode",
        choices=["actual", "zero"],
        default="actual",
        help="For --student-input-mode smpl, optionally zero the 3 SMPL root-local linear velocity features before normalization.",
    )
    p.add_argument(
        "--smpl-input-angular-vel-mode",
        choices=["actual", "zero"],
        default="actual",
        help="For --student-input-mode smpl, optionally zero the 3 SMPL root-local angular velocity features before normalization.",
    )
    p.add_argument(
        "--smpl-root-map",
        choices=["local", "world_z"],
        default="local",
        help=(
            "For --student-input-mode smpl with --root-motion-mode source/blend, choose SMPL-to-robot root mapping. "
            "world_z keeps the legacy planar mapping but uses streaming world z velocity for robot vertical motion."
        ),
    )
    p.add_argument("--student-device", default="cuda:0")
    p.add_argument("--root-motion-mode", choices=["student", "source", "blend"], default="source")
    p.add_argument("--root-blend-alpha", type=float, default=0.7)
    p.add_argument(
        "--target-root-rotation-mode",
        choices=["student", "yaw"],
        default="student",
        help="student uses the target root quaternion as predicted; yaw strips target roll/pitch before smoothing, viewer, ZMQ, and autosave.",
    )
    p.add_argument(
        "--heading-mode",
        choices=["integrate", "source"],
        default="source",
        help="Use integrated student/source yaw-rate, or directly mirror the current G1 source heading.",
    )
    p.add_argument("--dst-start-height", type=float, default=None)
    p.add_argument(
        "--show-src-viewer",
        action="store_true",
        help="Render G1 and target side by side in one OpenCV window.",
    )
    p.add_argument(
        "--show-camera-panel",
        action="store_true",
        help="Include the input camera/video frame as a third panel in the split view.",
    )
    p.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run headless: no MuJoCo/OpenCV viewer. Processing, ZMQ publishing, timing, and autosave still run.",
    )
    p.add_argument(
        "--viewer-ground",
        action="store_true",
        help="Render source/target viewer XMLs with their ground plane. Default removes the floor for local-frame live viewing.",
    )
    p.add_argument("--viewer-width", type=int, default=1280, help="Target render panel width for split view.")
    p.add_argument("--viewer-height", type=int, default=960, help="Total render panel height for split view.")
    p.add_argument("--render-width", type=int, default=640, help="Internal MuJoCo offscreen render width.")
    p.add_argument("--render-height", type=int, default=480, help="Internal MuJoCo offscreen render height.")
    p.add_argument(
        "--left-panel-width",
        type=int,
        default=0,
        help="Width for the stacked camera/G1 column. 0 uses half of --viewer-width.",
    )

    p.add_argument("--smplx-model-dir", default="body_models")
    p.add_argument("--smpl-model-path", default="body_models/smpl/SMPL_NEUTRAL.pkl")
    p.add_argument("--nn-model-dir", default="mhr2smpl/experiments/multiview_n30000_e500")
    p.add_argument("--mhr2smpl-mapping-path", default="mhr2smpl/data/mhr2smpl_mapping.npz")
    p.add_argument("--mhr-mesh-path", default="mhr2smpl/data/mhr_face_mask.ply")
    p.add_argument("--smoother-dir", default=None)
    p.add_argument("--yolo-model", default="checkpoints/yolo/yolo11m-pose.engine")
    p.add_argument("--image-size", type=int, default=384, choices=[256, 384, 512])
    p.add_argument("--camera-intrinsics", choices=["estimate", "default"], default="default")
    p.add_argument("--fastsam-inference-type", choices=["full", "body"], default="body")

    p.add_argument("--root-mode", choices=["zero", "camera"], default="camera")
    p.add_argument(
        "--root-translation-mode",
        choices=["legacy", "camera_delta_flat", "camera_local_flat", "camera_local", "camera_delta", "zero"],
        default="camera_local_flat",
        help=(
            "Map FastSAM pred_cam_t to MuJoCo/GMR root_pos. "
            "camera_local_flat uses camera depth/right as local forward/left motion; "
            "camera_local also includes camera vertical as root Z."
        ),
    )
    p.add_argument("--root-translation-scale", type=float, default=1.0)
    p.add_argument(
        "--root-translation-yaw-offset",
        type=float,
        default=0.0,
        help="Rotate camera-derived local planar root motion by this many degrees before applying G1 heading.",
    )
    p.add_argument(
        "--root-translation-frame",
        choices=["heading", "fixed"],
        default="heading",
        help=(
            "heading preserves the original behavior and rotates camera_local translation by robot/body heading; "
            "fixed maps camera deltas into a stable world frame. For camera walking tests use fixed with yaw offset -90."
        ),
    )
    p.add_argument("--root-orient-mode", choices=["zero", "fastsam", "raw_zyx"], default="raw_zyx")
    p.add_argument("--root-correction", default="x90", help="Root orientation correction for the GMR/G1 path. Default x90 matches the existing GMR convention.")
    p.add_argument(
        "--smpl-direct-root-correction",
        default=None,
        help=(
            "Root orientation correction for --student-input-mode smpl. "
            "Defaults to --root-correction, preserving the historical live behavior. "
            "Use 'none' or 'identity' to match uncorrected morph SMPL inference/training."
        ),
    )
    p.add_argument("--root-height", type=float, default=0.9)
    p.add_argument("--pose-smooth-alpha", type=float, default=0.65)
    p.add_argument("--root-smooth-alpha", type=float, default=0.5)
    p.add_argument(
        "--target-joint-smooth-alpha",
        type=float,
        default=0.0,
        help="Causal EMA on final target robot joint angles before viewer/pkl/ZMQ. 0 disables; try 0.2-0.6.",
    )
    p.add_argument(
        "--target-root-smooth-alpha",
        type=float,
        default=0.0,
        help="Optional causal smoothing on final target root position/quaternion. 0 disables.",
    )
    p.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = parse_args()
    configure_backbone_trt_for_image_size(args.image_size)

    repo_root = Path(__file__).resolve().parents[1]
    gmr_root = Path(args.gmr_root).expanduser().resolve()
    morph_root = Path(args.morph_root).expanduser().resolve()
    for path in (gmr_root, morph_root / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from mocap.core.multiview_mhr2smpl import MultiViewFusionRunner
    from mocap.core.setup_estimator import build_default_estimator
    from csmt.robots.registry import load_robot_spec

    cap, is_camera = _open_video_capture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(src_fps) or src_fps <= 0:
        src_fps = 30.0
    effective_fps = src_fps / max(int(args.stride), 1)
    display_fps = float(args.display_fps) if args.display_fps > 0 else effective_fps
    display_dt = 1.0 / max(display_fps, 1e-8)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if is_camera or total_frames <= 0:
        total_frames = 0
    if args.max_frames > 0:
        target_frames = int(args.max_frames)
    elif total_frames > 0:
        target_frames = total_frames
    else:
        target_frames = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    smpl_direct = args.student_input_mode == "smpl"
    root_correction_spec = (
        args.smpl_direct_root_correction
        if smpl_direct and args.smpl_direct_root_correction is not None
        else args.root_correction
    )
    if str(root_correction_spec).lower() in ("", "none", "identity", "zero"):
        root_correction_spec = ""
    root_correction = parse_root_correction(root_correction_spec)
    if not smpl_direct:
        from general_motion_retargeting import GeneralMotionRetargeting as GMR
    load_label = "FastSAM, mhr2smpl, and student_smpl models" if smpl_direct else "FastSAM, GMR, and student_rt models"
    logger.info(f"Loading {load_label}")
    with suppress_output(args.quiet):
        estimator = build_default_estimator(image_size=args.image_size, yolo_model_path=args.yolo_model)
        if args.camera_intrinsics == "default":
            estimator.fov_estimator = None
        fusion_runner = MultiViewFusionRunner(
            smpl_model_path=args.smpl_model_path,
            model_dir=args.nn_model_dir,
            mapping_path=args.mhr2smpl_mapping_path,
            mhr_mesh_path=args.mhr_mesh_path,
            smoother_dir=args.smoother_dir,
        )
        smplx_adapter = None
        retargeter = None
        if smpl_direct:
            if args.dual_yuna_config:
                student = DualYunaStudentManager(args, effective_fps, LiveStudentSMPLDirect, args.dual_yuna_config)
            else:
                student = LiveStudentSMPLDirect(args, src_fps=effective_fps)
        else:
            smplx_adapter = SmplxFrameAdapter(args.smplx_model_dir, device=device)
            retargeter = GMR(src_human="smplx", tgt_robot=args.src_robot, actual_human_height=1.75, verbose=False)
            if args.dual_yuna_config:
                student = DualYunaStudentManager(args, effective_fps, LiveStudentRT, args.dual_yuna_config)
            else:
                student = LiveStudentRT(args, src_fps=effective_fps)

    ref_publisher = None
    if args.publish_zmq:
        ref_publisher = RefPacketPublisher(
            endpoint=args.publish_zmq,
            robot=student.dst_robot_id,
            fps=display_fps,
            quat_convention=args.publish_quat_convention,
            offsets=args.publish_ref_offsets,
            num_joints=int(student.dst_stats.njoints),
            motion_mode=int(getattr(student, "motion_mode", 0)),
        )
        logger.info(
            f"Publishing delayed ref packets to {args.publish_zmq} "
            f"(robot={student.dst_robot_id}, joints={int(student.dst_stats.njoints)}, "
            f"frame_dim={int(student.dst_stats.njoints) + 10}, "
            f"quat={args.publish_quat_convention}, offsets={list(args.publish_ref_offsets)}, "
            f"motion_mode={int(getattr(student, 'motion_mode', 0))})"
        )

    smpl_publisher = None
    if args.publish_smpl_zmq:
        smpl_publisher = SmplPacketPublisher(
            endpoint=args.publish_smpl_zmq,
            fps=display_fps,
            offsets=args.publish_smpl_offsets,
        )
        logger.info(
            f"Publishing delayed SMPL context packets to {args.publish_smpl_zmq} "
            f"(offsets={list(args.publish_smpl_offsets)})"
        )

    gripper_estimator = None
    if args.publish_gripper_mediapipe:
        gripper_estimator = MediaPipeGripperEstimator(
            args.gripper_model,
            alpha=args.gripper_smooth_alpha,
            closed_score=args.gripper_closed_score,
            open_score=args.gripper_open_score,
            bins=args.gripper_bins,
            closed_angle=args.gripper_closed_angle,
            open_angle=args.gripper_open_angle,
            hand=args.gripper_hand,
            missing_timeout=args.gripper_missing_timeout,
            debug=not args.quiet,
        )
        logger.info(
            "Publishing MediaPipe gripper fields "
            f"(closed_score={args.gripper_closed_score:.3f}, open_score={args.gripper_open_score:.3f}, "
            f"bins={args.gripper_bins}, closed_angle={args.gripper_closed_angle:.3f}, "
            f"open_angle={args.gripper_open_angle:.3f}, hand={args.gripper_hand})"
        )

    dst_spec = load_robot_spec(morph_root / "configs" / "robots" / f"{student.dst_robot_id}.yaml")
    model = data = None
    has_free_base = False
    joint_qpos = []
    if not args.no_viewer:
        model, xml_path = _load_viewer_model(dst_spec, morph_root, floorless=not args.viewer_ground)
        logger.info(f"Target viewer XML: {xml_path}")
        data = mujoco.MjData(model)
        model.opt.gravity[:] = 0.0
        has_free_base = model.njnt > 0 and int(model.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE) and model.nq >= 7
        joint_qpos = _get_non_free_joint_qpos_addrs(model)
    map_dim = min(len(joint_qpos), int(student.dst_stats.njoints))

    src_model = src_data = None
    src_has_free_base = False
    src_joint_qpos = []
    src_map_dim = 0
    show_src_viewer = bool(args.show_src_viewer and not smpl_direct and not args.no_viewer)
    if args.show_src_viewer and smpl_direct:
        logger.info("Ignoring --show-src-viewer in --student-input-mode smpl because G1/GMR is skipped.")
    if args.show_src_viewer and args.no_viewer:
        logger.info("Ignoring --show-src-viewer because --no-viewer is set.")
    if show_src_viewer:
        src_spec = load_robot_spec(morph_root / "configs" / "robots" / f"{student.src_robot_id}.yaml")
        src_model, src_xml_path = _load_viewer_model(src_spec, morph_root, floorless=not args.viewer_ground)
        logger.info(f"Source viewer XML: {src_xml_path}")
        src_data = mujoco.MjData(src_model)
        src_model.opt.gravity[:] = 0.0
        src_has_free_base = (
            src_model.njnt > 0
            and int(src_model.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE)
            and src_model.nq >= 7
        )
        src_joint_qpos = _get_non_free_joint_qpos_addrs(src_model)
        src_map_dim = min(len(src_joint_qpos), int(student.src_stats.njoints))

    out_dof, out_root_pos, out_root_rot = [], [], []
    saved_pkl_paths = set()

    def save_target_pkl(path, reason):
        if not path or len(out_dof) == 0:
            return
        out_path = Path(path).expanduser()
        out_key = str(out_path.resolve()) if out_path.is_absolute() else str((Path.cwd() / out_path).resolve())
        if out_key in saved_pkl_paths:
            return
        payload = {
            "fps": float(display_fps),
            "dof_pos": np.asarray(out_dof, dtype=np.float32),
            "root_pos": np.asarray(out_root_pos, dtype=np.float32),
            "root_rot": np.asarray(out_root_rot, dtype=np.float32),
            "local_body_pos": None,
            "link_body_list": None,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f)
        os.replace(tmp_path, out_path)
        saved_pkl_paths.add(out_key)
        logger.success(f"Saved target robot pkl to {out_path} ({reason}, frames={len(out_dof)})")

    autosave_path = str(args.autosave_pkl) if args.autosave_pkl is not None else ""
    if autosave_path:
        atexit.register(lambda: save_target_pkl(autosave_path, "process exit"))

    timing_rows = []
    motion_debug_rows = []
    saved_csv_paths = set()

    def save_rows_csv(path, rows, label, reason):
        if not path or not rows:
            return
        requested_path = Path(path).expanduser()
        requested_key = str(requested_path.resolve()) if requested_path.is_absolute() else str((Path.cwd() / requested_path).resolve())
        if requested_key in saved_csv_paths:
            return
        out_path = _unique_output_path(requested_path)
        out_key = str(out_path.resolve()) if out_path.is_absolute() else str((Path.cwd() / out_path).resolve())
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, out_path)
        saved_csv_paths.add(requested_key)
        saved_csv_paths.add(out_key)
        logger.success(f"Saved {label} CSV to {out_path} ({reason}, rows={len(rows)})")

    if args.timing_csv:
        atexit.register(lambda: save_rows_csv(args.timing_csv, timing_rows, "timing", "process exit"))
    if args.motion_debug_csv:
        atexit.register(lambda: save_rows_csv(args.motion_debug_csv, motion_debug_rows, "motion debug", "process exit"))
    prev_debug_dst_root_pos = None
    skipped = 0
    prev_pose_body = None
    prev_root_orient = None
    root_translation_mapper = RootTranslationMapper(
        mode=(
            "zero"
            if args.root_mode == "zero" or args.root_translation_mode in ("camera_local_flat", "camera_local")
            else args.root_translation_mode
        ),
        root_height=args.root_height,
        scale=args.root_translation_scale,
    )
    camera_local_root = CameraLocalRootIntegrator(
        root_height=args.root_height,
        scale=args.root_translation_scale,
        yaw_offset_deg=args.root_translation_yaw_offset,
        include_z=(args.root_translation_mode == "camera_local"),
        frame=args.root_translation_frame,
    )
    smpl_feature_builder = StreamingSmplFeatureBuilder()
    target_smoother = FinalTargetSmoother(
        joint_alpha=args.target_joint_smooth_alpha,
        root_alpha=args.target_root_smooth_alpha,
    )
    frame_idx = 0
    produced = 0
    last_processed_frame_idx = None
    next_video_time = time.perf_counter()
    summary_logged = set()

    def log_final_summary(reason):
        if "summary" in summary_logged:
            return
        summary_logged.add("summary")
        if timing_rows:
            rows = timing_rows[2:] or timing_rows
            mean_total = float(np.mean([row["total_ms"] for row in rows]))
            logger.info(
                f"Produced {produced} frames, skipped detections={skipped}, "
                f"steady loop={1000.0 / max(mean_total, 1e-6):.1f}fps ({reason})"
            )
        else:
            logger.info(f"Produced {produced} frames, skipped detections={skipped}, steady loop=n/a ({reason})")

    atexit.register(lambda: log_final_summary("process exit"))

    def apply_dst_frame(dof, root_pos, root_rot):
        if model is None or data is None:
            return
        if has_free_base:
            data.qpos[0:3] = root_pos
            data.qpos[3:7] = _to_wxyz(root_rot)
        for i in range(map_dim):
            data.qpos[joint_qpos[i]] = dof[i]
        mujoco.mj_forward(model, data)

    def apply_src_frame(qpos):
        if src_model is None or src_data is None:
            return
        if src_has_free_base:
            src_data.qpos[0:3] = qpos[:3]
            # GMR qpos already stores MuJoCo quaternions as wxyz.
            src_data.qpos[3:7] = qpos[3:7]
        src_dof = qpos[7:]
        for i in range(src_map_dim):
            src_data.qpos[src_joint_qpos[i]] = src_dof[i]
        mujoco.mj_forward(src_model, src_data)

    pbar = tqdm(
        total=(target_frames if target_frames > 0 else None),
        desc=("Video -> SMPL -> student" if smpl_direct else "Video -> GMR -> student"),
        unit="src_frame",
    )
    mode_switch_key = student.switch_key if hasattr(student, "switch_key") and student.switch_key is not None else args.mode_switch_key
    mode_switcher = TerminalModeSwitcher(
        mode_switch_key,
        enabled=hasattr(student, "switch_to_next_mode"),
    )
    gripper_state = None
    with contextlib.ExitStack() as stack:
        if gripper_estimator is not None:
            stack.callback(gripper_estimator.close)
        if args.no_viewer:
            use_split_viewer = False
            viewer = stack.enter_context(NullViewer())
        else:
            use_split_viewer = bool(show_src_viewer or args.show_camera_panel or smpl_direct)
        if not args.no_viewer and use_split_viewer:
            viewer = stack.enter_context(
                SplitRenderViewer(
                    src_model if show_src_viewer else None,
                    src_data if show_src_viewer else None,
                    model,
                    data,
                    width=args.viewer_width,
                    height=args.viewer_height,
                    left_width=args.left_panel_width,
                    render_width=args.render_width,
                    render_height=args.render_height,
                    show_camera=args.show_camera_panel,
                )
            )
        elif not args.no_viewer:
            viewer = stack.enter_context(mujoco.viewer.launch_passive(model, data))
            viewer.cam.distance = 2.8
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -20

        while viewer.is_running():
            if mode_switcher.poll():
                old_mode = int(getattr(student, "motion_mode", 0))
                new_mode = int(student.switch_to_next_mode())
                target_smoother.reset()
                prev_debug_dst_root_pos = None
                if ref_publisher is not None:
                    ref_publisher.set_motion_mode(new_mode)
                    ref_publisher.reset()
                logger.info(f"Yuna mode trigger: {old_mode} -> {new_mode}. Ref packet history reset for clean live transition.")

            if target_frames > 0 and frame_idx >= target_frames:
                if not args.loop:
                    break
                if is_camera:
                    frame_idx = 0
                    last_processed_frame_idx = None
                    prev_pose_body = None
                    prev_root_orient = None
                    root_translation_mapper.reset()
                    camera_local_root.reset()
                    smpl_feature_builder.reset()
                    student.reset_state()
                    target_smoother.reset()
                    prev_debug_dst_root_pos = None
                    if ref_publisher is not None:
                        ref_publisher.reset()
                    if smpl_publisher is not None:
                        smpl_publisher.reset()
                    next_video_time = time.perf_counter()
                    pbar.reset(total=target_frames)
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                last_processed_frame_idx = None
                prev_pose_body = None
                prev_root_orient = None
                root_translation_mapper.reset()
                camera_local_root.reset()
                smpl_feature_builder.reset()
                student.reset_state()
                target_smoother.reset()
                prev_debug_dst_root_pos = None
                if ref_publisher is not None:
                    ref_publisher.reset()
                if smpl_publisher is not None:
                    smpl_publisher.reset()
                next_video_time = time.perf_counter()
                pbar.reset(total=target_frames)

            loop_start = time.perf_counter()
            read_start = time.perf_counter()
            ok, frame_bgr = cap.read()
            read_ms = (time.perf_counter() - read_start) * 1000.0
            if not ok:
                if is_camera or not args.loop:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                last_processed_frame_idx = None
                prev_pose_body = None
                prev_root_orient = None
                root_translation_mapper.reset()
                camera_local_root.reset()
                smpl_feature_builder.reset()
                student.reset_state()
                target_smoother.reset()
                prev_debug_dst_root_pos = None
                if ref_publisher is not None:
                    ref_publisher.reset()
                if smpl_publisher is not None:
                    smpl_publisher.reset()
                next_video_time = time.perf_counter()
                pbar.reset(total=target_frames)
                continue
            pbar.update(1)

            if frame_idx % max(int(args.stride), 1) != 0:
                frame_idx += 1
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if gripper_estimator is not None:
                gripper_state = gripper_estimator.update(frame_bgr)
            sync_cuda()
            fastsam_start = time.perf_counter()
            with suppress_output(args.quiet), torch.no_grad():
                outputs = estimator.process_one_image(
                    frame_rgb,
                    hand_box_source="yolo_pose",
                    inference_type=args.fastsam_inference_type,
                )
            sync_cuda()
            fastsam_ms = (time.perf_counter() - fastsam_start) * 1000.0

            out = choose_person(outputs)
            if out is None:
                skipped += 1
                frame_idx += 1
                continue

            sync_cuda()
            mhr_start = time.perf_counter()
            smpl_pose, _canonical_joints, betas, _weights = fusion_runner.infer(
                [(out["pred_vertices"], out["pred_cam_t"])]
            )
            sync_cuda()
            mhr2smpl_ms = (time.perf_counter() - mhr_start) * 1000.0

            smooth_start = time.perf_counter()
            pose_body = np.asarray(smpl_pose, dtype=np.float32).reshape(63)
            root_orient = root_orient_from_output(out, args.root_orient_mode, root_correction)
            pose_body = smooth_pose_body(prev_pose_body, pose_body, args.pose_smooth_alpha)
            root_orient = smooth_rotvec(prev_root_orient, root_orient, args.root_smooth_alpha)
            prev_pose_body = pose_body.copy()
            prev_root_orient = root_orient.copy()
            trans = root_translation_mapper.map(out["pred_cam_t"])
            smooth_ms = (time.perf_counter() - smooth_start) * 1000.0

            if last_processed_frame_idx is None:
                frame_dt = max(int(args.stride), 1) / max(src_fps, 1e-8)
            else:
                frame_dt = max(frame_idx - last_processed_frame_idx, 1) / max(src_fps, 1e-8)
            last_processed_frame_idx = frame_idx

            if smpl_direct:
                smplx_fk_ms = 0.0
                gmr_ms = 0.0
                if args.root_translation_mode in ("camera_local_flat", "camera_local"):
                    trans = camera_local_root.step(out["pred_cam_t"], _rotvec_to_wxyz(root_orient))
                smpl_feat = smpl_feature_builder.step(pose_body, root_orient, trans, frame_dt)
                smpl_world_vel = smpl_feature_builder.last_world_vel.copy()
                source_yaw = float(Rotation.from_rotvec(root_orient.astype(np.float64)).as_euler("ZYX")[0])
                qpos = None
                student_start = time.perf_counter()
                if smpl_publisher is not None:
                    smpl_publisher.append_and_publish(smpl_feat, smpl_world_vel)
                dst_dof, dst_root_pos, dst_root_rot = student.step(
                    smpl_feat,
                    dt=frame_dt,
                    source_yaw=source_yaw,
                    smpl_world_vel=smpl_world_vel,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                student_ms = (time.perf_counter() - student_start) * 1000.0
            else:
                sync_cuda()
                fk_start = time.perf_counter()
                smplx_frame = smplx_adapter.to_gmr_frame(root_orient, pose_body, trans, betas)
                sync_cuda()
                smplx_fk_ms = (time.perf_counter() - fk_start) * 1000.0

                gmr_start = time.perf_counter()
                qpos = retargeter.retarget(smplx_frame)
                if args.root_translation_mode in ("camera_local_flat", "camera_local"):
                    qpos[:3] = camera_local_root.step(out["pred_cam_t"], qpos[3:7])
                gmr_ms = (time.perf_counter() - gmr_start) * 1000.0

                student_start = time.perf_counter()
                g1_root_rot_xyzw = _wxyz_to_xyzw(qpos[3:7])
                dst_dof, dst_root_pos, dst_root_rot = student.step(
                    qpos[7:], qpos[:3], g1_root_rot_xyzw, dt=frame_dt
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                student_ms = (time.perf_counter() - student_start) * 1000.0

            if args.target_root_rotation_mode == "yaw":
                dst_root_rot = _yaw_only_quat_xyzw(dst_root_rot)

            dst_dof, dst_root_pos, dst_root_rot = target_smoother.step(
                dst_dof, dst_root_pos, dst_root_rot
            )

            if args.motion_debug_csv:
                dbg = camera_local_root.last_debug or {}
                student_dbg = getattr(student, "last_debug", {}) or {}
                dst_yaw = _extract_yaw_xyzw(dst_root_rot)
                if prev_debug_dst_root_pos is None:
                    dst_step = np.zeros(3, dtype=np.float32)
                else:
                    dst_step = dst_root_pos - prev_debug_dst_root_pos
                prev_debug_dst_root_pos = dst_root_pos.copy()
                motion_debug_rows.append({
                    "video_frame_idx": frame_idx,
                    "output_frame_idx": produced,
                    "mode": args.root_translation_mode,
                    "translation_frame": args.root_translation_frame,
                    "cam_x": dbg.get("cam_x", 0.0),
                    "cam_y": dbg.get("cam_y", 0.0),
                    "cam_z": dbg.get("cam_z", 0.0),
                    "cam_dx": dbg.get("cam_dx", 0.0),
                    "cam_dy": dbg.get("cam_dy", 0.0),
                    "cam_dz": dbg.get("cam_dz", 0.0),
                    "mapped_local_dx": dbg.get("local_dx", 0.0),
                    "mapped_local_dy": dbg.get("local_dy", 0.0),
                    "heading_yaw": dbg.get("heading_yaw", 0.0),
                    "mapped_world_dx": dbg.get("world_dx", 0.0),
                    "mapped_world_dy": dbg.get("world_dy", 0.0),
                    "mapped_world_dz": dbg.get("world_dz", 0.0),
                    "dst_root_x": float(dst_root_pos[0]),
                    "dst_root_y": float(dst_root_pos[1]),
                    "dst_root_z": float(dst_root_pos[2]),
                    "dst_step_x": float(dst_step[0]),
                    "dst_step_y": float(dst_step[1]),
                    "dst_step_z": float(dst_step[2]),
                    "dst_yaw": float(dst_yaw),
                    "dst_forward_x": float(np.cos(dst_yaw)),
                    "dst_forward_y": float(np.sin(dst_yaw)),
                    "smpl_raw_lin_x": student_dbg.get("smpl_raw_lin_x", 0.0),
                    "smpl_raw_lin_y": student_dbg.get("smpl_raw_lin_y", 0.0),
                    "smpl_raw_lin_z": student_dbg.get("smpl_raw_lin_z", 0.0),
                    "smpl_raw_ang_x": student_dbg.get("smpl_raw_ang_x", 0.0),
                    "smpl_raw_ang_y": student_dbg.get("smpl_raw_ang_y", 0.0),
                    "smpl_raw_ang_z": student_dbg.get("smpl_raw_ang_z", 0.0),
                    "smpl_world_vz": student_dbg.get("smpl_world_vz", 0.0),
                    "smpl_lin_x": student_dbg.get("smpl_lin_x", 0.0),
                    "smpl_lin_y": student_dbg.get("smpl_lin_y", 0.0),
                    "smpl_lin_z": student_dbg.get("smpl_lin_z", 0.0),
                    "smpl_ang_x": student_dbg.get("smpl_ang_x", 0.0),
                    "smpl_ang_y": student_dbg.get("smpl_ang_y", 0.0),
                    "smpl_ang_z": student_dbg.get("smpl_ang_z", 0.0),
                    "smpl_z_abs_max": student_dbg.get("smpl_z_abs_max", 0.0),
                    "smpl_z_rms": student_dbg.get("smpl_z_rms", 0.0),
                    "smpl_low_std_clamped": student_dbg.get("smpl_low_std_clamped", 0.0),
                    "pred_root_vx": student_dbg.get("pred_root_vx", 0.0),
                    "pred_root_vy": student_dbg.get("pred_root_vy", 0.0),
                    "pred_root_vz": student_dbg.get("pred_root_vz", 0.0),
                    "pred_yaw_rate": student_dbg.get("pred_yaw_rate", 0.0),
                })

            viewer_start = time.perf_counter()
            if show_src_viewer and qpos is not None:
                apply_src_frame(qpos)
            apply_dst_frame(dst_dof, dst_root_pos, dst_root_rot)
            if use_split_viewer:
                viewer.sync(frame_bgr=frame_bgr)
            else:
                viewer.sync()
            viewer_ms = (time.perf_counter() - viewer_start) * 1000.0

            out_dof.append(dst_dof.copy())
            out_root_pos.append(dst_root_pos.copy())
            out_root_rot.append(dst_root_rot.copy())
            zmq_publish_start = time.perf_counter()
            ref_packet_published = False
            if ref_publisher is not None:
                ref_packet = ref_publisher.append_and_publish(
                    dst_dof,
                    dst_root_pos,
                    dst_root_rot,
                    motion_mode=int(getattr(student, "motion_mode", 0)),
                    gripper_score=None if gripper_state is None else gripper_state["score"],
                    gripper_target_angle=None if gripper_state is None else gripper_state["target_angle"],
                    gripper_bin=None if gripper_state is None else gripper_state["bin"],
                )
                ref_packet_published = ref_packet is not None
            zmq_publish_ms = (time.perf_counter() - zmq_publish_start) * 1000.0
            produced += 1

            total_ms = (time.perf_counter() - loop_start) * 1000.0
            timing_rows.append(
                {
                    "video_frame_idx": frame_idx,
                    "output_frame_idx": produced - 1,
                    "read_ms": read_ms,
                    "camera_capture_ms": read_ms,
                    "fastsam_ms": fastsam_ms,
                    "fastsam3d_body_ms": fastsam_ms,
                    "mhr2smpl_ms": mhr2smpl_ms,
                    "smooth_ms": smooth_ms,
                    "smplx_fk_ms": smplx_fk_ms,
                    "gmr_ms": gmr_ms,
                    "gmr_retargeting_ms": gmr_ms,
                    "student_ms": student_ms,
                    "student_inference_ms": student_ms,
                    "zmq_publish_ms": zmq_publish_ms,
                    "zmq_ref_packet_published": int(ref_packet_published),
                    "viewer_ms": viewer_ms,
                    "gripper_score": -1.0 if gripper_state is None else float(gripper_state["score"]),
                    "gripper_target_angle": 0.0 if gripper_state is None else float(gripper_state["target_angle"]),
                    "gripper_bin": -1 if gripper_state is None else int(gripper_state["bin"]),
                    "total_ms": total_ms,
                }
            )

            frame_idx += 1
            if args.realtime:
                now = time.perf_counter()
                next_video_time += display_dt
                behind = now - next_video_time
                if behind > display_dt:
                    skip_count = int(behind / display_dt)
                    if target_frames > 0:
                        skip_count = min(skip_count, target_frames - frame_idx)
                    for _ in range(skip_count):
                        if not cap.grab():
                            break
                        frame_idx += 1
                        pbar.update(1)
                    next_video_time = time.perf_counter()
            else:
                elapsed = time.perf_counter() - loop_start
                if elapsed < display_dt:
                    time.sleep(display_dt - elapsed)
        pbar.close()

    cap.release()
    if ref_publisher is not None:
        ref_publisher.close()
    if smpl_publisher is not None:
        smpl_publisher.close()
    if args.output_pkl:
        save_target_pkl(args.output_pkl, "clean exit")
    if autosave_path:
        save_target_pkl(autosave_path, "clean exit autosave")

    save_rows_csv(args.timing_csv, timing_rows, "timing", "clean exit")
    save_rows_csv(args.motion_debug_csv, motion_debug_rows, "motion debug", "clean exit")

    log_final_summary("clean exit")


if __name__ == "__main__":
    main()
