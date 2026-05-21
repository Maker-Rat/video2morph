#!/usr/bin/env python3
"""Convert a video into a GMR-compatible SMPL-style npz.

This is intentionally a narrow offline bridge:
video frames -> Fast SAM 3D Body -> mhr2smpl -> AMASS/GMR-like npz.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from scipy.spatial.transform import Rotation
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def configure_fast_sam_env():
    """Match the optimized inference defaults used by run_demo.sh."""
    defaults = {
        "GPU_HAND_PREP": "1",
        "LAYER_DTYPE": "fp32",
        "SKIP_KEYPOINT_PROMPT": "1",
        "IMG_SIZE": "512",
        "USE_COMPILE": "1",
        "USE_COMPILE_BACKBONE": "1",
        "DECODER_COMPILE": "1",
        "COMPILE_MODE": "reduce-overhead",
        "COMPILE_WARMUP_BATCH_SIZES": "1",
        "MHR_USE_CUDA_GRAPH": "0",
        "KEYPOINT_PROMPT_INTERM_INTERVAL": "999",
        "BODY_INTERM_PRED_LAYERS": "0,1,2",
        "HAND_INTERM_PRED_LAYERS": "0,1",
        "MHR_NO_CORRECTIVES": "1",
        "FOV_FAST": "1",
        "FOV_MODEL": "s",
        "FOV_LEVEL": "0",
        "DEBUG_NAN": "0",
        "DEBUG_HAND_PREP": "0",
        "DEBUG_BACKBONE_INPUT": "0",
        "INTERM_TIMING": "0",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)

    fov_trt_engine = REPO_ROOT / "checkpoints/moge_trt/moge_dinov2_encoder_fp16.engine"
    os.environ.setdefault("FOV_TRT", "1" if fov_trt_engine.exists() else "0")


configure_fast_sam_env()

from mocap.core.multiview_mhr2smpl import MultiViewFusionRunner
from mocap.core.setup_estimator import build_default_estimator


def configure_backbone_trt_for_image_size(image_size):
    """Use a DINOv3 TRT engine only when it matches the requested image size."""
    trt_dir = REPO_ROOT / "checkpoints/sam-3d-body-dinov3/backbone_trt"
    candidates = [trt_dir / f"backbone_dinov3_{image_size}_fp16.engine"]
    if image_size == 512:
        candidates.append(trt_dir / "backbone_dinov3_fp16.engine")

    backbone_trt_engine = next((path for path in candidates if path.exists()), None)
    use_trt = backbone_trt_engine is not None
    os.environ["USE_TRT_BACKBONE"] = "1" if use_trt else "0"
    if use_trt:
        os.environ["TRT_BACKBONE_PATH"] = str(backbone_trt_engine)
    else:
        os.environ.pop("TRT_BACKBONE_PATH", None)

    dinov3_module = sys.modules.get("sam_3d_body.models.backbones.dinov3")
    if dinov3_module is not None:
        dinov3_module.USE_TRT_BACKBONE = use_trt
        dinov3_module.TRT_BACKBONE_PATH = str(backbone_trt_engine) if use_trt else ""


def body_quat_from_global_rot(global_rot):
    """Match the realtime publisher's body orientation convention."""
    global_rot = np.asarray(global_rot, dtype=np.float64).reshape(3)
    rot = Rotation.from_euler("ZYX", global_rot)
    x180 = Rotation.from_euler("x", 180.0, degrees=True)
    return (x180 * rot).as_quat().astype(np.float64)


def parse_root_correction(spec):
    if spec in ("none", "", None):
        return Rotation.identity()

    rotations = []
    for part in spec.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if len(part) < 3 or part[0] not in "xyz":
            raise ValueError(
                f"Invalid --root-correction entry '{part}'. Use forms like x90,y-90,z180."
            )
        axis = part[0]
        angle = float(part[1:])
        rotations.append(Rotation.from_euler(axis, angle, degrees=True))

    correction = Rotation.identity()
    for rot in rotations:
        correction = rot * correction
    return correction


def root_orient_from_output(out, mode, correction):
    if mode == "zero":
        rot = Rotation.identity()
    elif mode == "fastsam":
        rot = Rotation.from_quat(body_quat_from_global_rot(out["global_rot"]))
    elif mode == "raw_zyx":
        rot = Rotation.from_euler(
            "ZYX", np.asarray(out["global_rot"], dtype=np.float64).reshape(3)
        )
    else:
        raise ValueError(f"Unsupported root orientation mode: {mode}")
    return (correction * rot).as_rotvec().astype(np.float32)


def smooth_rotvec(prev_rotvec, curr_rotvec, alpha):
    """Causal exponential smoothing for one axis-angle rotation."""
    curr_rotvec = np.asarray(curr_rotvec, dtype=np.float64).reshape(3)
    if alpha <= 0.0 or prev_rotvec is None:
        return curr_rotvec.astype(np.float32)
    if alpha >= 1.0:
        return np.asarray(prev_rotvec, dtype=np.float32)

    prev_rot = Rotation.from_rotvec(np.asarray(prev_rotvec, dtype=np.float64).reshape(3))
    curr_rot = Rotation.from_rotvec(curr_rotvec)
    delta = prev_rot.inv() * curr_rot
    smoothed = prev_rot * Rotation.from_rotvec((1.0 - alpha) * delta.as_rotvec())
    return smoothed.as_rotvec().astype(np.float32)


def smooth_pose_body(prev_pose_body, curr_pose_body, alpha):
    """Causal smoothing for the 21 SMPL body joint rotations."""
    curr = np.asarray(curr_pose_body, dtype=np.float32).reshape(21, 3)
    if alpha <= 0.0 or prev_pose_body is None:
        return curr.reshape(63)
    prev = np.asarray(prev_pose_body, dtype=np.float32).reshape(21, 3)
    smoothed = np.empty_like(curr)
    for joint_idx in range(curr.shape[0]):
        smoothed[joint_idx] = smooth_rotvec(prev[joint_idx], curr[joint_idx], alpha)
    return smoothed.reshape(63)


class RootTranslationMapper:
    """Map FastSAM camera translation into the MuJoCo/GMR world frame."""

    def __init__(self, mode="legacy", root_height=None, scale=1.0):
        self.mode = str(mode)
        self.root_height = root_height
        self.scale = float(scale)
        self._anchor = None

    def reset(self):
        self._anchor = None

    def map(self, pred_cam_t):
        cam = np.asarray(pred_cam_t, dtype=np.float32).reshape(3)

        if self.mode == "zero":
            trans = np.zeros(3, dtype=np.float32)
        elif self.mode == "legacy":
            trans = cam.copy()
        else:
            if self._anchor is None:
                self._anchor = cam.copy()
            delta = (cam - self._anchor) * self.scale
            if self.mode == "camera_delta_flat":
                # Camera frame is x-right, y-down, z-forward.
                # MuJoCo/GMR world is x-forward, y-left, z-up.
                trans = np.array([delta[2], -delta[0], 0.0], dtype=np.float32)
            elif self.mode == "camera_delta":
                trans = np.array([delta[2], -delta[0], -delta[1]], dtype=np.float32)
            else:
                raise ValueError(f"Unsupported root translation mode: {self.mode}")

        if self.root_height is not None:
            trans[2] = float(self.root_height)
        return trans.astype(np.float32)


def choose_person(outputs):
    if not outputs:
        return None
    if len(outputs) == 1:
        return outputs[0]

    def score(out):
        for key in ("score", "scores", "bbox_score", "person_confidence"):
            if key in out:
                return float(np.asarray(out[key]).reshape(-1)[0])
        return 0.0

    return max(outputs, key=score)


def make_smplx_compatible_betas(betas):
    betas = np.asarray(betas, dtype=np.float32).reshape(-1)
    if betas.shape[0] >= 16:
        return betas[:16]
    return np.pad(betas, (0, 16 - betas.shape[0])).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Fast SAM 3D Body on a video and save a GMR-compatible SMPL npz."
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument(
        "--smpl-model-path",
        default="body_models/smpl/SMPL_NEUTRAL.pkl",
        help="SMPL neutral model pkl used by mhr2smpl",
    )
    parser.add_argument(
        "--nn-model-dir",
        default="mhr2smpl/experiments/multiview_n30000_e500",
        help="mhr2smpl model directory",
    )
    parser.add_argument(
        "--mhr2smpl-mapping-path",
        default="mhr2smpl/data/mhr2smpl_mapping.npz",
        help="MHR-to-SMPL barycentric mapping npz",
    )
    parser.add_argument(
        "--mhr-mesh-path",
        default="mhr2smpl/data/mhr_face_mask.ply",
        help="MHR mesh ply, needed when mapping uses triangle_ids",
    )
    parser.add_argument(
        "--smoother-dir",
        default=None,
        help="Optional mhr2smpl smoother directory, e.g. mhr2smpl/experiments/smoother_w5",
    )
    parser.add_argument(
        "--yolo-model",
        default="checkpoints/yolo/yolo11m-pose.engine",
        help="YOLO pose model or TensorRT engine",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        choices=[256, 384, 512],
        help="Fast-SAM image size",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    parser.add_argument(
        "--root-mode",
        choices=["zero", "camera"],
        default="zero",
        help="How to populate trans. Use zero first for pose-only GMR validation.",
    )
    parser.add_argument(
        "--root-orient-mode",
        choices=["zero", "fastsam", "raw_zyx"],
        default="fastsam",
        help="How to populate root_orient.",
    )
    parser.add_argument(
        "--root-correction",
        default="x90",
        help="Comma-separated fixed rotations applied before root orient, e.g. x90,y-90,z180.",
    )
    parser.add_argument(
        "--root-height",
        type=float,
        default=None,
        help="If set, force trans[:,2] to this height after root-mode is applied.",
    )
    parser.add_argument(
        "--pose-smooth-alpha",
        type=float,
        default=0.0,
        help="Causal EMA strength for pose_body rotations. 0 disables; try 0.3-0.7.",
    )
    parser.add_argument(
        "--root-smooth-alpha",
        type=float,
        default=0.0,
        help="Causal EMA strength for root_orient. 0 disables; try 0.3-0.7.",
    )
    parser.add_argument(
        "--fastsam-inference-type",
        choices=["full", "body"],
        default="body",
        help="Use body-only FastSAM inference for GMR npz export, or full body+hand inference.",
    )
    parser.add_argument(
        "--camera-intrinsics",
        choices=["estimate", "default"],
        default="estimate",
        help="Estimate camera intrinsics per frame with MoGe/FOV, or use default image-size intrinsics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configure_backbone_trt_for_image_size(args.image_size)
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if not 0.0 <= args.pose_smooth_alpha <= 1.0:
        raise ValueError("--pose-smooth-alpha must be in [0, 1]")
    if not 0.0 <= args.root_smooth_alpha <= 1.0:
        raise ValueError("--root-smooth-alpha must be in [0, 1]")
    root_correction = parse_root_correction(args.root_correction)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(src_fps) or src_fps <= 0:
        src_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frames = total_frames if args.max_frames <= 0 else min(total_frames, args.max_frames)

    logger.info("Loading Fast SAM 3D Body estimator")
    estimator = build_default_estimator(
        image_size=args.image_size,
        yolo_model_path=args.yolo_model,
    )
    if args.camera_intrinsics == "default":
        estimator.fov_estimator = None

    logger.info("Loading mhr2smpl runner")
    fusion_runner = MultiViewFusionRunner(
        smpl_model_path=args.smpl_model_path,
        model_dir=args.nn_model_dir,
        mapping_path=args.mhr2smpl_mapping_path,
        mhr_mesh_path=args.mhr_mesh_path,
        smoother_dir=args.smoother_dir,
    )

    pose_body = []
    root_orient = []
    trans = []
    betas_list = []
    frame_indices = []
    timestamps = []
    skipped = 0
    prev_pose_body = None
    prev_root_orient = None

    pbar = tqdm(total=target_frames, desc="Video -> SMPL", unit="frame")
    frame_idx = 0
    start = time.perf_counter()
    while frame_idx < target_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if frame_idx % args.stride != 0:
            frame_idx += 1
            pbar.update(1)
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            outputs = estimator.process_one_image(
                frame_rgb,
                hand_box_source="yolo_pose",
                inference_type=args.fastsam_inference_type,
            )

        out = choose_person(outputs)
        if out is None:
            skipped += 1
            frame_idx += 1
            pbar.update(1)
            continue

        required = ("pred_vertices", "pred_cam_t", "global_rot")
        missing = [key for key in required if key not in out]
        if missing:
            logger.warning(f"Skipping frame {frame_idx}: missing {missing}")
            skipped += 1
            frame_idx += 1
            pbar.update(1)
            continue

        smpl_pose, _canonical_joints, betas, _weights = fusion_runner.infer(
            [(out["pred_vertices"], out["pred_cam_t"])]
        )
        curr_pose_body = np.asarray(smpl_pose, dtype=np.float32).reshape(63)
        curr_root_orient = root_orient_from_output(
            out, args.root_orient_mode, root_correction
        )
        curr_pose_body = smooth_pose_body(
            prev_pose_body, curr_pose_body, args.pose_smooth_alpha
        )
        curr_root_orient = smooth_rotvec(
            prev_root_orient, curr_root_orient, args.root_smooth_alpha
        )
        prev_pose_body = curr_pose_body.copy()
        prev_root_orient = curr_root_orient.copy()

        pose_body.append(curr_pose_body)
        root_orient.append(curr_root_orient)
        if args.root_mode == "camera":
            root_trans = np.asarray(out["pred_cam_t"], dtype=np.float32).reshape(3)
        else:
            root_trans = np.zeros(3, dtype=np.float32)
        if args.root_height is not None:
            root_trans[2] = float(args.root_height)
        trans.append(root_trans)
        betas_list.append(make_smplx_compatible_betas(betas))
        frame_indices.append(frame_idx)
        timestamps.append(frame_idx / src_fps)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    if not pose_body:
        raise RuntimeError("No usable person frames were produced")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # GMR uses a sequence-level beta vector. Keep the average for stable shape.
    betas_arr = np.mean(np.stack(betas_list, axis=0), axis=0).astype(np.float32)
    mocap_fps = src_fps / args.stride
    np.savez(
        output_path,
        pose_body=np.stack(pose_body, axis=0).astype(np.float32),
        root_orient=np.stack(root_orient, axis=0).astype(np.float32),
        trans=np.stack(trans, axis=0).astype(np.float32),
        betas=betas_arr,
        gender=np.array("neutral"),
        mocap_frame_rate=np.array(mocap_fps, dtype=np.float32),
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        timestamps=np.asarray(timestamps, dtype=np.float64),
        source_video=np.array(os.path.abspath(args.video)),
        skipped_frames=np.array(skipped, dtype=np.int64),
        root_mode=np.array(args.root_mode),
        root_orient_mode=np.array(args.root_orient_mode),
        root_correction=np.array(args.root_correction),
        smoother_dir=np.array(args.smoother_dir or ""),
        pose_smooth_alpha=np.array(args.pose_smooth_alpha, dtype=np.float32),
        root_smooth_alpha=np.array(args.root_smooth_alpha, dtype=np.float32),
    )

    elapsed = time.perf_counter() - start
    logger.success(
        f"Saved {len(pose_body)} frames to {output_path} "
        f"(skipped={skipped}, fps={mocap_fps:.2f}, elapsed={elapsed:.1f}s)"
    )


if __name__ == "__main__":
    main()
