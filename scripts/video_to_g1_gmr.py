#!/usr/bin/env python3
"""Run Fast SAM 3D Body -> GMR retargeting frame by frame."""

import argparse
import contextlib
import csv
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from scipy.spatial.transform import Rotation
from smplx.joint_names import JOINT_NAMES
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _default_gmr_root():
    value = os.environ.get("GMR_ROOT")
    if value:
        return value
    sibling = (REPO_ROOT.parent / "GMR").resolve()
    return str(sibling) if sibling.exists() else None


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
from video_to_gmr_smpl_npz import (
    RootTranslationMapper,
    choose_person,
    parse_root_correction,
    root_orient_from_output,
    smooth_pose_body,
    smooth_rotvec,
)


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


@contextlib.contextmanager
def suppress_output(enabled):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8", errors="ignore") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


class SmplxFrameAdapter:
    def __init__(self, model_dir, device):
        import smplx

        self.device = torch.device(device)
        self.model = smplx.create(
            model_dir,
            "smplx",
            gender="neutral",
            use_pca=False,
            num_betas=10,
        ).to(self.device)
        self.model.eval()
        self.parents = self.model.parents.detach().cpu().numpy()
        self.joint_names = JOINT_NAMES[: len(self.parents)]

    def to_gmr_frame(self, root_orient, pose_body, trans, betas):
        pose_body = np.asarray(pose_body, dtype=np.float32).reshape(1, 63)
        root_orient = np.asarray(root_orient, dtype=np.float32).reshape(1, 3)
        trans = np.asarray(trans, dtype=np.float32).reshape(1, 3)
        betas = np.asarray(betas, dtype=np.float32).reshape(-1)[:10]
        if betas.shape[0] < 10:
            betas = np.pad(betas, (0, 10 - betas.shape[0]))

        with torch.no_grad():
            out = self.model(
                betas=torch.from_numpy(betas).to(self.device).view(1, -1),
                global_orient=torch.from_numpy(root_orient).to(self.device),
                body_pose=torch.from_numpy(pose_body).to(self.device),
                transl=torch.from_numpy(trans).to(self.device),
                left_hand_pose=torch.zeros(1, 45, device=self.device),
                right_hand_pose=torch.zeros(1, 45, device=self.device),
                jaw_pose=torch.zeros(1, 3, device=self.device),
                leye_pose=torch.zeros(1, 3, device=self.device),
                reye_pose=torch.zeros(1, 3, device=self.device),
                return_full_pose=True,
            )

        joints = out.joints[0].detach().cpu().numpy().squeeze()
        full_pose = out.full_pose[0].detach().cpu().numpy().reshape(-1, 3)

        joint_orientations = []
        result = {}
        for idx, joint_name in enumerate(self.joint_names):
            if idx == 0:
                rot = Rotation.from_rotvec(root_orient.reshape(3))
            else:
                rot = joint_orientations[self.parents[idx]] * Rotation.from_rotvec(
                    full_pose[idx]
                )
            joint_orientations.append(rot)
            result[joint_name] = (joints[idx], rot.as_quat(scalar_first=True))
        return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Fast SAM 3D Body on a video and retarget each frame to G1 with GMR."
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output-pkl", default=None, help="Optional GMR-style pkl output")
    parser.add_argument(
        "--gmr-root",
        default=_default_gmr_root(),
        help="Path to GMR checkout. Defaults to $GMR_ROOT, then ../GMR if present.",
    )
    parser.add_argument("--robot", default="unitree_g1", help="GMR target robot")
    parser.add_argument("--smplx-model-dir", default="body_models", help="Directory containing smplx models")
    parser.add_argument("--smpl-model-path", default="body_models/smpl/SMPL_NEUTRAL.pkl")
    parser.add_argument("--nn-model-dir", default="mhr2smpl/experiments/multiview_n30000_e500")
    parser.add_argument("--mhr2smpl-mapping-path", default="mhr2smpl/data/mhr2smpl_mapping.npz")
    parser.add_argument("--mhr-mesh-path", default="mhr2smpl/data/mhr_face_mask.ply")
    parser.add_argument("--smoother-dir", default=None)
    parser.add_argument("--yolo-model", default="checkpoints/yolo/yolo11m-pose.engine")
    parser.add_argument("--image-size", type=int, default=512, choices=[256, 384, 512])
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--root-mode", choices=["zero", "camera"], default="camera")
    parser.add_argument(
        "--root-translation-mode",
        choices=["legacy", "camera_delta_flat", "camera_delta", "zero"],
        default="legacy",
        help=(
            "How to map FastSAM pred_cam_t to GMR root_pos. "
            "legacy preserves the previous behavior; camera_delta_flat maps camera depth to forward X."
        ),
    )
    parser.add_argument("--root-translation-scale", type=float, default=1.0)
    parser.add_argument("--root-orient-mode", choices=["zero", "fastsam", "raw_zyx"], default="raw_zyx")
    parser.add_argument("--root-correction", default="x90")
    parser.add_argument("--root-height", type=float, default=1.0)
    parser.add_argument("--pose-smooth-alpha", type=float, default=0.45)
    parser.add_argument("--root-smooth-alpha", type=float, default=0.35)
    parser.add_argument(
        "--fastsam-inference-type",
        choices=["full", "body"],
        default="body",
        help="Use body-only FastSAM inference for GMR, or full body+hand inference.",
    )
    parser.add_argument(
        "--camera-intrinsics",
        choices=["estimate", "default"],
        default="estimate",
        help="Estimate camera intrinsics per frame with MoGe/FOV, or use default image-size intrinsics.",
    )
    parser.add_argument("--visualize", action="store_true", help="Show GMR MuJoCo viewer")
    parser.add_argument("--rate-limit", action="store_true", help="Throttle viewer to motion FPS")
    parser.add_argument(
        "--timing-csv",
        default=None,
        help="Optional CSV path for per-frame stage latency in milliseconds",
    )
    parser.add_argument(
        "--timing-log-interval",
        type=int,
        default=0,
        help="Print rolling timing summary every N produced frames; 0 disables",
    )
    parser.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress noisy model/debug prints. Progress bar and final script logs remain.",
    )
    return parser.parse_args()


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize_timings(rows, window=20):
    if not rows:
        return "no frames yet"
    sample = rows[-window:]
    keys = [
        "read_ms",
        "fastsam_ms",
        "mhr2smpl_ms",
        "smooth_ms",
        "smplx_fk_ms",
        "gmr_ms",
        "viewer_ms",
        "total_ms",
    ]
    parts = []
    for key in keys:
        vals = [row[key] for row in sample]
        parts.append(f"{key[:-3]}={np.mean(vals):.1f}ms")
    fps = 1000.0 / max(np.mean([row["total_ms"] for row in sample]), 1e-6)
    parts.append(f"loop={fps:.1f}fps")
    return ", ".join(parts)


def main():
    args = parse_args()
    configure_backbone_trt_for_image_size(args.image_size)
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if not 0.0 <= args.pose_smooth_alpha <= 1.0:
        raise ValueError("--pose-smooth-alpha must be in [0, 1]")
    if not 0.0 <= args.root_smooth_alpha <= 1.0:
        raise ValueError("--root-smooth-alpha must be in [0, 1]")

    if not args.gmr_root:
        raise RuntimeError("--gmr-root is required or set GMR_ROOT.")
    gmr_root = Path(os.path.expandvars(os.path.expanduser(args.gmr_root))).resolve()
    if not gmr_root.is_dir():
        raise RuntimeError(f"GMR root does not exist: {gmr_root}")
    if str(gmr_root) not in sys.path:
        sys.path.insert(0, str(gmr_root))

    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting import RobotMotionViewer

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(src_fps) or src_fps <= 0:
        src_fps = 30.0
    motion_fps = src_fps / args.stride
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frames = total_frames if args.max_frames <= 0 else min(total_frames, args.max_frames)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root_correction = parse_root_correction(args.root_correction)

    logger.info("Loading models")
    with suppress_output(args.quiet):
        estimator = build_default_estimator(
            image_size=args.image_size,
            yolo_model_path=args.yolo_model,
        )
        if args.camera_intrinsics == "default":
            estimator.fov_estimator = None
        fusion_runner = MultiViewFusionRunner(
            smpl_model_path=args.smpl_model_path,
            model_dir=args.nn_model_dir,
            mapping_path=args.mhr2smpl_mapping_path,
            mhr_mesh_path=args.mhr_mesh_path,
            smoother_dir=args.smoother_dir,
        )
        smplx_adapter = SmplxFrameAdapter(args.smplx_model_dir, device=device)
        retargeter = GMR(
            src_human="smplx",
            tgt_robot=args.robot,
            actual_human_height=1.75,
            verbose=False,
        )
    viewer = None
    if args.visualize:
        viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=motion_fps,
            transparent_robot=0,
        )

    qpos_list = []
    timing_rows = []
    skipped = 0
    prev_pose_body = None
    prev_root_orient = None
    root_translation_mapper = RootTranslationMapper(
        mode=("zero" if args.root_mode == "zero" else args.root_translation_mode),
        root_height=args.root_height,
        scale=args.root_translation_scale,
    )

    pbar = tqdm(total=target_frames, desc="Video -> GMR", unit="frame")
    frame_idx = 0
    start = time.perf_counter()
    try:
        while frame_idx < target_frames:
            frame_start = time.perf_counter()
            read_start = time.perf_counter()
            ok, frame_bgr = cap.read()
            read_ms = (time.perf_counter() - read_start) * 1000.0
            if not ok:
                break
            if frame_idx % args.stride != 0:
                frame_idx += 1
                pbar.update(1)
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            sync_cuda()
            fastsam_start = time.perf_counter()
            with suppress_output(args.quiet):
                with torch.no_grad():
                    outputs = estimator.process_one_image(
                        frame_rgb,
                        hand_box_source="yolo_pose",
                        inference_type=args.fastsam_inference_type,
                    )
            sync_cuda()
            fastsam_ms = (time.perf_counter() - fastsam_start) * 1000.0

            post_detect_start = time.perf_counter()
            out = choose_person(outputs)
            if out is None:
                skipped += 1
                frame_idx += 1
                pbar.update(1)
                continue
            missing = [
                key for key in ("pred_vertices", "pred_cam_t", "global_rot") if key not in out
            ]
            if missing:
                logger.warning(f"Skipping frame {frame_idx}: missing {missing}")
                skipped += 1
                frame_idx += 1
                pbar.update(1)
                continue
            post_detect_ms = (time.perf_counter() - post_detect_start) * 1000.0

            sync_cuda()
            mhr2smpl_start = time.perf_counter()
            smpl_pose, _canonical_joints, betas, _weights = fusion_runner.infer(
                [(out["pred_vertices"], out["pred_cam_t"])]
            )
            sync_cuda()
            mhr2smpl_ms = (time.perf_counter() - mhr2smpl_start) * 1000.0

            smooth_start = time.perf_counter()
            pose_body = np.asarray(smpl_pose, dtype=np.float32).reshape(63)
            root_orient = root_orient_from_output(
                out, args.root_orient_mode, root_correction
            )
            pose_body = smooth_pose_body(prev_pose_body, pose_body, args.pose_smooth_alpha)
            root_orient = smooth_rotvec(
                prev_root_orient, root_orient, args.root_smooth_alpha
            )
            prev_pose_body = pose_body.copy()
            prev_root_orient = root_orient.copy()

            trans = root_translation_mapper.map(out["pred_cam_t"])
            smooth_ms = (time.perf_counter() - smooth_start) * 1000.0

            sync_cuda()
            smplx_fk_start = time.perf_counter()
            smplx_frame = smplx_adapter.to_gmr_frame(root_orient, pose_body, trans, betas)
            sync_cuda()
            smplx_fk_ms = (time.perf_counter() - smplx_fk_start) * 1000.0

            gmr_start = time.perf_counter()
            qpos = retargeter.retarget(smplx_frame)
            gmr_ms = (time.perf_counter() - gmr_start) * 1000.0
            qpos_list.append(qpos.copy())

            viewer_ms = 0.0
            if viewer is not None:
                viewer_start = time.perf_counter()
                viewer.step(
                    root_pos=qpos[:3],
                    root_rot=qpos[3:7],
                    dof_pos=qpos[7:],
                    human_motion_data=retargeter.scaled_human_data,
                    rate_limit=args.rate_limit,
                    follow_camera=True,
                )
                viewer_ms = (time.perf_counter() - viewer_start) * 1000.0

            total_ms = (time.perf_counter() - frame_start) * 1000.0
            timing_rows.append(
                {
                    "video_frame_idx": frame_idx,
                    "output_frame_idx": len(qpos_list) - 1,
                    "read_ms": read_ms,
                    "fastsam_ms": fastsam_ms,
                    "post_detect_ms": post_detect_ms,
                    "mhr2smpl_ms": mhr2smpl_ms,
                    "smooth_ms": smooth_ms,
                    "smplx_fk_ms": smplx_fk_ms,
                    "gmr_ms": gmr_ms,
                    "viewer_ms": viewer_ms,
                    "total_ms": total_ms,
                }
            )
            if (
                args.timing_log_interval > 0
                and len(qpos_list) % args.timing_log_interval == 0
            ):
                logger.info(
                    f"Timing last {min(20, len(timing_rows))}: "
                    f"{summarize_timings(timing_rows)}"
                )

            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        if viewer is not None:
            viewer.close()

    if args.output_pkl:
        if not qpos_list:
            raise RuntimeError("No GMR frames were produced; refusing to write empty pkl")
        qpos_arr = np.asarray(qpos_list)
        root_pos = qpos_arr[:, :3]
        root_rot = qpos_arr[:, 3:7][:, [1, 2, 3, 0]]
        dof_pos = qpos_arr[:, 7:]
        motion_data = {
            "fps": motion_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": None,
            "link_body_list": None,
        }
        output_path = Path(args.output_pkl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(motion_data, f)
        logger.success(f"Saved GMR pkl to {output_path}")

    if args.timing_csv and timing_rows:
        timing_path = Path(args.timing_csv)
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        with open(timing_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(timing_rows[0].keys()))
            writer.writeheader()
            writer.writerows(timing_rows)
        logger.success(f"Saved timing CSV to {timing_path}")

    elapsed = time.perf_counter() - start
    if timing_rows:
        logger.info(f"Final timing mean: {summarize_timings(timing_rows, window=len(timing_rows))}")
    logger.success(
        f"Produced {len(qpos_list)} GMR frames "
        f"(skipped={skipped}, fps={motion_fps:.2f}, elapsed={elapsed:.1f}s)"
    )


if __name__ == "__main__":
    main()
