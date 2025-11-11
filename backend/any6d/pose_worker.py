#!/usr/bin/env python3
"""pose_worker.py.

Runs inside the any6d conda environment.
- Watches <jobs_dir>/inbox for JSON jobs
- Loads RGB/Depth tensors from 'obs_path'
- Runs estimate_pose_from_tensors(...) using pose_fn
- Writes results to <jobs_dir>/outbox as JSON (+ optional visualization PNG)

Usage:
  conda run -n any6d python pose_worker.py \
    --jobs-dir /tmp/crowd_obs_cache/pose_jobs \
    --object Cube_Blue \
    --mesh assets/meshes/cube_blue.obj \
    --prompt "blue cube"

Debug mode:
  Set POSE_WORKER_DEBUG=1 to enable debugpy - worker will WAIT for debugger to attach

"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh

# ========== DEBUGPY SUPPORT ==========
# Check if debug mode is enabled via environment variable
if os.getenv("POSE_WORKER_DEBUG", "0") == "1":
    try:
        import debugpy

        # Use a different port for each object to avoid conflicts
        object_name = sys.argv[sys.argv.index("--object") + 1] if "--object" in sys.argv else "unknown"
        # Hash object name to get a consistent port offset (0-9)
        port_offset = hash(object_name) % 10
        debug_port = 5678 + port_offset

        # Check if already listening (in case of restart)
        if not debugpy.is_client_connected():
            try:
                debugpy.listen(("localhost", debug_port))
                print(f"\n{'='*60}", flush=True)
                print(f"🐛 [{object_name}] DEBUGPY READY", flush=True)
                print(f"🐛 Port: {debug_port}", flush=True)
                print(f"🐛 Attach config: 'Attach to Pose Worker ({object_name})'", flush=True)
                print(f"🐛 WAITING FOR DEBUGGER TO ATTACH...", flush=True)
                print(f"{'='*60}\n", flush=True)

                # BLOCK until debugger attaches
                debugpy.wait_for_client()
                print(f"✅ [{object_name}] Debugger attached! Continuing...\n", flush=True)

                # Optional: break at the start
                # debugpy.breakpoint()
            except Exception as listen_err:
                print(f"⚠️  Failed to listen on port {debug_port}: {listen_err}", flush=True)
                print(f"⚠️  Continuing without debugger...", flush=True)
    except ImportError:
        print("⚠️  debugpy not installed in any6d environment.", flush=True)
        print("⚠️  Install with: conda activate any6d && pip install debugpy", flush=True)
    except Exception as e:
        print(f"⚠️  Debugpy setup failed: {e}", flush=True)
        import traceback

        traceback.print_exc()
# =====================================

# Ensure pose_fn is importable (assumes same repo)
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from estimate_pose import (
    create_engines,
    estimate_pose_from_tensors,
    reset_tracking,
    visualize_estimation,
)


def _load_obs(obs_path: str) -> dict:
    try:
        return torch.load(obs_path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"failed to torch.load obs_path={obs_path}: {e}")


def _extract_rgb_depth(obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Try common keys. Returns (rgb_t, depth_t).

    - RGB expected shape (480,640,3) uint8 or float
    - Depth expected shape (480,640) float meters

    """
    # RGB
    rgb = obs.get("observation.images.cam_main")
    if rgb is None:
        raise KeyError("RGB not found in obs dict (expected observation.images.cam_main, etc.)")
    if isinstance(rgb, np.ndarray):
        rgb = torch.from_numpy(rgb)
    assert isinstance(rgb, torch.Tensor), "RGB must be torch.Tensor or numpy.ndarray"
    if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[-1] != 3:
        rgb = rgb.permute(1, 2, 0).contiguous()  # CHW->HWC

    # DEPTH
    depth = obs.get("depth")
    if depth is None:
        raise KeyError("depth not found in obs dict (expected 'depth' key or similar)")
    if isinstance(depth, np.ndarray):
        depth = torch.from_numpy(depth)
    assert isinstance(depth, torch.Tensor), "Depth must be torch.Tensor or numpy.ndarray"
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    return rgb, depth


def _as_K_array(K_like) -> np.ndarray:
    K = np.asarray(K_like, dtype=np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {K.shape}")
    return K


def claim_job(inbox: Path, tmpdir: Path, object_name: str) -> Path | None:
    """Atomically move one job for 'object_name' from inbox -> tmpdir and return its path."""
    for p in inbox.glob("*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                hdr = json.load(f)
            if hdr.get("object") != object_name:
                continue
        except Exception:
            # unreadable; try to move it aside to avoid hot-looping
            try:
                bad = tmpdir / f"bad_{p.name}"
                os.replace(p, bad)
            except Exception:
                pass
            continue

        # Try to claim (atomic move)
        dest = tmpdir / p.name
        try:
            os.replace(p, dest)
            return dest
        except FileNotFoundError:
            # Raced; continue
            continue
        except Exception:
            continue
    return None


def write_json_atomic(obj: dict, outdir: Path, name: str):
    tmp = outdir / (name + ".tmp")
    dst = outdir / name
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-dir", required=True, type=Path)
    ap.add_argument("--object", required=True, help="Object name this worker owns")
    ap.add_argument("--mesh", required=True, type=str)
    ap.add_argument("--prompt", default=None, type=str)
    ap.add_argument("--est-refine-iter", type=int, default=None)
    ap.add_argument("--track-refine-iter", type=int, default=None)
    ap.add_argument("--save-viz", action="store_true", help="Save RGB overlay PNG per job")
    args = ap.parse_args()

    inbox = (args.jobs_dir / "inbox").resolve()
    outbox = (args.jobs_dir / "outbox").resolve()
    tmpdir = (args.jobs_dir / "tmp").resolve()
    inbox.mkdir(parents=True, exist_ok=True)
    outbox.mkdir(parents=True, exist_ok=True)
    tmpdir.mkdir(parents=True, exist_ok=True)

    # Load mesh & create engines (expensive) once
    print(f"[{args.object}] 🔄 Initializing models...", flush=True)
    try:
        mesh = trimesh.load(args.mesh)
        if mesh.is_empty:
            raise RuntimeError("mesh is empty")
        _ = mesh.vertex_normals
    except Exception as e:
        print(f"[{args.object}] ✖ mesh load failed: {e}", flush=True)
        sys.exit(1)

    try:
        engines = create_engines(mesh, debug=0)
    except Exception as e:
        print(f"[{args.object}] ✖ engines init failed: {e}", flush=True)
        sys.exit(2)

    print(f"[{args.object}] ✅ worker ready (watching {inbox})", flush=True)

    while True:
        job_path = claim_job(inbox, tmpdir, args.object)
        if not job_path:
            time.sleep(0.1)
            continue

        try:
            with open(job_path, "r", encoding="utf-8") as f:
                job = json.load(f)
        except Exception as e:
            print(f"[{args.object}] ⚠️ bad job file {job_path.name}: {e}", flush=True)
            try:
                job_path.unlink()
            except Exception:
                pass
            continue

        job_id = job.get("job_id", "unknown")
        episode_id = int(job.get("episode_id"))
        state_id = int(job.get("state_id"))
        obs_path = job.get("obs_path")
        prompt = job.get("prompt") or args.prompt or args.object
        K = _as_K_array(job.get("K"))
        est_iters = int(job.get("est_refine_iter") or args.est_refine_iter or 20)
        track_iters = int(job.get("track_refine_iter") or args.track_refine_iter or 8)

        print(f"[{args.object}] 🔍 Processing job {job_id} (ep={episode_id}, state={state_id})", flush=True)
        print(f"[{args.object}]    obs_path: {obs_path}", flush=True)
        print(f"[{args.object}]    prompt: {prompt}", flush=True)

        result = {
            "job_id": job_id,
            "episode_id": episode_id,
            "state_id": state_id,
            "object": args.object,
            "success": False,
        }

        try:
            print(f"[{args.object}] 📂 Loading observation from {obs_path}...", flush=True)
            obs = _load_obs(obs_path)
            print(f"[{args.object}] 🖼️  Extracting RGB and depth...", flush=True)
            rgb_t, depth_t = _extract_rgb_depth(obs)
            print(f"[{args.object}] ✓ Observation loaded successfully (RGB shape={rgb_t.shape}, depth shape={depth_t.shape})", flush=True)
        except Exception as e:
            print(f"[{args.object}] ❌ Failed to load observation: {e}", flush=True)
            result["error"] = f"obs load/extract failed: {e}"
            write_json_atomic(result, outbox, f"{job_id}.json")
            try:
                job_path.unlink()
            except Exception:
                pass
            continue

        # Run pose (stateless per call)
        try:
            # Make each call independent
            reset_tracking(engines)

            print(f"[{args.object}] 🎯 Running pose estimation...", flush=True)
            out = estimate_pose_from_tensors(
                mesh=mesh,
                rgb_t=rgb_t,
                depth_t=depth_t,
                K=K,
                language_prompt=prompt,
                est_refine_iter=est_iters,
                track_refine_iter=track_iters,
                debug=0,
                engines=engines,
            )
            if out.success:
                result["success"] = True
                result["pose_cam_T_obj"] = out.pose_cam_T_obj.detach().cpu().numpy().tolist()
                result["mask_area"] = int(out.extras.get("mask_area", 0))
                print(f"[{args.object}] ✅ Pose estimation SUCCESS! mask_area={result['mask_area']}", flush=True)
                print(f"[{args.object}]    pose_cam_T_obj: {result['pose_cam_T_obj']}", flush=True)
            else:
                # Extract all available error information
                result["error"] = out.extras.get("error", "pose failed")
                if "reason" in out.extras:
                    result["reason"] = out.extras["reason"]
                if "mask_area" in out.extras:
                    result["mask_area"] = int(out.extras["mask_area"])
                
                # Print detailed error info
                error_parts = [f"error={result['error']}"]
                if "reason" in result:
                    error_parts.append(f"reason={result['reason']}")
                if "mask_area" in result:
                    error_parts.append(f"mask_area={result['mask_area']}")
                
                error_msg = ", ".join(error_parts)
                print(f"[{args.object}] ❌ Pose estimation FAILED: {error_msg}", flush=True)
                print(f"[{args.object}]    extras: {out.extras}", flush=True)
        except Exception as e:
            result["error"] = f"pose exception: {e}"
            print(f"[{args.object}] ❌ Pose estimation EXCEPTION: {e}", flush=True)
            import traceback

            traceback.print_exc()

        # Optional visualization
        try:
            if result["success"] and (args.save_viz or bool(os.getenv("POSE_SAVE_VIZ", "0") == "1")):
                viz = visualize_estimation(
                    rgb_t=rgb_t,
                    depth_t=depth_t,
                    K=K,
                    pose_out=out,
                    axis_scale=0.05,
                    depth_min=0.1,
                    depth_max=1.0,
                    overlay_mask=True,
                )
                png_path = outbox / f"viz_{job_id}.png"
                import cv2  # local import to avoid import cycles

                cv2.imwrite(str(png_path), viz["rgb_with_pose_bgr"])
                result["pose_viz_path"] = str(png_path)
        except Exception:
            # viz is best-effort
            pass

        # Emit result JSON
        try:
            write_json_atomic(result, outbox, f"{job_id}.json")
            print(
                f"[{args.object}] 📤 Result written to outbox: {job_id}.json (success={result['success']})", flush=True
            )
        except Exception as e:
            print(f"[{args.object}] ⚠️ failed to write result for {job_id}: {e}", flush=True)

        # Remove the claimed job file
        try:
            job_path.unlink()
            print(f"[{args.object}] 🗑️  Job file removed", flush=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
