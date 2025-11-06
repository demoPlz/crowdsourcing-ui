from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

# Add external/any6d to sys.path so we can import estimater, foundationpose, etc.
# This file is at: backend/any6d/estimate_pose.py
# Target directory: external/any6d/
_BACKEND_DIR = Path(__file__).resolve().parent  # backend/any6d/
_REPO_ROOT = _BACKEND_DIR.parent.parent         # crowdsourcing-ui/
_EXTERNAL_ANY6D = _REPO_ROOT / "external" / "any6d"

if str(_EXTERNAL_ANY6D) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_ANY6D))

# Also add lang-segment-anything for lang_sam imports
_LANG_SAM_DIR = _EXTERNAL_ANY6D / "lang-segment-anything"
if str(_LANG_SAM_DIR) not in sys.path:
    sys.path.insert(0, str(_LANG_SAM_DIR))

# External deps expected by your environment:
#   nvdiffrast.torch as dr
#   lang_sam.LangSAM (from lang-segment-anything/)
#   estimater.Any6D (from external/any6d/)
#   foundationpose.* (from external/any6d/)

import nvdiffrast.torch as dr
from lang_sam import LangSAM

from estimater import Any6D
from foundationpose.estimater import FoundationPose
from foundationpose.learning.training.predict_pose_refine import PoseRefinePredictor
from foundationpose.learning.training.predict_score import ScorePredictor

# For visualization
import cv2
from foundationpose.Utils import draw_posed_3d_box, draw_xyz_axis


# ------------------------- Lazy singletons -------------------------

_LANGSAM_SINGLETON: Optional[LangSAM] = None
def _get_langsam() -> LangSAM:
    global _LANGSAM_SINGLETON
    if _LANGSAM_SINGLETON is None:
        _LANGSAM_SINGLETON = LangSAM()
    return _LANGSAM_SINGLETON

# ------------------------- Reusable engines -------------------------

@dataclass
class PoseEngines:
    langsam: LangSAM
    any6d: Any6D
    fpose: FoundationPose

def create_engines(
    mesh: trimesh.Trimesh,
    *,
    debug: int = 0,
    langsam: LangSAM | None = None,
) -> PoseEngines:
    """
    Build reusable inference engines tied to a specific mesh.
    Pass the returned PoseEngines into estimate_pose_from_tensors() for fast calls.
    """
    # LangSAM (allow passing a shared instance; otherwise lazily create)
    if langsam is None:
        langsam = _get_langsam()

    # Ensure normals exist on a private copy used by engines
    m = mesh.copy()
    _ = m.vertex_normals

    glctx_any6d = dr.RasterizeCudaContext()
    any6d = Any6D(mesh=m, debug=debug, debug_dir='./output/any6d_debug', glctx=glctx_any6d)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx_found = dr.RasterizeCudaContext()
    fpose = FoundationPose(
        model_pts=m.vertices, model_normals=m.vertex_normals, mesh=m,
        scorer=scorer, refiner=refiner, glctx=glctx_found, debug=debug, debug_dir='./output/any6d_debug',
    )
    return PoseEngines(langsam=langsam, any6d=any6d, fpose=fpose)

def reset_tracking(engines: PoseEngines) -> None:
    """
    Clear any persistent per-sequence tracking state so a new call is independent.
    """
    engines.fpose.pose_last = None
    if hasattr(engines.any6d, "pose_last"):
        engines.any6d.pose_last = None

# ------------------------- Visualization -------------------------

def _colorize_depth_bgr(depth_m: np.ndarray, depth_min: float, depth_max: float) -> np.ndarray:
    """Meters -> BGR heatmap (uint8). Invalid (<=0) becomes black."""
    valid = depth_m > 0.0
    if not np.any(valid):
        return np.zeros((*depth_m.shape, 3), dtype=np.uint8)

    vmin, vmax = float(depth_min), float(depth_max)
    if not (vmax > vmin):
        vals = depth_m[valid]
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        if not (vmax > vmin):
            vmax = vmin + 1e-6

    dep = np.clip(depth_m, vmin, vmax)
    dep = (dep - vmin) / max(vmax - vmin, 1e-6)
    dep = np.nan_to_num(dep, nan=0.0, posinf=1.0, neginf=0.0)
    dep_u8 = np.clip(dep * 255.0, 0, 255).astype(np.uint8)
    dep_u8[~valid] = 0
    heat = cv2.applyColorMap(dep_u8, cv2.COLORMAP_JET)
    heat[~valid] = 0
    return heat


def visualize_estimation(
    rgb_t: torch.Tensor,                # (H,W,3), uint8 or float
    depth_t: torch.Tensor,              # (H,W) float (meters)
    K: torch.Tensor | np.ndarray,       # 3x3
    pose_out: PoseOutput,
    *,
    axis_scale: float = 0.05,
    depth_min: float = 0.1,
    depth_max: float = 1.0,
    overlay_mask: bool = True,
    mask_alpha: float = 0.35,
    mask_color_bgr: Tuple[int, int, int] = (0, 255, 0),
) -> Dict[str, np.ndarray]:
    """
    Build visualization products for the current estimate.

    Returns dict with:
      - 'rgb_with_pose_bgr': RGB frame with bbox + axes drawn (BGR, uint8).
      - 'depth_viz_bgr': depth heatmap in BGR (uint8).
      - 'mask_gray': mask image (uint8, 0/255), if available.
    """
    # Convert inputs
    rgb_np = _rgb_torch_to_uint8_numpy(rgb_t)           # RGB uint8
    depth_np = _depth_torch_to_f32_numpy(depth_t)       # float32 meters
    K_np = _K_to_numpy(K)

    # Base BGR image for OpenCV drawing
    bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

    # Depth heatmap
    depth_viz = _colorize_depth_bgr(depth_np, depth_min, depth_max)

    mask_img = None
    if pose_out.mask is not None:
        mask_bool = pose_out.mask.detach().cpu().numpy().astype(bool)
        mask_img = (mask_bool.astype(np.uint8)) * 255

        if overlay_mask and mask_img is not None:
            colored = np.zeros_like(bgr)
            colored[:, :] = mask_color_bgr
            bgr = np.where(mask_bool[..., None], (mask_alpha * colored + (1.0 - mask_alpha) * bgr).astype(np.uint8), bgr)

    # If no successful estimate, just return overlays we have
    if not pose_out.success:
        out = {
            "rgb_with_pose_bgr": bgr,
            "depth_viz_bgr": depth_viz,
        }
        if mask_img is not None:
            out["mask_gray"] = mask_img
        return out

    # Pull pose & geometry; convert to numpy
    pose_44 = pose_out.pose_cam_T_obj.detach().cpu().numpy().astype(np.float32)            # (4,4)
    to_origin = pose_out.to_origin.detach().cpu().numpy().astype(np.float32)               # (4,4)
    bbox_obj = pose_out.bbox_obj_frame.detach().cpu().numpy().astype(np.float32)           # (2,3)

    # Convert object pose to the oriented box's center pose (matches original script)
    try:
        center_pose = pose_44 @ np.linalg.inv(to_origin)
    except np.linalg.LinAlgError:
        center_pose = pose_44

    # Draw bbox and axes (expects BGR image)
    vis = draw_posed_3d_box(K_np, bgr.copy(), center_pose, bbox_obj)
    vis = draw_xyz_axis(
        vis, ob_in_cam=center_pose, scale=axis_scale, K=K_np, thickness=2, transparency=0.3
    )

    out = {
        "rgb_with_pose_bgr": vis,
        "depth_viz_bgr": depth_viz,
    }
    if mask_img is not None:
        out["mask_gray"] = mask_img
    return out


# ------------------------- Small helpers -------------------------

def _rgb_torch_to_uint8_numpy(rgb_t: torch.Tensor) -> np.ndarray:
    """(H,W,3) torch -> (H,W,3) np.uint8 RGB."""
    if rgb_t.ndim != 3 or rgb_t.shape[-1] != 3:
        raise ValueError(f"rgb_t must be (H,W,3), got {tuple(rgb_t.shape)}")
    arr = rgb_t.detach().cpu().numpy()
    if np.issubdtype(arr.dtype, np.floating):
        mx = float(np.nanmax(arr)) if arr.size else 1.0
        # Heuristic: if max≤1.5 assume [0,1] inputs
        if mx <= 1.5:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        else:
            arr = np.clip(arr, 0.0, 255.0)
        arr = arr.astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

def _depth_torch_to_f32_numpy(depth_t: torch.Tensor) -> np.ndarray:
    """(H,W) torch -> (H,W) np.float32 meters."""
    if depth_t.ndim != 2:
        raise ValueError(f"depth_t must be (H,W), got {tuple(depth_t.shape)}")
    arr = depth_t.detach().cpu().numpy().astype(np.float32)
    return arr

def _K_to_numpy(K: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()
    K = np.asarray(K, dtype=np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {tuple(K.shape)}")
    return K


# ------------------------- Return type -------------------------

@dataclass
class PoseOutput:
    """Results of a single-view pose estimate."""
    success: bool
    pose_cam_T_obj: Optional[torch.Tensor]      # (4,4) float32 in torch, or None if failed
    mask: Optional[torch.Tensor]                # (H,W) bool torch
    bbox_obj_frame: Optional[torch.Tensor]      # (2,3) float32 torch (min,max corners of oriented bbox in object frame)
    to_origin: Optional[torch.Tensor]           # (4,4) float32 torch (object->OBB origin transform)
    extras: Dict[str, Any]                      # anything else (areas, scores, debug strings, etc.)


# ------------------------- Core function -------------------------

@torch.no_grad()
def estimate_pose_from_tensors(
    mesh: trimesh.Trimesh,
    rgb_t: torch.Tensor,                        # (480,640,3) torch
    depth_t: torch.Tensor,                      # (480,640) torch, meters
    K: torch.Tensor | np.ndarray,               # 3x3 intrinsics
    *,
    language_prompt: str = "object",
    depth_min: float = 0.1,
    depth_max: float = 1.0,
    mask_min_pixels: int = 500,
    mask_max_pixels: int = 80_000,              # 0 disables this upper check
    est_refine_iter: int = 20,
    track_refine_iter: int = 8,
    debug: int = 0,
    engines: PoseEngines| None = None,
) -> PoseOutput:
    """
    Minimal single-call 6D pose estimation:
      1) Build a language-guided mask (LangSAM),
      2) (Optionally) gate by depth range,
      3) Initialize with Any6D,
      4) Refine one step with FoundationPose.

    Returns torch tensors so you can keep everything in PyTorch-land upstream/downstream.

    Raises on obvious input-shape issues; otherwise catches model errors and returns success=False.
    """
    # ---- Input preparation
    rgb_np = _rgb_torch_to_uint8_numpy(rgb_t)                # (H,W,3) uint8 RGB
    depth_np = _depth_torch_to_f32_numpy(depth_t)            # (H,W) float32 meters
    depth_np *= 0.001
    K_np = _K_to_numpy(K)

    H, W = rgb_np.shape[:2]
    if (H, W) != (480, 640):
        # Not strictly required by the algorithms, but mirroring your stated expectation
        pass

    # ---- Mesh housekeeping (ensure normals; compute oriented bbox)
    if mesh is None or mesh.vertices.size == 0:
        raise ValueError("Mesh is empty or None.")
    to_origin_np, extents = trimesh.bounds.oriented_bounds(mesh)  # (4,4), (3,)
    bbox_np = np.stack([-extents / 2.0, extents / 2.0], axis=0).reshape(2, 3)

    # ---- Build LangSAM mask
    try:
        # Use provided engines if available; otherwise build a temporary set.
        if engines is None:
            engines = create_engines(mesh=mesh, debug=debug)
        langsam = engines.langsam
        any6d = engines.any6d
        fpose = engines.fpose

        reset_tracking(engines)

        image_pil = Image.fromarray(rgb_np)
        out = langsam.predict([image_pil], [language_prompt])[0]
        masks = np.asarray(out.get("masks", []))
    
        scores = np.asarray(out["mask_scores"])
        idx = int(np.argmax(scores)) if scores.size else 0
        lang_mask = masks[idx].astype(bool) if masks.size else np.zeros((H, W), dtype=bool)
    except Exception as e:
        return PoseOutput(
            success=False, pose_cam_T_obj=None, mask=None, bbox_obj_frame=None, to_origin=None,
            extras={"error": f"LangSAM failed: {e}"}
        )

    # ---- Optionally combine with a conservative depth range
    valid_depth = (depth_np > 0.0) & (depth_np >= depth_min) & (depth_np <= depth_max)
    ob_mask = (lang_mask & valid_depth)

    # ---- Sanity checks on mask area
    area = int(ob_mask.sum())
    if area < mask_min_pixels or (mask_max_pixels > 0 and area > mask_max_pixels):
        return PoseOutput(
            success=False, pose_cam_T_obj=None,
            mask=torch.from_numpy(ob_mask),
            bbox_obj_frame=torch.from_numpy(bbox_np.astype(np.float32)),
            to_origin=torch.from_numpy(to_origin_np.astype(np.float32)),
            extras={"mask_area": area, "reason": "mask area out of bounds"}
        )
    
    # ---- Run Any6D initialization
    try:
        pose_init_np = any6d.register_any6d(
            K=K_np,
            rgb=rgb_np,
            depth=depth_np,
            ob_mask=ob_mask,
            iteration=est_refine_iter,
            name="single_frame",
        )
        # Hand over internal pose state for FoundationPose
        fpose.pose_last = any6d.pose_last.detach().clone()
    except Exception as e:
        return PoseOutput(
            success=False, pose_cam_T_obj=None, mask=torch.from_numpy(ob_mask),
            bbox_obj_frame=torch.from_numpy(bbox_np.astype(np.float32)),
            to_origin=torch.from_numpy(to_origin_np.astype(np.float32)),
            extras={"error": f"Any6D init failed: {e}"}
        )

    # ---- Optional one-step refinement with FoundationPose (helps stability)
    try:
        pose_refined_np = fpose.track_one(
            rgb=rgb_np, depth=depth_np, K=K_np, iteration=track_refine_iter
        )
        pose_np = pose_refined_np.astype(np.float32)
    except Exception:
        # Fallback to init pose if refinement fails
        pose_np = np.asarray(pose_init_np, dtype=np.float32)

    # ---- Package results (torch outputs)
    return PoseOutput(
        success=True,
        pose_cam_T_obj=torch.from_numpy(pose_np),
        mask=torch.from_numpy(ob_mask),
        bbox_obj_frame=torch.from_numpy(bbox_np.astype(np.float32)),
        to_origin=torch.from_numpy(to_origin_np.astype(np.float32)),
        extras={"mask_area": area}
    )


# ------------------------- Tiny smoke-test harness -------------------------
if __name__ == "__main__":
    """
    Minimal standalone check (won't produce a meaningful pose without proper models/weights):
      - Creates a quick box mesh,
      - Uses random RGB/depth just to exercise the code path.

    Replace the fake inputs with real data in your environment for a true test.
    """
    import trimesh.creation as creation

    # Fake inputs just to run the function without crashing
    mesh = creation.box(extents=(0.05, 0.05, 0.05))  # 5cm cube
    rgb_t = torch.randint(0, 255, (480, 640, 3), dtype=torch.uint8)
    depth_t = torch.full((480, 640), 0.5, dtype=torch.float32)  # flat 0.5m plane
    K = torch.tensor([[600.0, 0.0, 320.0],
                      [0.0, 600.0, 240.0],
                      [0.0,   0.0,   1.0]], dtype=torch.float32)
    
    engines = create_engines(mesh, debug=0)

    out = estimate_pose_from_tensors(
        mesh, rgb_t, depth_t, K,
        language_prompt="cube",
        debug=0,
        engines=engines
    )
    print("Success:", out.success)
    if out.success:
        print("Pose (4x4):\n", out.pose_cam_T_obj.numpy())
        print("Mask area:", out.extras.get("mask_area"))
