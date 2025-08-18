#!/usr/bin/env python3
"""
Solve camera extrinsics (T_base->cam) using a fixed ChArUco board.

Inputs:
  - --images   : glob of images where the board is visible
  - --intr     : intrinsics .npz (from fisheye calib: K, D, Knew, optional map1/map2, width/height)
  - --cols/rows/square/marker/dict : board spec (identical to what you printed)
  - --tbb      : JSON/NPZ with T_base_board (4x4). If omitted, identity is used.
  - --min-corners : drop frames with too few Charuco corners
  - --out      : output .npz with T_base_cam and helpers
  - --preview  : show a few overlay previews

Notes:
  - We UNDISTORT each image to pinhole using (K,D)->Knew, then detect & pose with Knew and D=0.
  - Computes T_base_cam_i per-frame, rejects outliers by pixel RMS, averages rotations/translations.
  - Robust to OpenCV ArUco API differences (estimatePoseCharucoBoard signatures).
"""

import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np
import cv2

# ----- Dictionaries -----
DICT_MAP = {
    "4X4_50":  cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "5X5_50":  cv2.aruco.DICT_5X5_50,
    "6X6_50":  cv2.aruco.DICT_6X6_50,
    "APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}

# ----- Routines for transforms & averaging -----
def R_from_euler_xyz(rpy):
    rx, ry, rz = rpy
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    R = np.array([
        [cy*cz,            -cy*sz,           sy],
        [sx*sy*cz+cx*sz,   -sx*sy*sz+cx*cz, -sx*cy],
        [-cx*sy*cz+sx*sz,   cx*sy*sz+sx*cz,  cx*cy]], dtype=np.float64)
    return R

def pose_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = tvec.reshape(3)
    return T

def invert_T(T):
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3,:3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def avg_quaternions(Q):
    # Q: (N,4) as [w,x,y,z]
    Q = Q.copy()
    q0 = Q[0]
    for i in range(1, len(Q)):
        if np.dot(q0, Q[i]) < 0.0:
            Q[i] = -Q[i]
    A = (Q.T @ Q) / len(Q)
    w, V = np.linalg.eigh(A)
    q = V[:, np.argmax(w)]
    if q[0] < 0: q = -q
    return q / np.linalg.norm(q)

def R_to_quat(R):
    q = np.empty(4, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t+1.0) * 2
        q[0] = 0.25*s
        q[1] = (R[2,1]-R[1,2]) / s
        q[2] = (R[0,2]-R[2,0]) / s
        q[3] = (R[1,0]-R[0,1]) / s
    else:
        i = np.argmax([R[0,0], R[1,1], R[2,2]])
        if i == 0:
            s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            q[0] = (R[2,1]-R[1,2]) / s
            q[1] = 0.25*s
            q[2] = (R[0,1]+R[1,0]) / s
            q[3] = (R[0,2]+R[2,0]) / s
        elif i == 1:
            s = np.sqrt(1.0 - R[0,0] + R[1,1] - R[2,2]) * 2
            q[0] = (R[0,2]-R[2,0]) / s
            q[1] = (R[0,1]+R[1,0]) / s
            q[2] = 0.25*s
            q[3] = (R[1,2]+R[2,1]) / s
        else:
            s = np.sqrt(1.0 - R[0,0] - R[1,1] + R[2,2]) * 2
            q[0] = (R[1,0]-R[0,1]) / s
            q[1] = (R[0,2]+R[2,0]) / s
            q[2] = (R[1,2]+R[2,1]) / s
            q[3] = 0.25*s
    return q / np.linalg.norm(q)

def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=np.float64)

def to_three_js_cam(T_base_cam_opencv):
    # Three.js camera world = OpenCV camera world * Rflip
    Rflip = np.diag([1.0, -1.0, -1.0])
    M = np.eye(4, dtype=np.float64)
    M[:3,:3] = T_base_cam_opencv[:3,:3] @ Rflip
    M[:3, 3] = T_base_cam_opencv[:3, 3]
    return M

# ----- Board helpers -----
def board_obj_corners(board):
    if hasattr(board, "getChessboardCorners"):
        C = board.getChessboardCorners()
    else:
        C = board.chessboardCorners
    return np.asarray(C, dtype=np.float64).reshape(-1, 3)

def proj_error_charuco(K, corners_charuco, ids_charuco, board, rvec, tvec):
    pts3d = board_obj_corners(board)[ids_charuco.flatten(), :]
    pts2d = corners_charuco.reshape(-1, 2)
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, None)  # D=0 (pinhole)
    proj = proj.reshape(-1, 2)
    e = np.linalg.norm(proj - pts2d, axis=1)
    return float(np.sqrt((e**2).mean()))

# ----- Robust Charuco pose for all OpenCV variants -----
def estimate_pose_charuco_compat(ch_c, ch_ids, board, K, D=None):
    """
    Returns: (retval: bool, rvec: (3,1), tvec: (3,1))
    Handles:
      - Strict builds requiring rvec/tvec IO arrays
      - Legacy builds returning (retval,rvec,tvec)
      - Falls back to solvePnP on the Charuco correspondences
    """
    # normalize inputs
    cc = np.asarray(ch_c, dtype=np.float32)
    if cc.ndim == 2 and cc.shape[1] == 2:
        cc = cc.reshape(-1, 1, 2)
    elif not (cc.ndim == 3 and cc.shape[1] == 1 and cc.shape[2] == 2):
        raise ValueError(f"Unexpected charucoCorners shape {cc.shape}; need (N,1,2) or (N,2).")
    ids = np.asarray(ch_ids, dtype=np.int32).reshape(-1, 1)
    K = np.asarray(K, dtype=np.float64)
    D_in = np.zeros((0, 1), dtype=np.float64) if D is None else np.asarray(D, dtype=np.float64)

    # 1) strict signature with IO rvec/tvec
    r0 = np.zeros((3, 1), dtype=np.float64)
    t0 = np.zeros((3, 1), dtype=np.float64)
    try:
        res = cv2.aruco.estimatePoseCharucoBoard(cc, ids, board, K, D_in, r0, t0, False)
        if isinstance(res, tuple):  # some builds still return tuple
            retval, rvec, tvec = res
            return bool(retval), np.asarray(rvec, dtype=np.float64), np.asarray(tvec, dtype=np.float64)
        return bool(res), r0, t0
    except Exception:
        pass

    # 2) legacy signature (returns tuple)
    try:
        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(cc, ids, board, K, D_in)
        return bool(retval), np.asarray(rvec, dtype=np.float64), np.asarray(tvec, dtype=np.float64)
    except Exception:
        pass

    # 3) fallback: PnP on correspondences (undistorted → D=0)
    obj_all = board_obj_corners(board)
    obj = obj_all[ids.flatten(), :]
    img = cc.reshape(-1, 2).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    return bool(ok), np.asarray(rvec, dtype=np.float64), np.asarray(tvec, dtype=np.float64)

# ----- Load T_base_board -----
def load_T_base_board(path_or_none):
    if not path_or_none:
        return np.eye(4, dtype=np.float64)
    p = Path(path_or_none)
    if p.suffix.lower() in [".npz", ".npy"]:
        data = np.load(p, allow_pickle=True)
        T = data["T"] if "T" in data else data
        return np.array(T, dtype=np.float64).reshape(4,4)
    with open(p, "r", encoding="utf-8") as f:
        j = json.load(f)
    if "T" in j:
        return np.array(j["T"], dtype=np.float64).reshape(4,4)
    if "xyz" in j and "rpy_deg" in j:
        xyz = np.array(j["xyz"], dtype=np.float64)
        rpy = np.radians(np.array(j["rpy_deg"], dtype=np.float64))
        R = R_from_euler_xyz(rpy)
        T = np.eye(4, dtype=np.float64); T[:3,:3]=R; T[:3,3]=xyz
        return T
    raise ValueError("Unsupported T_base_board JSON format.")

# ----- Main -----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Glob of images with the board visible")
    ap.add_argument("--intr", required=True, help="Intrinsics .npz path (from fisheye calib)")
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--square", type=float, required=True)
    ap.add_argument("--marker", type=float, required=True)
    ap.add_argument("--dict", default="4X4_50", choices=DICT_MAP.keys())
    ap.add_argument("--min-corners", type=int, default=12)
    ap.add_argument("--tbb", help="File with T_base_board (4x4), JSON/NPZ; if omitted, identity.")
    ap.add_argument("--out", required=True, help="Output .npz with T_base_cam etc.")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.images))
    if not paths:
        print(f"No images match {args.images}")
        sys.exit(1)

    intr = np.load(args.intr, allow_pickle=True)
    K    = intr["K"]
    D    = intr["D"]
    Knew = intr["Knew"]
    W,H  = int(intr["width"]), int(intr["height"])
    map1 = intr["map1"] if "map1" in intr.files else None
    map2 = intr["map2"] if "map2" in intr.files else None
    print(f"Loaded intrinsics: size={W}x{H}")

    T_base_board = load_T_base_board(args.tbb)
    print("T_base_board=\n", T_base_board)

    # Build board & detector
    ar_dict = cv2.aruco.getPredefinedDictionary(DICT_MAP[args.dict])
    try:
        board = cv2.aruco.CharucoBoard((args.cols, args.rows), args.square, args.marker, ar_dict)
    except TypeError:
        board = cv2.aruco.CharucoBoard_create(args.cols, args.rows, args.square, args.marker, ar_dict)

    T_list = []
    kept = 0

    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue

        # Undistort to pinhole
        if map1 is not None and map2 is not None:
            und = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        else:
            und = cv2.fisheye.undistortImage(img, K, D, Knew=Knew)

        gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)

        # Detect Charuco
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ar_dict)
        if ids is None or len(ids) < 3:
            continue
        try:
            cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedCorners=None)
        except Exception:
            # not all builds have this signature; safe to skip
            pass
        ok, ch_c, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if not ok or ch_ids is None or ch_c is None:
            continue
        if len(ch_ids) < args.min_corners:
            continue

        # Pose of BOARD in CAMERA (pinhole model, D=0)
        retval, rvec, tvec = estimate_pose_charuco_compat(ch_c, ch_ids, board, Knew, None)
        if not retval:
            continue

        # Measure reprojection error on undistorted image
        rms = proj_error_charuco(Knew, ch_c, ch_ids, board, rvec, tvec)

        T_cam_board = pose_to_T(rvec, tvec)
        T_board_cam = invert_T(T_cam_board)
        T_base_cam  = T_base_board @ T_board_cam

        T_list.append((p, rms, T_base_cam))
        kept += 1

        if args.preview and kept <= 3:
            vis = und.copy()
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            cv2.aruco.drawDetectedCornersCharuco(vis, ch_c, ch_ids, (0,255,0))
            try:
                cv2.drawFrameAxes(vis, Knew, None, rvec, tvec, args.square * 2.0)
            except Exception:
                pass
            cv2.imshow("pose preview", vis)
            cv2.waitKey(10)

    if kept < 3:
        print("Too few valid frames. Make sure the board is visible and well-lit.")
        sys.exit(2)

    # Outlier reject (keep within 1.5 × median RMS)
    rms_vals = np.array([r for _, r, _ in T_list], dtype=np.float64)
    med = float(np.median(rms_vals))
    keep_mask = rms_vals <= 1.5 * med
    kept_T = [T_list[i][2] for i in range(len(T_list)) if keep_mask[i]]
    kept_paths = [T_list[i][0] for i in range(len(T_list)) if keep_mask[i]]
    kept_rms = rms_vals[keep_mask]
    print(f"Frames used: {len(kept_T)}/{len(T_list)}  (median RMS={med:.3f}px)")

    # Average rotation/translation
    Rs = np.stack([T[:3,:3] for T in kept_T], axis=0)
    ts = np.stack([T[:3,3]  for T in kept_T], axis=0)
    Qs = np.stack([R_to_quat(R) for R in Rs], axis=0)
    q_mean = avg_quaternions(Qs)
    R_mean = quat_to_R(q_mean)
    t_mean = np.mean(ts, axis=0)

    T_base_cam = np.eye(4, dtype=np.float64)
    T_base_cam[:3,:3] = R_mean
    T_base_cam[:3, 3] = t_mean

    # Three.js-ready camera world matrix
    T_three = to_three_js_cam(T_base_cam)

    print("T_base_cam (OpenCV camera coords, +Z forward):\n", T_base_cam)
    print("T_three (Three.js camera world, looks -Z):\n", T_three)

    # Save
    out = {
        "T_base_cam": T_base_cam,
        "T_three": T_three,
        "frames_used": kept_paths,
        "rms_per_frame": kept_rms,
        "T_base_board": T_base_board,
        "Knew": Knew,
        "W": W, "H": H
    }
    np.savez(args.out, **out)
    print(f"Saved: {args.out}")

    if args.preview:
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
