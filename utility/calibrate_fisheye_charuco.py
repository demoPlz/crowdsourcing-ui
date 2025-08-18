#!/usr/bin/env python3
"""
Calibrate fisheye intrinsics using a ChArUco board.

Outputs:
  - K (3x3), D (4x1) fisheye intrinsics
  - Knew (3x3) pinhole intrinsics for undistorted frames
  - Optional undistort maps
  - Reprojection error stats

Usage (example):
  python calibrate_fisheye_charuco.py \
    --images 'calib/front/*.png' \
    --cols 5 --rows 7 --square 0.035 --marker 0.026 \
    --dict 4X4_50 \
    --balance 0.4 \
    --out calib/intrinsics_front_1280x720.npz \
    --save-maps
"""
import argparse, glob, os
import numpy as np
import cv2

DICT_MAP = {
    "4X4_50":  cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "5X5_50":  cv2.aruco.DICT_5X5_50,
    "6X6_50":  cv2.aruco.DICT_6X6_50,
    "APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}

def reproj_errors_fisheye(K, D, obj_pts, img_pts, rvecs, tvecs):
    errs = []
    per_image = []
    for obj, img, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.fisheye.projectPoints(obj, r, t, K, D)
        proj = proj.reshape(-1, 2)
        e = np.linalg.norm(proj - img.reshape(-1,2), axis=1)
        errs.append(e)
        per_image.append(float(np.sqrt((e**2).mean())))
    errs = np.concatenate(errs) if errs else np.array([])
    rms = float(np.sqrt((errs**2).mean())) if errs.size else float('nan')
    med = float(np.median(errs)) if errs.size else float('nan')
    return rms, med, per_image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Glob for input images (quotes)")
    ap.add_argument("--cols", type=int, required=True, help="ChArUco squares across (columns)")
    ap.add_argument("--rows", type=int, required=True, help="ChArUco squares down (rows)")
    ap.add_argument("--square", type=float, required=True, help="Square size (meters)")
    ap.add_argument("--marker", type=float, required=True, help="Marker size (meters)")
    ap.add_argument("--dict", default="4X4_50", choices=DICT_MAP.keys())
    ap.add_argument("--min-corners", type=int, default=12, help="Min charuco corners per image to keep")
    ap.add_argument("--balance", type=float, default=0.4, help="0=crop, 1=keep FOV")
    ap.add_argument("--out", required=True, help="Output .npz path")
    ap.add_argument("--save-maps", action="store_true", help="Also save undistort maps (can be large)")
    ap.add_argument("--preview", action="store_true", help="Show detection and undistort preview windows")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.images))
    if not paths:
        raise SystemExit(f"No images match {args.images}")

    # ChArUco setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_MAP[args.dict])
    try:
        board = cv2.aruco.CharucoBoard((args.cols, args.rows), args.square, args.marker, aruco_dict)
    except TypeError:
        board = cv2.aruco.CharucoBoard_create(args.cols, args.rows, args.square, args.marker, aruco_dict)

    # --- OpenCV compatibility: get board object corners (Nx3) no matter the API variant ---
    def _board_object_corners(b):
        # Newer OpenCV: method; older: property
        if hasattr(b, "getChessboardCorners"):
            corners = b.getChessboardCorners()
        else:
            corners = b.chessboardCorners
        return np.asarray(corners, dtype=np.float64).reshape(-1, 3)
    board_obj = _board_object_corners(board)
    all_obj = []  # list of (Ni,1,3)
    all_img = []  # list of (Ni,1,2)
    img_size = None
    kept = 0

    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])  # (W,H)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
        if ids is None or len(ids) < 4:
            continue

        # Optional refinement (helps when there is clutter)
        cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedCorners=None)

        ok, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if not ok or ch_ids is None or ch_corners is None:
            continue

        if len(ch_ids) < args.min_corners:
            continue

        # Build matched 3D-2D correspondences on the ChArUco plane (Z=0 in board frame)
        # board.chessboardCorners: (N,3) in meters, indexed by charuco IDs
        obj = board_obj[ch_ids.flatten(), :].astype(np.float64)   # (N,3)
        imgp = ch_corners.reshape(-1, 2).astype(np.float64)                     # (N,2)

        all_obj.append(obj.reshape(-1,1,3))
        all_img.append(imgp.reshape(-1,1,2))
        kept += 1

        if args.preview:
            vis = img.copy()
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            cv2.aruco.drawDetectedCornersCharuco(vis, ch_corners, ch_ids, (0,255,0))
            cv2.imshow("charuco detect", vis)
            cv2.waitKey(1)

    if kept < 8:
        raise SystemExit(f"Too few valid images ({kept}). Capture more diverse views.")

    print(f"Using {kept} images, size = {img_size[0]}x{img_size[1]}")

    # Fisheye calibration
    K = np.eye(3, dtype=np.float64)
    D = np.zeros((4,1), dtype=np.float64)  # k1..k4
    rvecs, tvecs = [], []

    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
             cv2.fisheye.CALIB_CHECK_COND |
             cv2.fisheye.CALIB_FIX_SKEW)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-6)

    _res = cv2.fisheye.calibrate(
        objectPoints=all_obj,
        imagePoints=all_img,
        image_size=img_size,
        K=K, D=D, rvecs=rvecs, tvecs=tvecs,
        flags=flags,
        criteria=criteria
    )
    # OpenCV compatibility: some versions return just rms (float), others return (rms, K, D, rvecs, tvecs)
    if isinstance(_res, tuple):
        # Typical tuple form: (rms, K, D, rvecs, tvecs)
        rms = float(_res[0])
        if len(_res) >= 3 and _res[1] is not None: K = _res[1]
        if len(_res) >= 3 and _res[2] is not None: D = _res[2]
        if len(_res) >= 5:
            rvecs, tvecs = _res[3], _res[4]
    else:
        rms = float(_res)
        # K, D, rvecs, tvecs already filled in-place

    print(f"fisheye.calibrate RMS: {rms:.4f} px")
    print("K =\n", K)
    print("D^T =", D.ravel())

    # Extra reprojection metrics
    rms2, med, per_img = reproj_errors_fisheye(K, D, all_obj, all_img, rvecs, tvecs)
    print(f"Reprojection error  RMS={rms2:.4f} px,  median={med:.4f} px")
    if len(per_img) >= 1:
        print(f"Per-image RMS (first 10): {[round(x,3) for x in per_img[:10]]}")

    # Choose new pinhole intrinsics for undistorted view
    Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, img_size, np.eye(3), balance=float(args.balance)
    )
    print("Knew (for undistorted pinhole) =\n", Knew)

    # Build undistortion maps if requested
    map1 = map2 = None
    if args.save_maps:
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), Knew, img_size, cv2.CV_16SC2
        )

    # Save results
    save_dict = dict(
        model="fisheye",
        width=img_size[0], height=img_size[1],
        K=K, D=D, Knew=Knew, balance=float(args.balance),
        rms=float(rms2), median=float(med)
    )
    if map1 is not None and map2 is not None:
        save_dict["map1"] = map1
        save_dict["map2"] = map2

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **save_dict)
    print(f"Saved: {args.out}")

    if args.preview:
        # Side-by-side preview: original (distorted) vs undistorted of the SAME image.
        # Prefer the best (lowest per-image RMS); fallback to the last image.
        if per_img:
            idx = int(-1)
        else:
            idx = -1
        src_path = paths[idx]
        print(f"Previewing side-by-side with: {src_path}")

        src = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if src is not None:
            if map1 is not None and map2 is not None:
                und = cv2.remap(src, map1, map2, interpolation=cv2.INTER_LINEAR)
            else:
                und = cv2.fisheye.undistortImage(src, K, D, Knew=Knew)

            # Label for clarity
            src_l = src.copy()
            und_l = und.copy()
            cv2.putText(src_l, "original (distorted)", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(und_l, "undistorted (Knew)", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            combo = np.hstack([src_l, und_l])
            cv2.imshow("original | undistorted", combo)

            out_preview = os.path.splitext(args.out)[0] + "_preview_side_by_side.png"
            cv2.imwrite(out_preview, combo)
            print(f"Saved preview image: {out_preview}")
            print("Press any key to close previews...")
            cv2.waitKey(0)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
