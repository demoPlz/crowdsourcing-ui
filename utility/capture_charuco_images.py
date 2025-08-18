#!/usr/bin/env python3
# capture_charuco_images.py
# Preview one or more cameras and save raw frames for calibration (distorted images).
# Saves to calib/<camera-label>/img_0001.png ... and writes a session.json per camera.

import argparse, os, re, time, json, glob
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

DICT_MAP = {
    "4X4_50":  cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "5X5_50":  cv2.aruco.DICT_5X5_50,
    "6X6_50":  cv2.aruco.DICT_6X6_50,
    "APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}

def parse_args():
    ap = argparse.ArgumentParser(description="Capture ChArUco calibration images (distorted) from one or more cameras.")
    ap.add_argument("--cams", nargs="+", required=True,
                    help="List of camera identifiers: numeric indices (e.g., 0 2) or device paths (/dev/v4l/by-id/...-video-index0).")
    ap.add_argument("--width",  type=int, default=None, help="Request frame width (e.g., 1280).")
    ap.add_argument("--height", type=int, default=None, help="Request frame height (e.g., 720).")
    ap.add_argument("--fourcc", default="YUYV", help="FourCC (default YUYV).")
    ap.add_argument("--fps",    type=int, default=None, help="Request FPS (optional).")
    # ChArUco board (for live overlay ONLY; saved images are always raw)
    ap.add_argument("--cols",   type=int, default=5, help="ChArUco squares across (columns).")
    ap.add_argument("--rows",   type=int, default=7, help="ChArUco squares down (rows).")
    ap.add_argument("--square", type=float, default=0.035, help="Square size (meters).")
    ap.add_argument("--marker", type=float, default=0.026, help="Marker size (meters).")
    ap.add_argument("--dict",   default="4X4_50", choices=DICT_MAP.keys(), help="ArUco dictionary.")
    ap.add_argument("--outdir", default="calib", help="Base output directory.")
    ap.add_argument("--interval", type=float, default=1.5, help="Seconds between auto captures when 'n' toggled.")
    ap.add_argument("--no-overlay", action="store_true", help="Disable ChArUco overlay for max speed.")
    return ap.parse_args()

def to_index_or_path(s: str):
    return int(s) if re.fullmatch(r"\d+", s) else s

def try_set(prop, val, cap):
    if val is None: return
    cap.set(prop, val)

def open_capture(cam_id, fourcc, width, height, fps):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    # Reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Codec & format hints
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    try_set(cv2.CAP_PROP_FRAME_WIDTH,  width, cap)
    try_set(cv2.CAP_PROP_FRAME_HEIGHT, height, cap)
    try_set(cv2.CAP_PROP_FPS,          fps,   cap)
    # Warm up a bit
    for _ in range(5): cap.read()
    ok, frame = cap.read()
    return cap if ok else None

def read_sys(path):
    try:
        with open(path, "r", encoding="utf-8") as f: return f.read().strip()
    except Exception:
        return ""

def video_sys_base_from_cam(cam_id):
    # Find /sys/class/video4linux/videoN base
    if isinstance(cam_id, int):
        return f"/sys/class/video4linux/video{cam_id}"
    # Resolve /dev/v4l/by-*/... → /dev/videoN
    dev = os.path.realpath(cam_id)
    m = re.search(r"/dev/video(\d+)$", dev)
    if not m: return None
    return f"/sys/class/video4linux/video{m.group(1)}"

def find_usb_device_dir(sys_base):
    # Walk up from .../videoN/device to parent with idVendor/idProduct (USB device root)
    if not sys_base: return None
    cur = os.path.realpath(os.path.join(sys_base, "device"))
    for _ in range(12):
        if os.path.exists(os.path.join(cur, "idVendor")) and os.path.exists(os.path.join(cur, "idProduct")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur: break
        cur = parent
    return None

def stable_label_for_cam(cam_id):
    """Prefer /dev/v4l/by-id name, else serial, else by-path, else index."""
    # 1) by-id symlink name
    try:
        for name in os.listdir("/dev/v4l/by-id"):
            full = os.path.join("/dev/v4l/by-id", name)
            if os.path.islink(full) and os.path.realpath(full) == (cam_id if isinstance(cam_id,str) else f"/dev/video{cam_id}"):
                return name
    except FileNotFoundError:
        pass
    # 2) serial from sysfs
    sys_base = video_sys_base_from_cam(cam_id)
    usb_root = find_usb_device_dir(sys_base) if sys_base else None
    serial   = read_sys(os.path.join(usb_root, "serial")) if usb_root else ""
    if serial: return f"serial_{serial}"
    # 3) by-path symlink name
    try:
        for name in os.listdir("/dev/v4l/by-path"):
            full = os.path.join("/dev/v4l/by-path", name)
            if os.path.islink(full) and os.path.realpath(full) == (cam_id if isinstance(cam_id,str) else f"/dev/video{cam_id}"):
                return f"path_{name}"
    except FileNotFoundError:
        pass
    # 4) fallback
    return f"idx_{cam_id}"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def draw_overlay(frame, txt, color=(0,0,255)):
    cv2.putText(frame, txt, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

def build_charuco(rows, cols, square, marker, dict_name):
    ar_dict = cv2.aruco.getPredefinedDictionary(DICT_MAP[dict_name])
    try:
        board = cv2.aruco.CharucoBoard((cols, rows), square, marker, ar_dict)
    except TypeError:
        board = cv2.aruco.CharucoBoard_create(cols, rows, square, marker, ar_dict)
    return ar_dict, board

def detect_charuco(frame_gray, ar_dict, board):
    corners, ids, _ = cv2.aruco.detectMarkers(frame_gray, ar_dict)
    if ids is None or len(ids) < 3:
        return None
    cv2.aruco.refineDetectedMarkers(frame_gray, board, corners, ids, rejectedCorners=None)
    ok, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, frame_gray, board)
    if ok and ch_ids is not None and ch_corners is not None:
        return (corners, ids, ch_corners, ch_ids)
    return None

def main():
    args = parse_args()

    cams_in = [to_index_or_path(s) for s in args.cams]
    cams = []
    for cam_id in cams_in:
        cap = open_capture(cam_id, args.fourcc, args.width, args.height, args.fps)
        if not cap:
            print(f"[!] Could not open camera {cam_id}")
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        label = stable_label_for_cam(cam_id)
        out_dir = Path(args.outdir) / f"{label}_{w}x{h}"
        ensure_dir(out_dir)
        # Session metadata file (once)
        meta_path = out_dir / "session.json"
        if not meta_path.exists():
            meta = {
                "label": label, "device": str(cam_id),
                "width": w, "height": h, "fourcc": args.fourcc, "fps_req": args.fps,
                "board": {"cols": args.cols, "rows": args.rows, "square_m": args.square, "marker_m": args.marker, "dict": args.dict},
                "started_at": datetime.now().isoformat(timespec="seconds")
            }
            with open(meta_path, "w", encoding="utf-8") as f: json.dump(meta, f, indent=2)
        # Next index
        existing = sorted(glob.glob(str(out_dir / "img_*.png")))
        next_idx = len(existing) + 1
        cams.append({
            "id": cam_id, "cap": cap, "label": label, "dir": out_dir,
            "size": (w,h), "next_idx": next_idx, "window": f"{label} ({w}x{h})"
        })

    if not cams:
        print("No cameras opened.")
        return

    # Windows
    for c in cams:
        cv2.namedWindow(c["window"], cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow(c["window"], c["size"][0]//2, c["size"][1]//2)

    # ChArUco helpers
    ar_dict, board = (None, None)
    if not args.no_overlay:
        ar_dict, board = build_charuco(args.rows, args.cols, args.square, args.marker, args.dict)

    focused = 0  # index in cams
    auto = False
    last_auto = 0.0

    print("Controls: [TAB]=focus  [c]=capture focused  [a]=capture all  [n]=toggle auto  [q]=quit")
    print("Tips: Fill edges/corners of the frame, vary tilt/roll & distance, keep sharp.")

    while True:
        # Read and display each
        now = time.time()
        for i, c in enumerate(cams):
            ok, frame = c["cap"].read()
            if not ok:
                continue
            vis = frame.copy()

            if ar_dict is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                det = detect_charuco(gray, ar_dict, board)
                if det is not None:
                    corners, ids, ch_c, ch_ids = det
                    cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                    cv2.aruco.drawDetectedCornersCharuco(vis, ch_c, ch_ids, (0,255,0))
                    draw_overlay(vis, f"{c['label']}  corners:{len(ch_ids)}", (0,255,0))
                else:
                    draw_overlay(vis, f"{c['label']}  corners:0", (0,0,255))
            else:
                draw_overlay(vis, f"{c['label']}", (0,0,255))

            if i == focused:
                cv2.rectangle(vis, (8,8), (220, 52), (0,255,255), 2)

            cv2.imshow(c["window"], vis)

        # Auto capture?
        if auto and (now - last_auto) >= args.interval:
            for c in cams:
                save_path = c["dir"] / f"img_{c['next_idx']:04d}.png"
                ok, frame = c["cap"].read()
                if ok:
                    cv2.imwrite(str(save_path), frame)
                    c["next_idx"] += 1
            last_auto = now
            print(f"[auto] captured to all cams at {datetime.now().strftime('%H:%M:%S')}")

        key = cv2.waitKey(1) & 0xFFFF
        if key == ord('q'):
            break
        elif key == 9:  # TAB
            focused = (focused + 1) % len(cams)
        elif key == ord('c'):  # capture focused
            c = cams[focused]
            ok, frame = c["cap"].read()
            if ok:
                save_path = c["dir"] / f"img_{c['next_idx']:04d}.png"
                cv2.imwrite(str(save_path), frame)
                c["next_idx"] += 1
                print(f"[saved] {save_path}")
        elif key == ord('a'):  # capture all
            for c in cams:
                ok, frame = c["cap"].read()
                if ok:
                    save_path = c["dir"] / f"img_{c['next_idx']:04d}.png"
                    cv2.imwrite(str(save_path), frame)
                    c["next_idx"] += 1
                    print(f"[saved] {save_path}")
        elif key == ord('n'):  # toggle auto
            auto = not auto
            last_auto = 0.0
            print(f"[auto] {'ON' if auto else 'OFF'} (interval={args.interval}s)")

    # Cleanup
    for c in cams:
        c["cap"].release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
