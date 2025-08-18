# make_charuco_pdf.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------- Board parameters (edit if you change design) ----------
COLS, ROWS       = 5, 7                 # chessboard squares across, down
SQUARE_MM        = 35.0                 # edge of a chess square (mm)
MARKER_MM        = 26.0                 # ArUco marker edge (mm)
DICT_NAME        = cv2.aruco.DICT_4X4_50

# ---------- Page setup ----------
USE_US_LETTER    = True                 # set False for A4
DPI              = 600                  # bitmap DPI inside the PDF (high)
MARGIN_MM        = 10                   # white border around the board

if USE_US_LETTER:
    PAGE_W_MM, PAGE_H_MM = 215.9, 279.4   # 8.5" x 11"
else:
    PAGE_W_MM, PAGE_H_MM = 210.0, 297.0   # A4

# ---------- Derived dimensions ----------
board_w_mm = COLS * SQUARE_MM
board_h_mm = ROWS * SQUARE_MM

# Check it fits
assert board_w_mm + 2*MARGIN_MM <= PAGE_W_MM + 1e-6, "Board too wide for page."
assert board_h_mm + 2*MARGIN_MM <= PAGE_H_MM + 1e-6, "Board too tall for page."

# Pixel sizes for the bitmap we’ll place on the PDF
board_w_px = int(round(board_w_mm / 25.4 * DPI))
board_h_px = int(round(board_h_mm / 25.4 * DPI))
margin_px  = int(round(MARGIN_MM  / 25.4 * DPI))

# Make the Charuco board
aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_NAME)
try:
    board = cv2.aruco.CharucoBoard((COLS, ROWS), SQUARE_MM/1000.0, MARKER_MM/1000.0, aruco_dict)
except TypeError:
    # Older OpenCV uses different constructor signature (same semantics)
    board = cv2.aruco.CharucoBoard_create(COLS, ROWS, SQUARE_MM/1000.0, MARKER_MM/1000.0, aruco_dict)

# Render board bitmap (high-res)
try:
    img = board.generateImage((board_w_px, board_h_px), marginSize=0, borderBits=1)
except AttributeError:
    img = board.draw((board_w_px, board_h_px), marginSize=0, borderBits=1)

# Place onto a white page canvas, centered
page_w_px = int(round(PAGE_W_MM / 25.4 * DPI))
page_h_px = int(round(PAGE_H_MM / 25.4 * DPI))
canvas = 255 * np.ones((page_h_px, page_w_px), dtype=np.uint8)

off_x = (page_w_px - board_w_px) // 2
off_y = (page_h_px - board_h_px) // 2
canvas[off_y:off_y+board_h_px, off_x:off_x+board_w_px] = img

# Save as PDF at the correct physical size
fig_w_in = PAGE_W_MM / 25.4
fig_h_in = PAGE_H_MM / 25.4
fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=DPI)
ax = plt.axes([0,0,1,1])
ax.imshow(canvas, cmap='gray', vmin=0, vmax=255)
ax.axis('off')
plt.savefig("charuco_5x7_35mm_USletter.pdf", format="pdf", dpi=DPI)
plt.close(fig)

print("Wrote charuco_5x7_35mm_USletter.pdf")
print(f"Physical board area: {board_w_mm:.1f} mm × {board_h_mm:.1f} mm")
