#!/usr/bin/env python3
"""
Self-contained webcam viewer with device picker.

Requirements:
    pip install opencv-python pillow
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk


def list_cameras(max_index: int = 10) -> list[int]:
    """
    Probe device indices [0, max_index) and return those that open successfully.
    """
    available = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
        if cap.isOpened():
            available.append(idx)
            cap.release()
    return available


class WebcamViewer(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Webcam Viewer")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Discover cameras once at startup
        self.cam_indices = list_cameras()
        if not self.cam_indices:
            messagebox.showerror("No Camera Found", "No webcams detected.")
            self.destroy()
            return

        # --- UI widgets ----------------------------------------------------
        ttk.Label(self, text="Select camera:").grid(row=0, column=0, padx=5, pady=5)

        self.cam_var = tk.IntVar(value=self.cam_indices[0])
        self.selector = ttk.Combobox(
            self,
            textvariable=self.cam_var,
            values=self.cam_indices,
            state="readonly",
            width=5,
        )
        self.selector.grid(row=0, column=1, padx=5, pady=5)
        self.selector.bind("<<ComboboxSelected>>", self.change_camera)

        self.canvas = tk.Label(self, borderwidth=2, relief="sunken")
        self.canvas.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

        # --- capture state --------------------------------------------------
        self.cap = None
        self.frame_delay_ms = 15  # ~66 FPS cap; adjust if CPU-heavy
        self.change_camera()      # open default cam and start loop

    # ---------------------------------------------------------------------#
    # Capture / UI helpers                                                 #
    # ---------------------------------------------------------------------#
    def change_camera(self, *_):
        """(Re)open the selected camera index."""
        self.stop_camera()
        idx = int(self.cam_var.get())
        self.cap = cv2.VideoCapture(
            idx, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
        )
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", f"Unable to open camera {idx}.")
            self.after(100, self.stop_camera)
            return
        self.after(self.frame_delay_ms, self.update_frame)

    def update_frame(self):
        """Grab a frame and display it, then reschedule itself."""
        if not (self.cap and self.cap.isOpened()):
            return

        ret, frame = self.cap.read()
        if ret:
            # Convert BGR -> RGB -> PIL -> Tk
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(img)

            # Keep reference to avoid GC
            self.canvas.imgtk = imgtk
            self.canvas.configure(image=imgtk)

        self.after(self.frame_delay_ms, self.update_frame)

    def stop_camera(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None

    def on_close(self):
        self.stop_camera()
        self.destroy()


if __name__ == "__main__":
    WebcamViewer().mainloop()
