from typing import Tuple, Optional, Any, List
import threading
import time
import cv2 as cv
import numpy as np


def compute_hist_distance(hist1, hist2, method="L2"):
    if hist1 is None or hist2 is None:
        return 0.0
        
    hist1 = np.asarray(hist1).flatten()
    hist2 = np.asarray(hist2).flatten()
    
    if method == "L1":
        return float(np.linalg.norm(hist1 - hist2, ord=1))
    elif method == "cosine":
        norm1 = np.linalg.norm(hist1)
        norm2 = np.linalg.norm(hist2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        return float(1.0 - np.dot(hist1, hist2) / (norm1 * norm2))
    elif method == "chi2":
        hist1 = hist1.astype(np.float32)
        hist2 = hist2.astype(np.float32)
        denom = hist1 + hist2
        denom[denom == 0] = 1
        return float(np.sum((hist1 - hist2) ** 2 / denom))
    else:
        return float(np.linalg.norm(hist1 - hist2))


# Pre-computed crop ROI for 640x480 input: (start_y, end_y, start_x, end_x)
DEFAULT_CROP_ROI = (192, 288, 236, 403)

class ThreadedCamera:
    """Continuously reads from VideoCapture in a background thread,
    always providing the most recent frame."""

    def __init__(self, cap: cv.VideoCapture):
        self._cap = cap
        self._frame = None
        self._ret = False
        self._has_new_frame = False
        self._lock = threading.Lock()
        self._stopped = False
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self):
        while not self._stopped:
            ret, frame = self._cap.read()
            with self._lock:
                self._ret = ret
                self._frame = frame
                self._has_new_frame = True
            if not ret:
                break

    def read(self):
        """Returns the latest frame and a boolean indicating if it is a new frame."""
        with self._lock:
            has_new = self._has_new_frame
            self._has_new_frame = False
            if self._frame is not None:
                return self._ret, self._frame, has_new
            return self._ret, None, has_new

    def release(self):
        self._stopped = True
        self._thread.join(timeout=2.0)
        self._cap.release()


def get_camera() -> Optional[ThreadedCamera]:
    # 1. Try GStreamer with explicit v4l2src — avoids Jetson Argus/CSI backend confusion
    #    and works when OpenCV is compiled with GStreamer (system install on Jetson).
    gst = (
        "v4l2src device=/dev/video0 io-mode=2 ! "
        "image/jpeg, width=640, height=480, framerate=30/1 ! "
        "nvjpegdec ! "
        "video/x-raw ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=true max-buffers=1"
    )
    cap = cv.VideoCapture(gst, cv.CAP_GSTREAMER)
    if cap.isOpened():
        print("[camera] Opened via GStreamer v4l2src pipeline")
        return ThreadedCamera(cap)
    cap.release()

    # 2. Explicit V4L2 backend — works with pip opencv-contrib-python (no GStreamer).
    cap = cv.VideoCapture(0, cv.CAP_V4L2)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    if cap.isOpened():
        print("[camera] Opened via CAP_V4L2 (index 0)")
        return ThreadedCamera(cap)
    cap.release()

    # 3. Generic fallback — let OpenCV pick any available backend.
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    if cap.isOpened():
        print("[camera] Opened via generic VideoCapture(0)")
        return ThreadedCamera(cap)
    cap.release()

    print("[camera] Cannot open camera — all backends failed")
    return None
