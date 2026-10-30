"""Live fabric segmentation test: camera at 30 fps -> ONNX UNet (onnxruntime) -> mask overlay.

Run (Ubuntu / Windows; needs only opencv, numpy, onnxruntime):
    python cuonglm/segmentation/view_segmentation.py

The camera is read in its own thread at full speed (30 fps); the model runs on the newest frame, so a
slow model drops frames instead of building a delay. The window shows the camera fps and the model fps.
Keys:  m = overlay / mask only | SPACE = freeze | s = save frame | q / ESC = quit
"""
import os
import sys
import threading
import time

import cv2
import numpy as np
import onnxruntime as ort

# ---- config
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "experiments", "unet_fabric_segmentation", "best.onnx")
CAM_INDEX = 0
WIDTH, HEIGHT, CAM_FPS = 640, 480, 30
THRESHOLD = 0.5
ALPHA = 0.45                    # opacity of the mask colour
MASK_BGR = (20, 164, 77)        # green
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
WIN = "view_segmentation"


def open_camera(index, width, height, fps):
    backends = (cv2.CAP_V4L2, cv2.CAP_ANY) if sys.platform.startswith("linux") else \
               (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # raw YUYV cannot reach 30 fps
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()
    raise RuntimeError(f"Cannot open camera {index}")


class Grabber(threading.Thread):
    """Reads the camera continuously; `latest()` returns the newest frame and its counter."""

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap, self.lock = cap, threading.Lock()
        self.frame, self.count, self.fps, self.running = None, 0, 0.0, True

    def run(self):
        t0, n = time.perf_counter(), 0
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self.lock:
                self.frame, self.count = f, self.count + 1
            n += 1
            now = time.perf_counter()
            if now - t0 >= 1.0:
                self.fps, t0, n = n / (now - t0), now, 0

    def latest(self):
        with self.lock:
            return self.frame, self.count


class Segmenter:
    def __init__(self, path):
        avail = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        self.sess = ort.InferenceSession(path, providers=providers)
        inp = self.sess.get_inputs()[0]
        self.name, (_, _, self.h, self.w) = inp.name, inp.shape
        print(f"[seg] {os.path.basename(path)} input {self.w}x{self.h}, providers {self.sess.get_providers()}")

    def __call__(self, bgr):
        """bgr uint8 (any size) -> fabric probability (h, w) float32 at the model size."""
        if bgr.shape[1] != self.w or bgr.shape[0] != self.h:
            bgr = cv2.resize(bgr, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        x = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None]
        return self.sess.run(None, {self.name: np.ascontiguousarray(x)})[0][0, 0]


def render(bgr, prob, mask_only):
    mask = (prob >= THRESHOLD).astype(np.uint8)
    if mask.shape != bgr.shape[:2]:
        mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    if mask_only:
        return cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR), mask
    out = bgr.copy()
    sel = mask > 0
    out[sel] = ((1 - ALPHA) * out[sel] + ALPHA * np.array(MASK_BGR)).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 210, 242), 2, cv2.LINE_AA)
    return out, mask


def put(img, s, org):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    if not os.path.isfile(MODEL):
        sys.exit(f"Model not found: {MODEL}\nExport it first: python cuonglm/segmentation/export_onnx.py")
    seg = Segmenter(MODEL)
    cap = open_camera(CAM_INDEX, WIDTH, HEIGHT, CAM_FPS)
    grab = Grabber(cap)
    grab.start()
    print("[cam] requested %dx%d @ %d fps, driver reports %.0f fps" % (WIDTH, HEIGHT, CAM_FPS, cap.get(cv2.CAP_PROP_FPS)))

    mask_only, frozen, saved = False, False, 0
    last_count, infer_ms, infer_fps = 0, 0.0, 0.0
    shown = None
    t_fps, n_fps = time.perf_counter(), 0
    try:
        while True:
            frame, count = grab.latest()
            if frame is None or (count == last_count and shown is not None) or frozen:
                if shown is not None:
                    cv2.imshow(WIN, shown)
                k = cv2.waitKey(2) & 0xFF
            else:
                last_count = count
                t = time.perf_counter()
                prob = seg(frame)
                infer_ms = (time.perf_counter() - t) * 1000
                view, mask = render(frame, prob, mask_only)
                n_fps += 1
                now = time.perf_counter()
                if now - t_fps >= 1.0:
                    infer_fps, t_fps, n_fps = n_fps / (now - t_fps), now, 0
                put(view, f"camera {grab.fps:4.1f} fps | model {infer_fps:4.1f} fps ({infer_ms:.0f} ms) | "
                          f"fabric {float(mask.mean()):.0%}", (8, 22))
                shown = view
                cv2.imshow(WIN, shown)
                k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                print(f"[seg] stopped: key {k}")
                break
            elif k == ord("m"):
                mask_only = not mask_only
            elif k == ord(" "):
                frozen = not frozen
            elif k == ord("s") and shown is not None:
                path = os.path.join(SAVE_DIR, f"seg_{time.strftime('%Y%m%d_%H%M%S')}_{saved}.png")
                cv2.imwrite(path, shown)
                saved += 1
                print("[saved]", path)
    except KeyboardInterrupt:
        print("[seg] stopped: Ctrl+C")
    finally:
        grab.running = False
        grab.join(timeout=1.0)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
