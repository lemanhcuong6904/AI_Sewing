"""Live camera view: find the bottom-left right-angle corner of the fabric and the move to the target.

Colour segmentation (segment.py) -> contour -> corner.find_bottom_left_corner (edges on the image
border are ignored, the rectangle may be rotated) -> corner.plan_translate_then_rotate.

Run (Windows):
    python cuonglm/view_camera/view_camera.py                 # camera 0
    python cuonglm/view_camera/view_camera.py image.png       # one image
    python cuonglm/view_camera/view_camera.py some_folder     # all images, n / p = next / previous

Fabric colour: automatic (the warmer class of the frame, independent of the lamp); left click forces a colour.
Add --static to ignore the fixed layer learnt by build_static_mask.py.
Keys:  left click = pick the fabric colour | c = back to automatic | m = colour mask
       SPACE = freeze / unfreeze (camera) | s = save the frame | q / ESC = quit
Sliders: tolerance of the colour match, opening (leak cut), closing (seam bridge).
"""
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cuonglm.curve_matcher.config import load_machine  # noqa: E402
from cuonglm.curve_matcher.corner import find_bottom_left_corner, plan_translate_then_rotate  # noqa: E402
from cuonglm.curve_matcher.edge_detect import sample_color  # noqa: E402
from segment import fit_static, load_static, occluder_from_static, segment_fabric  # noqa: E402

CAM_INDEX, WIDTH, HEIGHT = 0, 640, 480
USE_STATIC = "--static" in sys.argv          # use static_mask.png (a bed with a fixed layer of the fabric's colour)
sys.argv = [a for a in sys.argv if a != "--static"]
SETTINGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
WIN = "view_camera"
YELLOW, GREEN, RED, WHITE = (0, 210, 242), (77, 164, 20), (27, 27, 224), (255, 255, 255)


def open_camera(index, width, height):
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()
    raise RuntimeError(f"Cannot open camera {index}")


def load_image(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)    # non-ASCII paths on Windows


def pt(p):
    return int(round(float(p[0]))), int(round(float(p[1])))


def text(img, s, org, color=(30, 30, 30), scale=0.55):
    x, y = org
    (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + 4), (255, 255, 255), cv2.FILLED)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


class App:
    def __init__(self):
        self.fabric_bgr = None
        self.static = load_static() if USE_STATIC else None   # fixed layer of the bed (build_static_mask.py)
        self.show_mask = False
        self.frame = None
        self.click = None
        self.load_settings()

    def load_settings(self):
        try:
            with open(SETTINGS, encoding="utf-8") as f:
                s = json.load(f)
        except (OSError, ValueError, TypeError):
            pass

    def save_settings(self):
        try:
            with open(SETTINGS, "w", encoding="utf-8") as f:
                json.dump({"fabric_bgr": None if self.fabric_bgr is None else [float(v) for v in self.fabric_bgr]}, f)
        except OSError:
            pass

    def on_mouse(self, event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and self.frame is not None:
            self.fabric_bgr = sample_color(self.frame, x, y)
            self.save_settings()

    def process(self, bgr):
        """Returns the annotated image and a one-line status."""
        h, w = bgr.shape[:2]
        tol = max(1, cv2.getTrackbarPos("tolerance", WIN))
        opn = 2 * cv2.getTrackbarPos("open", WIN) + 1
        cls = 2 * cv2.getTrackbarPos("close", WIN) + 1
        static = fit_static(self.static, w, h)
        contour, mask, raw = segment_fabric(bgr, self.fabric_bgr, tol, opn, cls, static=static, static_grow=5)
        out = bgr.copy()
        if self.show_mask:
            out[raw > 0] = (0.5 * out[raw > 0] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)

        m = load_machine((w, h))
        B, L = m.basis, m.axis_len
        for u in (m.u1, m.u2):                                  # target: right angle at B
            cv2.arrowedLine(out, pt(B), pt(B + u * L), RED, 2, tipLength=0.08)
        cv2.circle(out, pt(B), 5, RED, 2)

        if contour is None:
            return out, "No fabric region. Click on the fabric to pick its colour."
        cv2.polylines(out, [contour.astype(np.int32).reshape(-1, 1, 2)], True, YELLOW, 1, cv2.LINE_AA)
        corner, why = find_bottom_left_corner(contour, (w, h), m.u1, m.u2, "auto", min_cut_edge_frac=0.04, long_h_min_theta=35.0,
                                              occluder=None if static is None else occluder_from_static(static))
        if corner is None:
            return out, why
        plan = plan_translate_then_rotate(corner, m.basis, m.pivot, "P", m.u1, m.u2, m.px_per_cm, m.max_rot_deg)
        cn = corner
        cv2.line(out, pt(cn.point), pt(cn.point + cn.h_dir * cn.h_len), GREEN, 3, cv2.LINE_AA)
        cv2.line(out, pt(cn.point), pt(cn.point + cn.v_dir * cn.v_len), GREEN, 3, cv2.LINE_AA)
        cv2.circle(out, pt(cn.point), 7, WHITE, 3)
        cv2.circle(out, pt(cn.point), 7, GREEN, 1)
        cv2.line(out, pt(cn.point), pt(B), (0, 140, 255), 1, cv2.LINE_AA)       # where it has to go
        msg = f"tx {plan.tx_mm:+.1f} mm  ty {plan.ty_mm:+.1f} mm  theta {plan.theta:+.1f} deg"
        if plan.warnings:
            msg += "  !" + plan.warnings[0]
        return out, msg


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    files, cap = [], None
    if arg and os.path.isdir(arg):
        for ext in ("png", "jpg", "jpeg", "bmp"):
            files += glob.glob(os.path.join(arg, f"*.{ext}"))
        files.sort()
    elif arg:
        files = [arg]
    if arg and not files:
        sys.exit(f"No image found: {arg}")
    if not files:
        cap = open_camera(CAM_INDEX, WIDTH, HEIGHT)

    app = App()
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, app.on_mouse)
    cv2.createTrackbar("tolerance", WIN, 25, 80, lambda _: None)
    cv2.createTrackbar("open", WIN, 3, 20, lambda _: None)          # kernel = 2 v + 1
    cv2.createTrackbar("close", WIN, 7, 30, lambda _: None)
    idx, frozen, saved = 0, False, 0

    while True:
        if cap is not None:
            if not frozen:
                ok, f = cap.read()
                if ok:
                    app.frame = f
        else:
            app.frame = load_image(files[idx])
            if app.frame is None:
                sys.exit(f"Cannot read {files[idx]}")
        if app.frame is not None:
            out, status = app.process(app.frame)
            text(out, status, (8, 22))
            if files:
                text(out, f"[{idx + 1}/{len(files)}] {os.path.abspath(files[idx])}", (8, out.shape[0] - 10), scale=0.45)
            cv2.imshow(WIN, out)
        k = cv2.waitKey(30 if cap is not None else 100) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord("m"):
            app.show_mask = not app.show_mask
        elif k == ord("c"):
            app.fabric_bgr = None
            app.save_settings()
        elif k == ord(" ") and cap is not None:
            frozen = not frozen
        elif k == ord("n") and files:
            idx = (idx + 1) % len(files)
        elif k == ord("p") and files:
            idx = (idx - 1) % len(files)
        elif k == ord("s") and app.frame is not None:
            name = f"view_camera_{time.strftime('%Y%m%d_%H%M%S')}_{saved}.png"
            cv2.imencode(".png", app.frame)[1].tofile(name)
            saved += 1
            print("[saved]", os.path.abspath(name))
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
