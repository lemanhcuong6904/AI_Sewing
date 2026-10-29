"""Fabric corner -> machine target (PyQt6), colour segmentation version (no DexiNed).

1. Open an image (or grab a camera frame) with a piece of fabric -> colour mask -> fabric contour.
2. The target comes from the machine constants (utils/constants.py, gui_config.json): a right angle
   at B with arms along U1 and U2.  It is drawn on the image.
3. The bottom-left (approximately right-angle) corner of the fabric is found from the contour by
   fitting two straight edges.  The move that brings it onto the target is computed and animated:
   first translate by (tx, ty), then rotate by theta about the rotation centre.  tx, ty are in mm.

Run:  python cuonglm/view_camera/app.py [image]
"""
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (QColor, QFont, QFontDatabase, QImage, QPainter, QPen, QPixmap, QPolygonF)
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QHBoxLayout,
                             QLabel, QPushButton, QScrollArea, QSlider, QVBoxLayout, QWidget)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from cuonglm.curve_matcher.config import load_machine  # noqa: E402
from cuonglm.curve_matcher.corner import find_bottom_left_corner, plan_translate_then_rotate  # noqa: E402
from cuonglm.curve_matcher.edge_detect import bgr_to_lab, sample_color  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segment import (build_static_mask, fit_static, load_static, occluder_from_static,  # noqa: E402
                     save_static, segment_fabric)

# Computer Modern (the TeX font) ships with matplotlib, which this project already depends on.
SERIF, MONO = "cmr10", "cmtt10"
CAM_INDEX = 0
STATIC_GROW = 5                  # px the fixed layer is grown by before it is cut out of the colour mask
SETTINGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")


def load_computer_modern():
    global SERIF, MONO
    try:
        import matplotlib
        d = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
        fams = []
        for name in ("cmr10", "cmtt10"):
            fid = QFontDatabase.addApplicationFont(os.path.join(d, name + ".ttf"))
            fams.append(QFontDatabase.applicationFontFamilies(fid)[0] if fid >= 0 else None)
        SERIF, MONO = fams[0] or "serif", fams[1] or "monospace"
    except Exception as e:  # font is cosmetic; never block the tool on it
        print("[font] Computer Modern not available:", e)
        SERIF, MONO = "serif", "monospace"


def polygon(pts):
    return QPolygonF([QPointF(float(x), float(y)) for x, y in pts])


def smoothstep(x):
    return x * x * (3 - 2 * x)


# ------------------------------------------------------------------ canvas
class Canvas(QWidget):
    color_clicked = pyqtSignal(float, float)
    anim_changed = pyqtSignal()

    ROT_MS, TRANS_MS, HOLD_MS = 1800, 1800, 350

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.image = None            # QImage
        self.fabric_pix = None       # QPixmap cut-out of the fabric (ARGB)
        self.contour_poly = None
        self.machine = None          # config.Machine: target B, axes U1/U2, pivot P
        self.res = None              # corner.Plan being shown
        self.pick_mode = None        # None | "color" (pick fabric colour)
        self.match_pix = None        # QPixmap: pixels matching the fabric colour
        self.show_match = False
        self.trans_f = 0.0           # step 1: translate
        self.rot_f = 0.0             # step 2: rotate
        self.speed = 1.0
        self.queue, self.cur, self.t0, self.hold_until = [], None, 0.0, 0.0
        self.timer = QTimer(self, interval=16)
        self.timer.timeout.connect(self._tick)

    # ---- data
    def set_image(self, bgr):
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self.fabric_pix = self.contour_poly = self.res = self.match_pix = None
        self.reset_anim()
        self.update()

    def set_machine(self, machine):
        self.machine = machine
        self.update()

    def set_match(self, m):
        if m is None:
            self.match_pix = None
        else:
            h, w = m.shape
            rgba = np.zeros((h, w, 4), np.uint8)
            rgba[..., 0], rgba[..., 2], rgba[..., 3] = 255, 255, np.where(m > 0, 120, 0)
            self.match_pix = QPixmap.fromImage(QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy())
        self.update()

    def set_fabric(self, bgr, contour, mask):
        if contour is None:
            self.fabric_pix = self.contour_poly = None
        else:
            h, w = mask.shape
            rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
            rgba[..., 3] = mask
            self.fabric_pix = QPixmap.fromImage(QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy())
            self.contour_poly = polygon(contour)
        self.set_result(None)

    def set_result(self, res):
        self.res = res
        self.reset_anim()
        self.update()

    # ---- animation: step 1 translate, step 2 rotate
    def reset_anim(self):
        self.timer.stop()
        self.queue, self.cur = [], None
        self.rot_f = self.trans_f = 0.0
        self.anim_changed.emit()
        self.update()

    def play(self, steps):
        if self.res is None:
            return
        self.queue = list(steps)
        self._next()

    def _next(self):
        if not self.queue:
            self.cur = None
            self.timer.stop()
            self.anim_changed.emit()
            return
        self.cur = self.queue.pop(0)
        self.t0 = time.perf_counter()
        self.hold_until = 0.0
        self.timer.start()
        self.anim_changed.emit()

    def _tick(self):
        now = time.perf_counter()
        if self.hold_until:
            if now >= self.hold_until:
                self._next()
            return
        dur = (self.ROT_MS if self.cur == "rot" else self.TRANS_MS) / self.speed / 1000.0
        f = min(1.0, (now - self.t0) / dur)
        if self.cur == "rot":
            self.rot_f = f
        else:
            self.trans_f = f
        if f >= 1.0:
            self.hold_until = now + self.HOLD_MS / self.speed / 1000.0
        self.update()

    def step_text(self):
        if self.res is None:
            return ""
        r = self.res
        if self.cur == "rot" or self.rot_f > 0:
            return f"Step 2 / 2: rotate {r.theta * smoothstep(self.rot_f):+.2f} deg"
        if self.cur == "trans" or self.trans_f > 0:
            k = smoothstep(self.trans_f)
            return f"Step 1 / 2: translate ({r.tx_mm * k:+.1f}, {r.ty_mm * k:+.1f}) mm"
        return "Ready"

    # ---- view mapping
    def _view(self):
        iw, ih = self.image.width(), self.image.height()
        s = min(self.width() / iw, self.height() / ih)
        return s, (self.width() - iw * s) / 2, (self.height() - ih * s) / 2

    def to_img(self, pos):
        s, ox, oy = self._view()
        return (pos.x() - ox) / s, (pos.y() - oy) / s

    # ---- mouse: only the click-to-pick colour mode
    def mousePressEvent(self, e):
        if self.image is None or e.button() != Qt.MouseButton.LeftButton:
            return
        x, y = self.to_img(e.position())
        if self.pick_mode == "color":
            self.color_clicked.emit(x, y)

    # ---- painting
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor("#e9e9e6"))
        if self.image is None:
            p.setPen(QColor("#666"))
            p.setFont(QFont(SERIF, 18))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open an image with a piece of fabric")
            return
        s, ox, oy = self._view()
        p.translate(ox, oy)
        p.scale(s, s)
        W, H = self.image.width(), self.image.height()
        p.drawImage(0, 0, self.image)
        p.setClipRect(QRectF(0, 0, W, H))        # the machine frame may lie (partly) outside the image

        def pen(color, w):
            q = QPen(QColor(color), w)
            q.setCosmetic(True)
            q.setCapStyle(Qt.PenCapStyle.RoundCap)
            q.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            return q

        def line(a, b):
            p.drawLine(QPointF(float(a[0]), float(a[1])), QPointF(float(b[0]), float(b[1])))

        if self.show_match and self.match_pix is not None:
            p.drawPixmap(0, 0, self.match_pix)

        r = self.res
        if r is not None and self.fabric_pix is not None:
            # ghost of the original position, then the fabric cut-out that moves: translate, then rotate
            p.fillRect(QRectF(0, 0, W, H), QColor(0, 0, 0, 150))
            p.save()
            pc, t = r.rot_center, r.t_px
            kt, kr = smoothstep(self.trans_f), smoothstep(self.rot_f)
            p.translate(float(pc[0]), float(pc[1]))
            p.rotate(r.theta * kr)
            p.translate(float(-pc[0] + t[0] * kt), float(-pc[1] + t[1] * kt))
            p.drawPixmap(0, 0, self.fabric_pix)
            p.setPen(pen("#f2c200", 1.5))
            p.drawPolyline(self.contour_poly)
            cn = r.corner
            p.setPen(pen("#14a44d", 4.5))                       # the two fitted edges and the corner
            line(cn.point, cn.point + cn.h_dir * cn.h_fit)
            line(cn.point, cn.point + cn.v_dir * cn.v_fit)
            for color, wd in (("#ffffff", 7), ("#14a44d", 3)):
                p.setPen(pen(color, wd))
                p.drawEllipse(QPointF(float(cn.point[0]), float(cn.point[1])), 5 / s, 5 / s)
            p.restore()
        elif self.contour_poly is not None:
            p.setPen(pen("#f2c200", 2))
            p.drawPolyline(self.contour_poly)

        # target: right angle at B, arms along U1 and U2
        m = self.machine
        if m is not None:
            B, L = m.basis, m.axis_len
            hd = 14 / s
            p.setPen(pen("#e01b1b", 4))
            for u in (m.u1, m.u2):
                tip = B + u * L
                n = np.array([-u[1], u[0]])
                line(B, tip)
                line(tip, tip - u * hd + n * hd * 0.5)
                line(tip, tip - u * hd - n * hd * 0.5)
            for color, wd in (("#ffffff", 8), ("#e01b1b", 4)):
                p.setPen(pen(color, wd))
                p.drawEllipse(QPointF(float(B[0]), float(B[1])), 4 / s, 4 / s)

        # captions (screen space)
        p.setClipping(False)
        p.resetTransform()
        if m is not None:
            p.setFont(QFont(SERIF, 14))
            sx, sy = ox + float(m.basis[0]) * s, oy + float(m.basis[1]) * s
            label = f"B ({m.basis[0]:.0f}, {m.basis[1]:.0f})"
            fm = p.fontMetrics()
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 215))              # readable on dark and bright areas
            p.drawRoundedRect(QRectF(sx + 8, sy + 6, fm.horizontalAdvance(label) + 12, fm.height() + 4), 4, 4)
            p.setPen(QColor("#b3261e"))
            p.drawText(QPointF(sx + 14, sy + 6 + 2 + fm.ascent()), label)
            for name, u in (("U1", m.u1), ("U2", m.u2)):
                tip = m.basis + u * m.axis_len
                if 0 <= tip[0] <= W and 0 <= tip[1] <= H:
                    p.drawText(QPointF(ox + float(tip[0]) * s + 8, oy + float(tip[1]) * s - 6), name)
        txt = self.step_text()
        if txt:
            p.setFont(QFont(SERIF, 16))
            fm = p.fontMetrics()
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 220))
            p.drawRoundedRect(QRectF(12, 12, fm.horizontalAdvance(txt) + 24, fm.height() + 12), 6, 6)
            p.setPen(QColor("#222"))
            p.drawText(QPointF(24, 12 + 6 + fm.ascent()), txt)


# ------------------------------------------------------------------ window
class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fabric corner matcher (colour)")
        self.resize(1360, 900)
        self.bgr = None
        self.contour = self.mask = None
        self.plan = None
        self.machine = None
        self.extract_msg = ""
        self.fabric_bgr = None       # fabric colour picked by the user (BGR); None = auto (bright side)
        self.match = None
        self.tol_default = 25
        self.static = load_static()      # fixed fabric-coloured layer of the bed (or None)
        self.load_settings()

        self.canvas = Canvas()
        self.canvas.color_clicked.connect(self.on_color)
        self.canvas.anim_changed.connect(self.update_buttons)

        def btn(text, slot):
            b = QPushButton(text)
            b.clicked.connect(slot)
            return b

        self.b_open = btn("Open image...", self.open_image)
        self.b_cam = btn("Grab camera frame", self.grab_camera)
        self.path_label = QLabel("No image loaded.")
        self.path_label.setWordWrap(True)
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.path_label.setStyleSheet("font-family:Consolas,'Courier New',monospace; font-size:10pt;")
        self.b_color = QPushButton("Pick fabric colour (click)")
        self.b_color.setCheckable(True)
        self.b_color.toggled.connect(self.set_pick)
        self.b_auto = btn("Auto (independent of lighting)", self.clear_color)
        self.swatch = QLabel()
        self.swatch.setFixedSize(46, 30)
        self.color_text = QLabel()
        self.color_text.setWordWrap(True)
        self.tol_label = QLabel()
        self.tol = QSlider(Qt.Orientation.Horizontal, minimum=5, maximum=80, value=self.tol_default)
        self.tol.valueChanged.connect(self.on_slider)
        self.op_label = QLabel()
        self.op = QSlider(Qt.Orientation.Horizontal, minimum=1, maximum=20, value=3)     # kernel = 2*v+1
        self.op.valueChanged.connect(self.on_slider)
        self.cl_label = QLabel()
        self.cl = QSlider(Qt.Orientation.Horizontal, minimum=1, maximum=30, value=7)     # kernel = 2*v+1
        self.cl.valueChanged.connect(self.on_slider)
        self.timer_seg = QTimer(self, singleShot=True, interval=150)
        self.timer_seg.timeout.connect(self.reextract)
        self.cb_match = QCheckBox("Show colour mask (magenta)")
        self.cb_match.toggled.connect(self.on_show_match)
        self.cb_static = QCheckBox("Ignore the fixed layer of the bed")
        self.cb_static.setChecked(False)     # only for a bed with a fixed layer of the fabric's colour
        self.cb_static.toggled.connect(self.reextract)
        self.b_static = btn("Learn fixed layer from folder...", self.learn_static)
        self.static_text = QLabel()
        self.static_text.setWordWrap(True)

        # target / machine
        self.mode = QComboBox()
        self.mode.addItems(["L1: basis", "L2: basis + offset"])
        self.mode.currentIndexChanged.connect(self.refresh_machine)
        self.cb_scale = QCheckBox("Scale constants to image width")
        self.cb_scale.setChecked(True)
        self.cb_scale.toggled.connect(self.refresh_machine)
        self.rc = QComboBox()
        self.rc.addItems(["Rotate about P (hand)", "Rotate about B (corner)"])
        self.rc.currentIndexChanged.connect(self.run_plan)
        self.ra = QComboBox()
        self.ra.addItems(["Angle: both edges", "Angle: horizontal edge", "Angle: vertical edge"])
        self.ra.currentIndexChanged.connect(self.run_plan)
        self.machine_text = QLabel()
        self.machine_text.setWordWrap(True)

        self.b_reset = btn("Reset", self.canvas.reset_anim)
        self.b_trans = btn("1. Translate", lambda: self.step(["trans"], reset=True))
        self.b_rot = btn("2. Rotate", self.do_rotate)
        self.b_all = btn("Play all", lambda: self.step(["trans", "rot"], reset=True))
        self.speed = QComboBox()
        self.speed.addItems(["0.5x", "1x", "2x"])
        self.speed.setCurrentIndex(1)
        self.speed.currentIndexChanged.connect(
            lambda i: setattr(self.canvas, "speed", (0.5, 1.0, 2.0)[i]))

        self.result = QLabel()
        self.result.setTextFormat(Qt.TextFormat.RichText)
        self.result.setWordWrap(True)
        self.result.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.status = QLabel("Open an image or grab a camera frame to start.")
        self.status.setWordWrap(True)

        def section(title):
            l = QLabel(title)
            l.setStyleSheet("font-size:17pt; margin-top:10px;")
            return l

        side = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(self.b_open)
        row.addWidget(self.b_cam)
        side.addLayout(row)
        side.addWidget(self.path_label)
        side.addWidget(section("Fabric colour"))
        row = QHBoxLayout()
        row.addWidget(self.swatch)
        row.addWidget(self.color_text, 1)
        side.addLayout(row)
        side.addWidget(self.b_color)
        side.addWidget(self.b_auto)
        side.addWidget(self.tol_label)
        side.addWidget(self.tol)
        side.addWidget(self.cb_match)
        side.addWidget(section("Fixed layer"))
        side.addWidget(self.cb_static)
        side.addWidget(self.b_static)
        side.addWidget(self.static_text)
        side.addWidget(section("Segmentation"))
        side.addWidget(self.op_label)
        side.addWidget(self.op)
        side.addWidget(self.cl_label)
        side.addWidget(self.cl)
        side.addWidget(section("Target"))
        side.addWidget(self.mode)
        side.addWidget(self.cb_scale)
        side.addWidget(self.rc)
        side.addWidget(self.ra)
        side.addWidget(self.machine_text)
        side.addWidget(section("Animation"))
        row = QHBoxLayout()
        row.addWidget(self.b_trans)
        row.addWidget(self.b_rot)
        side.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self.b_all)
        row.addWidget(self.b_reset)
        row.addWidget(self.speed)
        side.addLayout(row)
        side.addWidget(section("Transform"))
        side.addWidget(self.result, 1)
        side.addWidget(self.status)
        panel = QWidget()
        panel.setLayout(side)
        scroll = QScrollArea()                       # the panel is long: scroll instead of clipping
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(372)

        lay = QHBoxLayout(self)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(scroll)
        self.on_slider()
        self.update_swatch()
        self.update_static_text()
        self.update_buttons()

    # ---- loading
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif)")
        if path:
            self.load_image(path)

    def load_image(self, path):
        data = np.fromfile(path, np.uint8)               # imdecode: works with non-ASCII paths on Windows
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            self.status.setText(f"Cannot read {path}")
            return
        full = os.path.abspath(path)
        self.path_label.setText(full.replace("\\", "\\​").replace("/", "/​"))   # wrap at separators
        self.path_label.setToolTip(full)
        self.setWindowTitle(f"Fabric corner matcher (colour) - {full}")
        print("[image]", full)
        self.load_bgr(img)

    def grab_camera(self):
        cap = None
        img = None
        try:
            for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
                cap = cv2.VideoCapture(CAM_INDEX, backend)
                if cap.isOpened():
                    break
                cap.release()
                cap = None
            if cap is None:
                self.status.setText(f"Cannot open camera {CAM_INDEX}.")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            for _ in range(8):                           # let the exposure settle, keep the last frame
                ok, f = cap.read()
                if ok:
                    img = f
        finally:
            if cap is not None:
                cap.release()
        if img is None:
            self.status.setText("The camera returned no frame.")
            return
        stamp = time.strftime("%H:%M:%S")
        self.path_label.setText(f"Camera {CAM_INDEX}, frame grabbed at {stamp}")
        self.path_label.setToolTip("")
        self.setWindowTitle(f"Fabric corner matcher (colour) - camera {CAM_INDEX} {stamp}")
        self.load_bgr(img)

    def load_bgr(self, img):
        self.bgr, self.contour, self.mask = img, None, None
        self.plan = None
        self.canvas.set_image(img)
        self.refresh_machine()
        self.show_result()
        self.extract_msg = ""
        self.reextract()

    # ---- segmentation
    def on_slider(self):
        self.tol_label.setText(f"Colour tolerance: {self.tol.value()}"
                               + ("" if self.fabric_bgr is not None else " (unused in auto mode)"))
        self.op_label.setText(f"Leak cut (opening): {2 * self.op.value() + 1} px")
        self.cl_label.setText(f"Seam bridge (closing): {2 * self.cl.value() + 1} px")
        if self.bgr is not None:
            self.timer_seg.start()

    def reextract(self):
        if self.bgr is None:
            return
        h, w = self.bgr.shape[:2]
        c, m, raw = segment_fabric(self.bgr, self.fabric_bgr, self.tol.value(),
                                   2 * self.op.value() + 1, 2 * self.cl.value() + 1,
                                   static=self.current_static(w, h), static_grow=STATIC_GROW)
        self.match = raw
        self.canvas.set_match(raw)
        self.contour, self.mask = c, m
        self.canvas.set_fabric(self.bgr, c, m)
        if c is None:
            self.status.setText("No fabric in view" + (" (no clear warm region)." if self.fabric_bgr is None
                                                          else ". Pick the colour again or raise the tolerance."))
            self.plan = None
            self.show_result()
            return
        how = "picked colour" if self.fabric_bgr is not None else "auto"
        msg = f"Fabric contour: {len(c)} points ({how}), {float(np.mean(m > 0)):.0%} of the image."
        cover = float(np.mean(raw > 0))
        if cover > 0.6:
            msg += f" WARNING: the colour mask covers {cover:.0%} of the image: lower the tolerance."
        self.extract_msg = msg                      # kept: the corner step appends to it, never replaces it
        self.status.setText(msg)
        self.save_settings()
        self.run_plan()

    def current_static(self, w, h):
        """The fixed-layer mask for an image of w x h, or None (not learnt, or switched off)."""
        return fit_static(self.static, w, h) if self.cb_static.isChecked() else None

    def update_static_text(self):
        if self.static is None:
            self.static_text.setText("Not learnt. If a layer of the fabric's colour is fixed to the bed, "
                                     "learn it from a folder of frames.")
        else:
            self.static_text.setText(f"Learnt: {self.static.mean():.0%} of the frame.")

    def learn_static(self):
        folder = QFileDialog.getExistingDirectory(self, "Folder of frames (fabric in different places)")
        if not folder:
            return
        files = []
        for ext in ("png", "jpg", "jpeg", "bmp"):
            files += glob.glob(os.path.join(folder, f"*.{ext}"))
        frames = [cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR) for f in sorted(files)]
        frames = [f for f in frames if f is not None]
        if len(frames) < 5:
            self.status.setText(f"Only {len(frames)} frame(s) in that folder: need at least 5.")
            return
        self.static = build_static_mask(frames, self.fabric_bgr)
        save_static(self.static)
        self.cb_static.setChecked(True)
        self.update_static_text()
        self.status.setText(f"Fixed layer learnt from {len(frames)} frames and saved.")
        self.reextract()

    def on_show_match(self, on):
        self.canvas.show_match = on
        self.canvas.update()

    def set_pick(self, on):
        self.canvas.pick_mode = "color" if on else None
        self.canvas.setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)

    def on_color(self, x, y):
        self.b_color.setChecked(False)
        self.fabric_bgr = sample_color(self.bgr, x, y)
        self.update_swatch()
        self.on_slider()
        self.reextract()

    def clear_color(self):
        self.fabric_bgr = None
        self.update_swatch()
        self.on_slider()
        self.reextract()

    def update_swatch(self):
        if self.fabric_bgr is None:
            self.swatch.setStyleSheet("background:#d8d8d4; border:1px dashed #888;")
            self.color_text.setText("Auto: the fabric is the warmer class of the frame, so it does not "
                                    "depend on the lamp. To force a colour, click the button above, then the fabric.")
            return
        b, g, r = (int(round(v)) for v in self.fabric_bgr)
        L, a, bb = bgr_to_lab(self.fabric_bgr)
        self.swatch.setStyleSheet(f"background:rgb({r},{g},{b}); border:1px solid #444;")
        self.color_text.setText(f"RGB ({r}, {g}, {b})<br>Lab ({L:.0f}, {a:.0f}, {bb:.0f})")

    def load_settings(self):
        try:
            with open(SETTINGS, encoding="utf-8") as f:
                s = json.load(f)
            self.tol_default = int(s.get("tol", self.tol_default))
        except (OSError, ValueError, TypeError):
            pass

    def save_settings(self):
        try:
            with open(SETTINGS, "w", encoding="utf-8") as f:
                json.dump({"fabric_bgr": None if self.fabric_bgr is None else [float(v) for v in self.fabric_bgr],
                           "tol": self.tol.value()}, f)
        except OSError:
            pass

    # ---- machine target + plan
    def refresh_machine(self, *_):
        if self.bgr is None:
            return
        h, w = self.bgr.shape[:2]
        m = load_machine((w, h), "L2" if self.mode.currentIndex() == 1 else "L1", self.cb_scale.isChecked())
        self.machine = m
        self.canvas.set_machine(m)
        warn = "".join(f"<br><span style='color:#b3261e'>{t}</span>" for t in m.warnings)
        num = f"font-family:'{MONO}'"
        self.machine_text.setText(
            f"<span style='font-size:11pt'>B = <span style=\"{num}\">({m.basis[0]:.1f}, {m.basis[1]:.1f})</span> px, "
            f"P = <span style=\"{num}\">({m.pivot[0]:.1f}, {m.pivot[1]:.1f})</span> px<br>"
            f"scale <span style=\"{num}\">{m.px_per_cm[0]:.1f}</span> x "
            f"<span style=\"{num}\">{m.px_per_cm[1]:.1f}</span> px/cm"
            f"{warn}</span>")
        self.run_plan()

    def run_plan(self, *_):
        if self.contour is None or self.machine is None or self.bgr is None:
            return
        m = self.machine
        h, w = self.bgr.shape[:2]
        static = self.current_static(w, h)
        corner, why = find_bottom_left_corner(self.contour, (w, h), m.u1, m.u2,
                                              ("auto", "horizontal", "vertical")[self.ra.currentIndex()],
                                              min_cut_edge_frac=0.04, long_h_min_theta=35.0,
                                              occluder=None if static is None else occluder_from_static(static))
        if corner is None:
            self.plan = None
            self.show_result()
            self.status.setText((self.extract_msg + " " if self.extract_msg else "") + why)
            return
        about_p = self.rc.currentIndex() == 0
        self.plan = plan_translate_then_rotate(corner, m.basis, m.pivot if about_p else m.basis,
                                               "P" if about_p else "B", m.u1, m.u2, m.px_per_cm, m.max_rot_deg)
        self.show_result()
        self.status.setText((self.extract_msg + " " if self.extract_msg else "") +
                            "Corner found. Use 1. Translate / 2. Rotate, or Play all.")
        self.step(["trans", "rot"], reset=True)

    def show_result(self):
        r = self.plan
        self.canvas.set_result(r)
        if r is None:
            self.result.setText("")
        else:
            num = f"font-family:'{MONO}'"
            cn = r.corner
            warn = "".join(f"<p style='color:#b3261e'>{t}</p>" for t in r.warnings)
            self.result.setText(
                f"<table cellspacing=4 style='font-size:13pt'>"
                f"<tr><td>tx</td><td style=\"{num}\">{r.tx_mm:+.2f} mm</td></tr>"
                f"<tr><td>ty</td><td style=\"{num}\">{r.ty_mm:+.2f} mm</td></tr>"
                f"<tr><td>theta</td><td style=\"{num}\">{r.theta:+.2f} deg</td></tr></table>"
                f"<p style='font-size:11pt'>1. Translate by (tx, ty) along U1 (right) and U2 (up), "
                f"then 2. rotate by theta about {r.rot_center_name} = "
                f"<span style=\"{num}\">({r.rot_center[0]:.0f}, {r.rot_center[1]:.0f})</span> "
                f"(positive = clockwise on screen).</p>"
                f"<table cellspacing=3 style='font-size:11pt'>"
                f"<tr><td>corner</td><td style=\"{num}\">({cn.point[0]:.1f}, {cn.point[1]:.1f}) px</td></tr>"
                f"<tr><td>target B</td><td style=\"{num}\">({r.basis[0]:.1f}, {r.basis[1]:.1f}) px</td></tr>"
                f"<tr><td>corner angle</td><td style=\"{num}\">{cn.angle_deg:.1f} deg</td></tr>"
                f"<tr><td>edges</td><td style=\"{num}\">{cn.h_len:.0f} / {cn.v_len:.0f} px</td></tr>"
                f"<tr><td>line fit</td><td style=\"{num}\">{cn.h_rmse:.1f} / {cn.v_rmse:.1f} px</td></tr></table>"
                f"{warn}")
        self.update_buttons()

    # ---- animation controls
    def step(self, steps, reset=False):
        if reset:
            self.canvas.reset_anim()
        self.canvas.play(steps)

    def do_rotate(self):
        cv = self.canvas
        cv.timer.stop()
        cv.trans_f, cv.rot_f = 1.0, 0.0
        cv.play(["rot"])

    def update_buttons(self):
        has = self.plan is not None
        for b in (self.b_rot, self.b_trans, self.b_all, self.b_reset):
            b.setEnabled(has)
        self.canvas.update()


def main():
    app = QApplication(sys.argv)
    load_computer_modern()
    app.setFont(QFont(SERIF, 13))
    w = Window()
    w.show()
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        w.load_image(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
