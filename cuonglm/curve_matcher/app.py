"""Fabric corner -> machine target (PyQt6).

1. Open an image with a piece of fabric -> DexiNed edge map -> fabric contour.
2. The target comes from the machine constants (utils/constants.py, gui_config.json): a right angle
   at B with arms along U1 and U2.  It is drawn on the image.
3. The bottom-left (approximately right-angle) corner of the fabric is found from the contour by
   fitting two straight edges.  The move that brings it onto the target is computed and animated:
   first translate by (tx, ty), then rotate by theta about the rotation centre.  tx, ty are in mm.

Run:  python cuonglm/curve_matcher/app.py [image]
"""
import json
import os
import sys
import time

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (QColor, QFont, QFontDatabase, QImage, QPainter, QPen, QPixmap, QPolygonF)
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QHBoxLayout,
                             QLabel, QPushButton, QScrollArea, QSlider, QVBoxLayout, QWidget)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from cuonglm.curve_matcher.config import load_machine  # noqa: E402
from cuonglm.curve_matcher.corner import find_bottom_left_corner, plan_translate_then_rotate  # noqa: E402
from cuonglm.curve_matcher.edge_detect import (bgr_to_lab, bgr_to_lab_image, color_match_mask,  # noqa: E402
                                              extract_fabric, infer_edge_map, sample_color)

# Computer Modern (the TeX font) ships with matplotlib, which this project already depends on.
SERIF, MONO = "cmr10", "cmtt10"
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


class EdgeWorker(QThread):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, bgr):
        super().__init__()
        self.bgr = bgr

    def run(self):
        try:
            self.done.emit(infer_edge_map(self.bgr))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


# ------------------------------------------------------------------ canvas
class Canvas(QWidget):
    seed_clicked = pyqtSignal(float, float)
    color_clicked = pyqtSignal(float, float)
    anim_changed = pyqtSignal()

    ROT_MS, TRANS_MS, HOLD_MS = 1800, 1800, 350

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.image = None            # QImage
        self.edge_pix = None         # QPixmap (edge probability, ARGB)
        self.fabric_pix = None       # QPixmap cut-out of the fabric (ARGB)
        self.contour_poly = None
        self.machine = None          # config.Machine: target B, axes U1/U2, pivot P
        self.res = None              # corner.Plan being shown
        self.show_edges = False
        self.pick_mode = None        # None | "seed" (pick region) | "color" (pick fabric colour)
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
        self.edge_pix = self.fabric_pix = self.contour_poly = self.res = self.match_pix = None
        self.reset_anim()
        self.update()

    def set_machine(self, machine):
        self.machine = machine
        self.update()

    def set_edges(self, prob):
        a = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
        rgba = np.zeros((*a.shape, 4), np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3] = 255, 210, 0, a
        h, w = a.shape
        self.edge_pix = QPixmap.fromImage(QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy())
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

    # ---- mouse: only the two click-to-pick modes
    def mousePressEvent(self, e):
        if self.image is None or e.button() != Qt.MouseButton.LeftButton:
            return
        x, y = self.to_img(e.position())
        if self.pick_mode == "seed":
            self.seed_clicked.emit(x, y)
        elif self.pick_mode == "color":
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

        if self.show_edges and self.edge_pix is not None:
            p.drawPixmap(0, 0, self.edge_pix)
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
        self.setWindowTitle("Fabric corner matcher")
        self.resize(1360, 900)
        self.bgr = None
        self.prob = None
        self.contour = self.mask = None
        self.seed = None
        self.plan = None
        self.machine = None
        self.worker = None
        self.extract_msg = ""
        self.fabric_bgr = None       # fabric colour picked by the user (BGR), kept across images
        self.lab_img = self.match = None
        self.tol_default = 25
        self.load_settings()

        self.canvas = Canvas()
        self.canvas.seed_clicked.connect(self.on_seed)
        self.canvas.color_clicked.connect(self.on_color)
        self.canvas.anim_changed.connect(self.update_buttons)

        def btn(text, slot):
            b = QPushButton(text)
            b.clicked.connect(slot)
            return b

        self.b_open = btn("Open image...", self.open_image)
        self.path_label = QLabel("No image loaded.")
        self.path_label.setWordWrap(True)
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.th_label = QLabel()
        self.th = QSlider(Qt.Orientation.Horizontal, minimum=10, maximum=80, value=25)
        self.th.valueChanged.connect(self.on_threshold)
        self.op_label = QLabel()
        self.op = QSlider(Qt.Orientation.Horizontal, minimum=1, maximum=30, value=10)   # kernel = 2*v+1
        self.op.valueChanged.connect(self.on_threshold)
        self.th_timer = QTimer(self, singleShot=True, interval=150)
        self.th_timer.timeout.connect(self.reextract)
        self.cb_edges = QCheckBox("Show DexiNed edge map")
        self.cb_edges.toggled.connect(self.on_show_edges)
        self.b_seed = QPushButton("Pick fabric region (click)")
        self.b_seed.setCheckable(True)
        self.b_seed.toggled.connect(lambda on: self.set_pick("seed", on))
        self.b_color = QPushButton("Pick fabric colour (click)")
        self.b_color.setCheckable(True)
        self.b_color.toggled.connect(lambda on: self.set_pick("color", on))
        self.swatch = QLabel()
        self.swatch.setFixedSize(46, 30)
        self.color_text = QLabel()
        self.color_text.setWordWrap(True)
        self.tol_label = QLabel()
        self.tol = QSlider(Qt.Orientation.Horizontal, minimum=5, maximum=60, value=self.tol_default)
        self.tol.valueChanged.connect(self.on_threshold)
        self.cb_match = QCheckBox("Show colour match (magenta)")
        self.cb_match.toggled.connect(self.on_show_match)

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
        self.status = QLabel("Open an image to start.")
        self.status.setWordWrap(True)

        def section(title):
            l = QLabel(title)
            l.setStyleSheet("font-size:17pt; margin-top:10px;")
            return l

        side = QVBoxLayout()
        side.addWidget(self.b_open)
        side.addWidget(self.path_label)
        side.addWidget(section("Fabric colour"))
        row = QHBoxLayout()
        row.addWidget(self.swatch)
        row.addWidget(self.color_text, 1)
        side.addLayout(row)
        side.addWidget(self.b_color)
        side.addWidget(self.tol_label)
        side.addWidget(self.tol)
        side.addWidget(self.cb_match)
        side.addWidget(section("Edges"))
        side.addWidget(self.th_label)
        side.addWidget(self.th)
        side.addWidget(self.op_label)
        side.addWidget(self.op)
        side.addWidget(self.cb_edges)
        side.addWidget(self.b_seed)
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
        self.on_threshold()
        self.update_swatch()
        self.update_buttons()

    # ---- loading + edges
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
        self.path_label.setText(full.replace("\\", "\\​").replace("/", "/​"))   # allow wrapping at separators
        self.path_label.setToolTip(full)
        self.setWindowTitle(f"Fabric corner matcher - {full}")
        print("[image]", full)
        self.bgr, self.prob, self.contour, self.mask, self.seed = img, None, None, None, None
        self.lab_img = bgr_to_lab_image(img)
        self.plan = None
        self.canvas.set_image(img)
        self.update_match()
        self.refresh_machine()
        self.show_result()
        self.extract_msg = ""
        self.status.setText("Running DexiNed...")
        self.worker = EdgeWorker(img)
        self.worker.done.connect(self.on_edges)
        self.worker.failed.connect(lambda m: self.status.setText("DexiNed failed: " + m))
        self.worker.start()

    def on_edges(self, prob):
        self.prob = prob
        self.canvas.set_edges(prob)
        self.reextract()

    def on_threshold(self):
        self.th_label.setText(f"Edge threshold: {self.th.value() / 100:.2f}")
        self.op_label.setText(f"Leak cut (opening): {2 * self.op.value() + 1} px")
        self.tol_label.setText(f"Colour tolerance: {self.tol.value()}")
        if self.prob is not None:
            self.th_timer.start()
        elif self.lab_img is not None:
            self.update_match()

    def reextract(self):
        if self.prob is None:
            return
        self.update_match()
        c, m, info = extract_fabric(self.prob, thresh=self.th.value() / 100,
                                    open_px=2 * self.op.value() + 1, seed=self.seed,
                                    color_match=self.match, lab_img=self.lab_img)
        self.contour, self.mask = c, m
        self.canvas.set_fabric(self.bgr, c, m)
        if c is None:
            self.status.setText("No fabric region found. Lower the threshold or pick the region manually.")
            self.plan = None
            self.show_result()
            return
        if info["frac"] is None:
            msg = f"Fabric contour: {len(c)} points. Pick the fabric colour for a reliable choice of region."
        else:
            msg = f"Fabric contour: {len(c)} points, {info['frac']:.0%} of it matches the fabric colour."
            if info["frac"] < 0.6:
                msg += " LOW: adjust the tolerance / colour, or pick the region manually."
            cover = float(np.mean(self.match > 0))
            if cover > 0.5:
                msg += (f" WARNING: the colour mask covers {cover:.0%} of the image (the background "
                        f"matches too): lower the colour tolerance.")
        self.extract_msg = msg                      # kept: the corner step appends to it, never replaces it
        self.status.setText(msg)
        self.save_settings()
        self.run_plan()

    def on_show_edges(self, on):
        self.canvas.show_edges = on
        self.canvas.update()

    def on_show_match(self, on):
        self.canvas.show_match = on
        self.canvas.update()

    def set_pick(self, kind, on):
        """Click-to-pick modes ('seed' = region, 'color' = fabric colour) are mutually exclusive."""
        other = self.b_color if kind == "seed" else self.b_seed
        if on and other.isChecked():
            other.blockSignals(True)
            other.setChecked(False)
            other.blockSignals(False)
        if on:
            self.canvas.pick_mode = kind
        elif self.canvas.pick_mode == kind:
            self.canvas.pick_mode = None
        self.canvas.setCursor(Qt.CursorShape.CrossCursor if self.canvas.pick_mode else Qt.CursorShape.ArrowCursor)

    def on_seed(self, x, y):
        self.b_seed.setChecked(False)
        self.seed = (x, y)
        self.reextract()

    def on_color(self, x, y):
        self.b_color.setChecked(False)
        self.fabric_bgr = sample_color(self.bgr, x, y)
        self.seed = None                       # a new colour supersedes a manual region pick
        self.update_swatch()
        self.update_match()
        self.save_settings()
        if self.prob is not None:
            self.reextract()
        else:
            self.status.setText("Fabric colour set. Waiting for DexiNed...")

    def update_match(self):
        """Recompute the per-pixel 'looks like the fabric' mask for the current image."""
        if self.fabric_bgr is None or self.lab_img is None:
            self.match = None
        else:
            self.match = color_match_mask(self.lab_img, bgr_to_lab(self.fabric_bgr), self.tol.value())
        self.canvas.set_match(self.match)

    def update_swatch(self):
        if self.fabric_bgr is None:
            self.swatch.setStyleSheet("background:#d8d8d4; border:1px dashed #888;")
            self.color_text.setText("Not set. Click the button, then click on the fabric.")
            return
        b, g, r = (int(round(v)) for v in self.fabric_bgr)
        L, a, bb = bgr_to_lab(self.fabric_bgr)
        self.swatch.setStyleSheet(f"background:rgb({r},{g},{b}); border:1px solid #444;")
        self.color_text.setText(f"RGB ({r}, {g}, {b})<br>Lab ({L:.0f}, {a:.0f}, {bb:.0f})")

    def load_settings(self):
        try:
            with open(SETTINGS, encoding="utf-8") as f:
                s = json.load(f)
            if s.get("fabric_bgr"):
                self.fabric_bgr = np.array(s["fabric_bgr"], np.float32)
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
        corner, why = find_bottom_left_corner(self.contour, (w, h), m.u1, m.u2,
                                              ("auto", "horizontal", "vertical")[self.ra.currentIndex()])
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
