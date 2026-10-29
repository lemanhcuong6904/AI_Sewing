"""Fabric edge <-> drawn curve matcher (PyQt6).

1. Open an image with a piece of fabric -> DexiNed edge map -> fabric contour.
2. Draw the target curve with the mouse (hold the left button and drag).
3. The rigid transform (rotation theta, then translation tx, ty) is computed and
   animated step by step: first rotate, then translate.

Run:  python cuonglm/curve_matcher/app.py [image]
"""
import json
import os
import sys
import time

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (QColor, QFont, QFontDatabase, QImage, QPainter, QPainterPath, QPen,
                         QPixmap, QPolygonF)
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel,
                             QPushButton, QSlider, QVBoxLayout, QWidget)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from cuonglm.curve_matcher.edge_detect import (bgr_to_lab, bgr_to_lab_image, color_match_mask,  # noqa: E402
                                              extract_fabric, infer_edge_map, sample_color)
from cuonglm.curve_matcher.matcher import match_curves  # noqa: E402

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
    stroke_finished = pyqtSignal()
    seed_clicked = pyqtSignal(float, float)
    color_clicked = pyqtSignal(float, float)
    anim_changed = pyqtSignal()

    ROT_MS, TRANS_MS, HOLD_MS = 1800, 1800, 350

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setMouseTracking(False)
        self.image = None            # QImage
        self.edge_pix = None         # QPixmap (edge probability, ARGB)
        self.fabric_pix = None       # QPixmap cut-out of the fabric (ARGB)
        self.contour_poly = None
        self.drawn = []              # [(x, y)] image coords
        self.drawn_poly = None
        self.res = None              # MatchResult being shown
        self.seg_poly = None
        self.show_edges = False
        self.pick_mode = None        # None | "seed" (pick region) | "color" (pick fabric colour)
        self.match_pix = None        # QPixmap: pixels matching the fabric colour
        self.show_match = False
        self._stroking = False
        self.rot_f = 0.0
        self.trans_f = 0.0
        self.speed = 1.0
        self.queue, self.cur, self.t0, self.hold_until = [], None, 0.0, 0.0
        self.timer = QTimer(self, interval=16)
        self.timer.timeout.connect(self._tick)

    # ---- data
    def set_image(self, bgr):
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self.edge_pix = self.fabric_pix = self.contour_poly = self.seg_poly = self.res = self.match_pix = None
        self.drawn, self.drawn_poly = [], None
        self.reset_anim()
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
        self.seg_poly = polygon(res.contour[res.seg_idx]) if res is not None else None
        self.reset_anim()
        self.update()

    def clear_drawing(self):
        self.drawn, self.drawn_poly = [], None
        self.set_result(None)

    # ---- animation
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
        if self.cur == "rot" or (self.cur is None and 0 < self.rot_f and self.trans_f == 0):
            return f"Step 1 / 2: rotate {r.theta_deg * self.rot_f:+.1f} deg"
        if self.cur == "trans" or self.trans_f > 0:
            return f"Step 2 / 2: translate ({r.tx * self.trans_f:+.1f}, {r.ty * self.trans_f:+.1f}) px"
        return "Ready"

    # ---- view mapping
    def _view(self):
        iw, ih = self.image.width(), self.image.height()
        s = min(self.width() / iw, self.height() / ih)
        return s, (self.width() - iw * s) / 2, (self.height() - ih * s) / 2

    def to_img(self, pos):
        s, ox, oy = self._view()
        return (pos.x() - ox) / s, (pos.y() - oy) / s

    # ---- mouse
    def mousePressEvent(self, e):
        if self.image is None or e.button() != Qt.MouseButton.LeftButton:
            return
        x, y = self.to_img(e.position())
        if self.pick_mode == "seed":
            self.seed_clicked.emit(x, y)
            return
        if self.pick_mode == "color":
            self.color_clicked.emit(x, y)
            return
        self.timer.stop()
        self.drawn = [(x, y)]
        self.drawn_poly = None
        self.res = self.seg_poly = None
        self.rot_f = self.trans_f = 0.0
        self._stroking = True
        self.update()

    def mouseMoveEvent(self, e):
        if not self._stroking:
            return
        x, y = self.to_img(e.position())
        lx, ly = self.drawn[-1]
        if (x - lx) ** 2 + (y - ly) ** 2 >= 4.0:
            self.drawn.append((x, y))
            self.drawn_poly = polygon(self.drawn)
            self.update()

    def mouseReleaseEvent(self, e):
        if self._stroking:
            self._stroking = False
            if len(self.drawn) >= 2:
                self.drawn_poly = polygon(self.drawn)
                self.stroke_finished.emit()
            self.update()

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

        def pen(color, w, dash=False):
            q = QPen(QColor(color), w)
            q.setCosmetic(True)
            q.setCapStyle(Qt.PenCapStyle.RoundCap)
            q.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            if dash:
                q.setStyle(Qt.PenStyle.DashLine)
            return q

        if self.show_edges and self.edge_pix is not None:
            p.drawPixmap(0, 0, self.edge_pix)
        if self.show_match and self.match_pix is not None:
            p.drawPixmap(0, 0, self.match_pix)

        if self.res is not None and self.fabric_pix is not None:
            # ghost of the original position, then the fabric cut-out that moves
            p.fillRect(QRectF(0, 0, W, H), QColor(0, 0, 0, 150))
            p.save()
            r = self.res
            p.translate(r.pivot[0] + r.tx * smoothstep(self.trans_f), r.pivot[1] + r.ty * smoothstep(self.trans_f))
            p.rotate(r.theta_deg * smoothstep(self.rot_f))
            p.translate(-r.pivot[0], -r.pivot[1])
            p.drawPixmap(0, 0, self.fabric_pix)
            p.setPen(pen("#f2c200", 1.5))
            p.drawPolyline(self.contour_poly)
            p.setPen(pen("#14a44d", 4.5))
            p.drawPolyline(self.seg_poly)
            for color, wd in (("#ffffff", 5), ("#111111", 2)):       # dark cross with a white halo
                p.setPen(pen(color, wd))
                p.drawEllipse(QPointF(r.pivot[0], r.pivot[1]), 4 / s, 4 / s)
                p.drawLine(QPointF(r.pivot[0] - 9 / s, r.pivot[1]), QPointF(r.pivot[0] + 9 / s, r.pivot[1]))
                p.drawLine(QPointF(r.pivot[0], r.pivot[1] - 9 / s), QPointF(r.pivot[0], r.pivot[1] + 9 / s))
            p.restore()
        elif self.contour_poly is not None:
            p.setPen(pen("#f2c200", 2))
            p.drawPolyline(self.contour_poly)

        if self.drawn_poly is not None:
            p.setPen(pen("#e01b1b", 4))
            p.drawPolyline(self.drawn_poly)
        elif len(self.drawn) == 1:
            p.setPen(pen("#e01b1b", 6))
            p.drawPoint(QPointF(*self.drawn[0]))

        # step caption (screen space)
        p.resetTransform()
        txt = self.step_text()
        if txt:
            p.setFont(QFont(SERIF, 16))
            m = p.fontMetrics()
            w = m.horizontalAdvance(txt) + 24
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 220))
            p.drawRoundedRect(QRectF(12, 12, w, m.height() + 12), 6, 6)
            p.setPen(QColor("#222"))
            p.drawText(QPointF(24, 12 + 6 + m.ascent()), txt)


# ------------------------------------------------------------------ window
class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fabric edge matcher")
        self.resize(1320, 880)
        self.bgr = None
        self.prob = None
        self.contour = self.mask = None
        self.seed = None
        self.results = []
        self.cur = 0
        self.worker = None
        self.fabric_bgr = None       # fabric colour picked by the user (BGR), kept across images
        self.lab_img = self.match = None
        self.tol_default = 25
        self.load_settings()

        self.canvas = Canvas()
        self.canvas.stroke_finished.connect(self.run_match)
        self.canvas.seed_clicked.connect(self.on_seed)
        self.canvas.color_clicked.connect(self.on_color)
        self.canvas.anim_changed.connect(self.update_buttons)

        def btn(text, slot):
            b = QPushButton(text)
            b.clicked.connect(slot)
            return b

        self.b_open = btn("Open image...", self.open_image)
        self.th_label = QLabel()
        self.th = QSlider(Qt.Orientation.Horizontal, minimum=10, maximum=80, value=20)
        self.th.valueChanged.connect(self.on_threshold)
        self.op_label = QLabel()
        self.op = QSlider(Qt.Orientation.Horizontal, minimum=1, maximum=30, value=17)   # kernel = 2*v+1
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
        self.b_clear = btn("Clear drawing", self.on_clear)
        self.pivot = QComboBox()
        self.pivot.addItems(["Matched segment centre", "Image centre"])
        self.pivot.currentIndexChanged.connect(self.run_match)

        self.b_prev = btn("Prev", lambda: self.pick(-1))
        self.b_next = btn("Next", lambda: self.pick(+1))
        self.cand = QLabel("Candidate -")
        self.cand.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.b_reset = btn("Reset", self.canvas.reset_anim)
        self.b_rot = btn("1. Rotate", lambda: self.step(["rot"], reset=True))
        self.b_trans = btn("2. Translate", self.do_translate)
        self.b_all = btn("Play all", lambda: self.step(["rot", "trans"], reset=True))
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
        side.addWidget(section("Curve"))
        hint = QLabel("Drag on the image to draw the target curve.")
        hint.setWordWrap(True)
        side.addWidget(hint)
        side.addWidget(self.b_clear)
        side.addWidget(QLabel("Rotation pivot"))
        side.addWidget(self.pivot)
        row = QHBoxLayout()
        row.addWidget(self.b_prev)
        row.addWidget(self.cand, 1)
        row.addWidget(self.b_next)
        side.addLayout(row)
        side.addWidget(section("Animation"))
        row = QHBoxLayout()
        row.addWidget(self.b_rot)
        row.addWidget(self.b_trans)
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
        panel.setFixedWidth(340)

        lay = QHBoxLayout(self)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(panel)
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
        self.bgr, self.prob, self.contour, self.mask, self.seed = img, None, None, None, None
        self.lab_img = bgr_to_lab_image(img)
        self.results, self.cur = [], 0
        self.canvas.set_image(img)
        self.update_match()
        self.show_result()
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
                                    color_match=self.match)
        self.contour, self.mask = c, m
        self.canvas.set_fabric(self.bgr, c, m)
        if c is None:
            self.status.setText("No fabric region found. Lower the threshold or pick the region manually.")
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
        self.status.setText(msg + " Now draw the target curve.")
        self.save_settings()
        self.run_match()

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

    def on_clear(self):
        self.canvas.clear_drawing()
        self.results, self.cur = [], 0
        self.show_result()

    # ---- matching
    def run_match(self, *_):
        cv = self.canvas
        if self.contour is None or len(cv.drawn) < 2:
            return
        try:
            h, w = self.bgr.shape[:2]
            self.results = match_curves(self.contour, np.array(cv.drawn),
                                        pivot="image" if self.pivot.currentIndex() == 1 else "segment",
                                        image_size=(w, h))
        except ValueError as e:
            self.results = []
            self.status.setText(str(e))
            cv.set_result(None)
            self.show_result()
            return
        self.cur = 0
        self.show_result()
        self.status.setText("Matched. Use 1. Rotate / 2. Translate, or Play all.")
        self.step(["rot", "trans"], reset=True)

    def pick(self, d):
        if self.results:
            self.cur = (self.cur + d) % len(self.results)
            self.show_result()
            self.step(["rot", "trans"], reset=True)

    def show_result(self):
        res = self.results[self.cur] if self.results else None
        self.canvas.set_result(res)
        self.cand.setText(f"Candidate {self.cur + 1} / {len(self.results)}" if res else "Candidate -")
        if res is None:
            self.result.setText("")
        else:
            num = f"font-family:'{MONO}'"
            warn = "".join(f"<p style='color:#b3261e'>{w}</p>" for w in res.warnings)
            self.result.setText(
                f"<table cellspacing=4 style='font-size:13pt'>"
                f"<tr><td>theta</td><td style=\"{num}\">{res.theta_deg:+.2f} deg</td></tr>"
                f"<tr><td>tx</td><td style=\"{num}\">{res.tx:+.1f} px</td></tr>"
                f"<tr><td>ty</td><td style=\"{num}\">{res.ty:+.1f} px</td></tr></table>"
                f"<p style='font-size:11pt'>Rotate theta about "
                f"<span style=\"{num}\">({res.pivot[0]:.0f}, {res.pivot[1]:.0f})</span> first "
                f"(positive = clockwise on screen), then translate by (tx, ty).</p>"
                f"<table cellspacing=3 style='font-size:11pt'>"
                f"<tr><td>RMSE</td><td style=\"{num}\">{res.rmse:.2f} px</td></tr>"
                f"<tr><td>max error</td><td style=\"{num}\">{res.max_err:.2f} px</td></tr>"
                f"<tr><td>inliers (4px)</td><td style=\"{num}\">{res.inlier_ratio:.0%}</td></tr>"
                f"<tr><td>edge segment</td><td style=\"{num}\">{res.seg_len_px:.0f} px</td></tr></table>"
                f"{warn}")
        self.update_buttons()

    # ---- animation controls
    def step(self, steps, reset=False):
        if reset:
            self.canvas.reset_anim()
        self.canvas.play(steps)

    def do_translate(self):
        cv = self.canvas
        cv.timer.stop()
        cv.rot_f, cv.trans_f = 1.0, 0.0
        cv.play(["trans"])

    def update_buttons(self):
        has = bool(self.results)
        for b in (self.b_rot, self.b_trans, self.b_all, self.b_reset):
            b.setEnabled(has)
        for b in (self.b_prev, self.b_next):
            b.setEnabled(has and len(self.results) > 1)
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
