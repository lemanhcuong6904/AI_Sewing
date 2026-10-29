"""Frame browser for the fabric corner matcher (PyQt6), same look as cuonglm/curve_matcher.

The image you open is processed on its own (Harris corner + fabric colour, see corner_first.py).
Open an image (button or command line); the translate-then-rotate animation of that image to the
target (B, U1, U2 from utils/constants.py) is shown; Pause / Reset / Play control it.

Run:  python test_frames/app.py [image]
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
                             QPushButton, QScrollArea, QSlider, QVBoxLayout, QWidget)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)
import corner_first as CF  # noqa: E402
from cuonglm.curve_matcher import app as CA  # noqa: E402  (Canvas, fonts, helpers)
from cuonglm.curve_matcher.config import load_machine  # noqa: E402
from cuonglm.curve_matcher.corner import plan_translate_then_rotate  # noqa: E402

DEFAULT_DIR = os.path.abspath(os.path.join(HERE, "..", "adjustor_data", "square-has_light", "data"))
SETTINGS = os.path.join(HERE, "settings.json")
ROTATIONS = [("Rotate 90 deg CCW", cv2.ROTATE_90_COUNTERCLOCKWISE), ("Rotate 90 deg CW", cv2.ROTATE_90_CLOCKWISE),
             ("No rotation", None)]


def read_image(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


class Window(QWidget):
    def __init__(self, image=None):
        super().__init__()
        self.setWindowTitle("Fabric corner matcher - frames")
        self.resize(1400, 920)
        self.bgr = self.lab = None
        self.det = self.plan = self.machine = None
        self.ref_lab = np.array([37.4, 0.3, 1.5], np.float32)       # dark neutral fabric (measured)
        self.tol_default, self.lw_default = 25, 100          # white fabric on a dark mat
        self.load_settings()

        self.canvas = CA.Canvas()
        self.canvas.color_clicked.connect(self.on_color)
        self.canvas.anim_changed.connect(self.update_buttons)

        def btn(text, slot):
            b = QPushButton(text)
            b.clicked.connect(slot)
            return b

        self.b_open = btn("Open image...", self.open_image)
        self.frame_label = QLabel("No image")
        self.path = None
        self.rot = QComboBox()
        self.rot.addItems([r[0] for r in ROTATIONS])
        self.rot.setCurrentIndex(2)                       # the machine frames are already landscape
        self.rot.currentIndexChanged.connect(lambda *_: self.load_frame())

        self.swatch = QLabel()
        self.swatch.setFixedSize(46, 30)
        self.color_text = QLabel()
        self.cb_auto_col = QCheckBox("Auto colour (brightest large blob)")
        self.cb_auto_col.setChecked(True)
        self.cb_auto_col.toggled.connect(self.recompute)
        self.b_color = QPushButton("Pick fabric colour (click)")
        self.b_color.setCheckable(True)
        self.b_color.toggled.connect(self.on_pick_mode)
        self.tol_label, self.lw_label = QLabel(), QLabel()
        self.tol = QSlider(Qt.Orientation.Horizontal, minimum=3, maximum=40, value=self.tol_default)
        self.lw = QSlider(Qt.Orientation.Horizontal, minimum=0, maximum=100, value=self.lw_default)
        self.tol.valueChanged.connect(self.recompute)
        self.lw.valueChanged.connect(self.recompute)
        self.cb_match = QCheckBox("Show colour match (magenta)")
        self.cb_match.toggled.connect(self.on_show_match)

        self.mode = QComboBox()
        self.mode.addItems(["L1: basis", "L2: basis + offset"])
        self.mode.currentIndexChanged.connect(self.refresh_machine)
        self.rc = QComboBox()
        self.rc.addItems(["Rotate about P (hand)", "Rotate about B (corner)"])
        self.rc.currentIndexChanged.connect(self.run_plan)
        self.machine_text = QLabel()
        self.machine_text.setWordWrap(True)

        self.b_trans = btn("1. Translate", lambda: self.step(["trans"], reset=True))
        self.b_rot = btn("2. Rotate", self.do_rotate)
        self.b_all = btn("Play all", lambda: self.step(["trans", "rot"], reset=True))
        self.b_pause = btn("Pause", self.toggle_pause)
        self.b_reset = btn("Reset", self.canvas.reset_anim)
        self.speed = QComboBox()
        self.speed.addItems(["0.5x", "1x", "2x"])
        self.speed.setCurrentIndex(1)
        self.speed.currentIndexChanged.connect(lambda i: setattr(self.canvas, "speed", (0.5, 1.0, 2.0)[i]))
        self.cb_auto = QCheckBox("Auto-play when the frame changes")
        self.cb_auto.setChecked(True)

        self.result = QLabel()
        self.result.setTextFormat(Qt.TextFormat.RichText)
        self.result.setWordWrap(True)
        self.result.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.status = QLabel("Open an image to start.")
        self.status.setWordWrap(True)

        def section(title):
            lab = QLabel(title)
            lab.setStyleSheet("font-size:17pt; margin-top:10px;")
            return lab

        side = QVBoxLayout()
        side.addWidget(self.b_open)
        side.addWidget(self.frame_label)
        side.addWidget(self.rot)
        side.addWidget(section("Fabric colour"))
        row = QHBoxLayout()
        row.addWidget(self.swatch)
        row.addWidget(self.color_text, 1)
        side.addLayout(row)
        side.addWidget(self.cb_auto_col)
        side.addWidget(self.b_color)
        side.addWidget(self.tol_label)
        side.addWidget(self.tol)
        side.addWidget(self.lw_label)
        side.addWidget(self.lw)
        side.addWidget(self.cb_match)
        side.addWidget(section("Target"))
        side.addWidget(self.mode)
        side.addWidget(self.rc)
        side.addWidget(self.machine_text)
        side.addWidget(section("Animation"))
        row = QHBoxLayout()
        row.addWidget(self.b_trans)
        row.addWidget(self.b_rot)
        side.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self.b_all)
        row.addWidget(self.b_pause)
        row.addWidget(self.b_reset)
        side.addLayout(row)
        side.addWidget(self.speed)
        side.addWidget(self.cb_auto)
        side.addWidget(section("Transform"))
        side.addWidget(self.result, 1)
        side.addWidget(self.status)
        panel = QWidget()
        panel.setLayout(side)
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(372)
        lay = QHBoxLayout(self)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(scroll)
        self.update_labels()
        self.update_swatch()
        self.update_buttons()
        if image and os.path.isfile(image):
            self.load_image(image)

    # ---- settings
    def load_settings(self):
        try:
            with open(SETTINGS, encoding="utf-8") as f:
                s = json.load(f)
            if s.get("ref_lab"):
                self.ref_lab = np.array(s["ref_lab"], np.float32)
            self.tol_default = int(s.get("tol", self.tol_default))
            self.lw_default = int(s.get("lw", self.lw_default))
        except (OSError, ValueError, TypeError):
            pass

    def save_settings(self):
        try:
            with open(SETTINGS, "w", encoding="utf-8") as f:
                json.dump({"ref_lab": [float(v) for v in self.ref_lab], "tol": self.tol.value(), "lw": self.lw.value()}, f)
        except OSError:
            pass

    # ---- frames
    def open_image(self):
        start = os.path.dirname(self.path) if self.path else DEFAULT_DIR
        path, _ = QFileDialog.getOpenFileName(self, "Open image", start, "Images (*.png *.jpg *.jpeg *.bmp *.tif)")
        if path:
            self.load_image(path)

    def load_image(self, path):
        self.path = path
        self.load_frame()

    def load_frame(self):
        if not self.path:
            return
        img = read_image(self.path)
        if img is None:
            self.status.setText("Cannot read " + self.path)
            return
        code = ROTATIONS[self.rot.currentIndex()][1]
        if code is not None:
            img = cv2.rotate(img, code)
        self.bgr = img
        self.lab = CF.bgr_to_lab_image(img, blur=max(2.0, min(img.shape[:2]) / 330.0))
        self.canvas.set_image(img)
        self.frame_label.setText(os.path.basename(self.path))
        self.refresh_machine(play=False)
        self.recompute()

    # ---- detection (per frame, independent)
    def recompute(self, *_):
        self.update_labels()
        if self.bgr is None:
            return
        if self.cb_auto_col.isChecked():
            self.ref_lab = self.auto_colour()
            self.update_swatch()
        self.det = CF.detect(self.bgr, self.ref_lab, self.tol.value(), self.machine.u1, self.machine.u2,
                             lab=self.lab, l_weight=self.lw.value() / 100.0)
        d = self.det
        self.canvas.set_match(d.mask if False else (CF.color_distance(self.lab, self.ref_lab, self.lw.value() / 100.0)
                                                    <= self.tol.value()).astype(np.uint8))
        cnts, _ = cv2.findContours(d.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if d.corner is not None and cnts:
            c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
            self.canvas.set_fabric(self.bgr, c, d.mask)
        else:
            self.canvas.set_fabric(self.bgr, None, None)
        self.save_settings()
        self.run_plan()

    def on_show_match(self, on):
        self.canvas.show_match = on
        self.canvas.update()

    # ---- colour
    def on_pick_mode(self, on):
        self.canvas.pick_mode = "color" if on else None
        self.canvas.setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)

    def auto_colour(self):
        g = cv2.GaussianBlur(cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY), (0, 0), 20)
        y, x = np.unravel_index(int(np.argmax(g)), g.shape)
        r = max(3, min(self.bgr.shape[:2]) // 60)
        patch = self.lab[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].reshape(-1, 3)
        return np.median(patch, 0).astype(np.float32)

    def on_color(self, x, y):
        self.b_color.setChecked(False)
        self.cb_auto_col.blockSignals(True)
        self.cb_auto_col.setChecked(False)
        self.cb_auto_col.blockSignals(False)
        r = max(3, min(self.bgr.shape[:2]) // 100)
        h, w = self.lab.shape[:2]
        x, y = int(round(x)), int(round(y))
        patch = self.lab[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].reshape(-1, 3)
        self.ref_lab = np.median(patch, 0).astype(np.float32)
        self.update_swatch()
        self.recompute()

    def update_swatch(self):
        px = np.array(self.ref_lab, np.float32).reshape(1, 1, 3)
        b, g, r = (int(round(v * 255)) for v in cv2.cvtColor(px, cv2.COLOR_Lab2BGR).reshape(3))
        self.swatch.setStyleSheet(f"background:rgb({r},{g},{b}); border:1px solid #444;")
        L, a, bb = self.ref_lab
        self.color_text.setText(f"RGB ({r}, {g}, {b})<br>Lab ({L:.0f}, {a:.0f}, {bb:.0f})")

    def update_labels(self):
        self.tol_label.setText(f"Colour tolerance: {self.tol.value()}")
        self.lw_label.setText(f"Lightness weight: {self.lw.value() / 100:.2f}")

    # ---- machine target + plan
    def refresh_machine(self, *_, play=True):
        if self.bgr is None:
            return
        h, w = self.bgr.shape[:2]
        m = load_machine((w, h), "L2" if self.mode.currentIndex() == 1 else "L1", True)
        self.machine = m
        self.canvas.set_machine(m)
        warn = "".join(f"<br><span style='color:#b3261e'>{t}</span>" for t in m.warnings)
        num = f"font-family:'{CA.MONO}'"
        self.machine_text.setText(
            f"<span style='font-size:11pt'>B = <span style=\"{num}\">({m.basis[0]:.1f}, {m.basis[1]:.1f})</span> px, "
            f"P = <span style=\"{num}\">({m.pivot[0]:.1f}, {m.pivot[1]:.1f})</span> px<br>"
            f"scale <span style=\"{num}\">{m.px_per_cm[0]:.1f}</span> x <span style=\"{num}\">{m.px_per_cm[1]:.1f}</span>"
            f" px/cm{warn}</span>")
        if play:
            self.run_plan()

    def run_plan(self, *_):
        if self.det is None or self.machine is None:
            return
        m = self.machine
        cn = self.det.corner
        if cn is None:
            self.plan = None
            self.show_result()
            self.status.setText(self.det.why)
            return
        about_p = self.rc.currentIndex() == 0
        self.plan = plan_translate_then_rotate(cn, m.basis, m.pivot if about_p else m.basis,
                                               "P" if about_p else "B", m.u1, m.u2, m.px_per_cm, m.max_rot_deg)
        self.show_result()
        self.status.setText("Corner found. Use Play all / Pause / Reset.")
        if self.cb_auto.isChecked():
            self.step(["trans", "rot"], reset=True)

    def show_result(self):
        r = self.plan
        self.canvas.set_result(r)
        if r is None:
            self.result.setText("")
        else:
            num = f"font-family:'{CA.MONO}'"
            cn = r.corner
            warn = "".join(f"<p style='color:#b3261e'>{t}</p>" for t in r.warnings)
            self.result.setText(
                f"<table cellspacing=4 style='font-size:13pt'>"
                f"<tr><td>tx</td><td style=\"{num}\">{r.tx_mm:+.2f} mm</td></tr>"
                f"<tr><td>ty</td><td style=\"{num}\">{r.ty_mm:+.2f} mm</td></tr>"
                f"<tr><td>theta</td><td style=\"{num}\">{r.theta:+.2f} deg</td></tr></table>"
                f"<p style='font-size:11pt'>1. Translate by (tx, ty) along U1 (right) and U2 (up), then 2. rotate by "
                f"theta about {r.rot_center_name} = <span style=\"{num}\">({r.rot_center[0]:.0f}, {r.rot_center[1]:.0f})"
                f"</span> (positive = clockwise on screen).</p>"
                f"<table cellspacing=3 style='font-size:11pt'>"
                f"<tr><td>corner</td><td style=\"{num}\">({cn.point[0]:.1f}, {cn.point[1]:.1f}) px</td></tr>"
                f"<tr><td>target B</td><td style=\"{num}\">({r.basis[0]:.1f}, {r.basis[1]:.1f}) px</td></tr>"
                f"<tr><td>corner angle</td><td style=\"{num}\">{cn.angle_deg:.1f} deg</td></tr>"
                f"<tr><td>edges</td><td style=\"{num}\">{cn.h_len:.0f} / {cn.v_len:.0f} px</td></tr>"
                f"<tr><td>line fit</td><td style=\"{num}\">{cn.h_rmse:.1f} / {cn.v_rmse:.1f} px</td></tr></table>{warn}")
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

    def toggle_pause(self):
        cv = self.canvas
        if cv.timer.isActive():
            cv.timer.stop()
            self.b_pause.setText("Resume")
        elif cv.cur is not None:
            # continue the running step from where it stopped
            import time as _t
            f = cv.rot_f if cv.cur == "rot" else cv.trans_f
            dur = (cv.ROT_MS if cv.cur == "rot" else cv.TRANS_MS) / cv.speed / 1000.0
            cv.t0 = _t.perf_counter() - f * dur
            cv.timer.start()
            self.b_pause.setText("Pause")

    def update_buttons(self):
        has = self.plan is not None
        for b in (self.b_rot, self.b_trans, self.b_all, self.b_reset, self.b_pause):
            b.setEnabled(has)
        if not self.canvas.timer.isActive() and self.canvas.cur is None:
            self.b_pause.setText("Pause")
        self.canvas.update()


def main():
    app = QApplication(sys.argv)
    CA.load_computer_modern()
    app.setFont(QFont(CA.SERIF, 13))
    w = Window(sys.argv[1] if len(sys.argv) > 1 else None)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
