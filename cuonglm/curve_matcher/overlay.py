"""Preview-only overlay: the machine target B and its axes U1 / U2 (utils/constants.py).

Used by collect_frames.py and record_video.py. The overlay is drawn on a copy of the frame that is
only shown; the saved image / recorded video stays clean.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from cuonglm.curve_matcher.config import load_machine  # noqa: E402

RED, WHITE = (27, 27, 224), (255, 255, 255)


def _pt(p):
    return int(round(float(p[0]))), int(round(float(p[1])))


class AxesOverlay:
    def __init__(self, size):
        """size = (w, h) of the camera frame."""
        self.m = load_machine(size)

    def draw(self, img):
        m = self.m
        B = m.basis
        for u, name in ((m.u1, "U1"), (m.u2, "U2")):
            tip = B + np.asarray(u) * m.axis_len
            cv2.arrowedLine(img, _pt(B), _pt(tip), RED, 2, cv2.LINE_AA, tipLength=0.08)
            cv2.putText(img, name, (_pt(tip)[0] + 6, _pt(tip)[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, RED, 1, cv2.LINE_AA)
        cv2.circle(img, _pt(B), 6, WHITE, 3, cv2.LINE_AA)
        cv2.circle(img, _pt(B), 6, RED, 1, cv2.LINE_AA)
        cv2.putText(img, f"B ({B[0]:.0f}, {B[1]:.0f})", (_pt(B)[0] + 10, _pt(B)[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA)
        return img
