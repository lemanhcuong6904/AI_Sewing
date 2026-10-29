"""Machine constants for the target: utils/constants.py + gui_config.json.

All values are in camera pixels of the frame the machine was calibrated on (640x480, the size of
the frames in adjustor_data).  For another image size they can be scaled with the image width.
"""
import importlib.util
import json
import os
from dataclasses import dataclass, field

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REF_SIZE = (640, 480)                  # (w, h) of the calibrated camera frame
MAX_ROT_DEG = 10.0                     # actuator limit used by modules/fabric_adjustor.py


def _load_constants():
    # utils/__init__.py imports camera/helper (heavy, GPU libs), so load constants.py by path
    spec = importlib.util.spec_from_file_location("sewing_constants", os.path.join(ROOT, "utils", "constants.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_gui_config():
    try:
        with open(os.path.join(ROOT, "gui_config.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


@dataclass
class Machine:
    mode: str                          # "L1" (basis) or "L2" (basis + offset)
    basis: np.ndarray                  # B: where the fabric corner must end up (px)
    pivot: np.ndarray                  # P: centre of rotation of the robot hand (px)
    u1: np.ndarray                     # unit vector of the horizontal axis
    u2: np.ndarray                     # unit vector of the vertical axis
    px_per_cm: tuple                   # (x, y)
    axis_len: float                    # length of the drawn axes (px)
    rot_alignment: str                 # 'horizontal' | 'vertical' | 'mean'
    max_rot_deg: float
    scale: float                       # image width / 640 (1.0 when not scaled)
    offset_cm: tuple
    warnings: list = field(default_factory=list)


def load_machine(image_size, mode="L1", scale_to_image=True):
    """image_size = (w, h) of the loaded image."""
    k = _load_constants()
    cfg = _load_gui_config()
    w, h = image_size

    x_basis = float(cfg.get("x_basis", k.DEFAULT_X_BASIS))
    y_basis = float(cfg.get("y_basis", k.DEFAULT_Y_BASIS))
    off = (float(cfg.get("x_offset_cm", 0.0)), float(cfg.get("y_offset_cm", 0.0)))
    ppx, ppy = float(k.PX_PER_CM_X), float(k.PX_PER_CM_Y)
    if mode == "L2":                   # same rule as gui_main.py (trigger L2)
        x_basis += off[0] * ppx
        y_basis += off[1] * ppy

    s = (w / float(REF_SIZE[0])) if scale_to_image else 1.0
    warnings = []
    if abs(w - REF_SIZE[0]) > 1 or abs(h - REF_SIZE[1]) > 1:
        if scale_to_image:
            warnings.append(f"Image is {w}x{h}, the constants are for {REF_SIZE[0]}x{REF_SIZE[1]}: "
                            f"they were scaled by x{s:.3f}.")
        else:
            warnings.append(f"Image is {w}x{h}, the constants are for {REF_SIZE[0]}x{REF_SIZE[1]}: "
                            f"used as they are (pixels), mm values may be wrong.")
        if abs((h / float(w)) - (REF_SIZE[1] / float(REF_SIZE[0]))) > 0.02:
            warnings.append("The aspect ratio differs from 4:3: the image is probably cropped, so the "
                            "constants may not apply.")
    return Machine(
        mode=mode,
        basis=np.array([x_basis, y_basis]) * s,
        pivot=np.array([float(k.DEFAULT_PIVOT_X), float(k.DEFAULT_PIVOT_Y)]) * s,
        u1=np.asarray(k.U1_VECTOR, np.float64),
        u2=np.asarray(k.U2_VECTOR, np.float64),
        px_per_cm=(ppx * s, ppy * s),
        axis_len=float(k.AXIS_LENGTH_PX) * s,
        rot_alignment=str(cfg.get("rot_alignment", "horizontal")),
        max_rot_deg=MAX_ROT_DEG,
        scale=s,
        offset_cm=off,
        warnings=warnings,
    )
