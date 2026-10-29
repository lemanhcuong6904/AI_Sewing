"""Colour-only fabric segmentation (no DexiNed).

For fabric that differs clearly in colour / brightness from the background (beige piece on black
felt), the colour mask is a far more stable fabric region than "areas without edges": felt seams and
folds produce strong edges, but they do not change the colour.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from cuonglm.curve_matcher.edge_detect import bgr_to_lab, bgr_to_lab_image, color_match_mask  # noqa: E402


def _ellipse(px):
    px = max(1, int(px)) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))


def _otsu_1d(vals, lo=-40.0, hi=60.0, bins=200):
    """Otsu threshold of a 1-D sample: (threshold, mean below, mean above, fraction above)."""
    hist, edges = np.histogram(vals, bins=bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    s0 = np.cumsum(hist * c)
    m0 = s0 / np.maximum(w0, 1)
    m1 = (s0[-1] - s0) / np.maximum(w1, 1)
    k = int(np.argmax(w0 * w1 * (m0 - m1) ** 2))
    return float(c[k]), float(m0[k]), float(m1[k]), float(w1[k] / max(w0[-1], 1))


def auto_fabric_mask(bgr, min_sep=7.0, max_frac=0.8):
    """uint8 0/1 mask of the fabric without a given colour: robust to the lamp being on or off.

    The fabric (brown card) is *warmer* than the dark felt around it: its Lab b* is higher.  A change of
    illumination (brightness, white balance) moves the fabric and the felt the same way, so the
    difference stays and the threshold is set per frame (Otsu on b*).  The white and the red parts of
    the machine (very bright / very red) and black are left out of the statistics.  When the two classes
    are not clearly apart, or one of them takes (almost) everything, there is no fabric in view and the
    mask is empty.
    """
    lab = bgr_to_lab_image(bgr, 2.0)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    valid = (L < 80) & (L > 8) & (a < 25)
    if int(valid.sum()) < 0.05 * valid.size:
        return np.zeros(valid.shape, np.uint8)
    thr, m_lo, m_hi, frac = _otsu_1d(b[valid])
    if m_hi - m_lo < min_sep or frac > max_frac:
        return np.zeros(valid.shape, np.uint8)
    return (valid & (b > thr)).astype(np.uint8)


def build_static_mask(frames, fabric_bgr=None, tol=14, frac=0.8, open_px=5):
    """uint8 0/1 mask of the fabric-coloured layer that is present in (almost) every frame.

    The camera is fixed, so a beige layer that stays in the same place in >= `frac` of the frames is
    part of the machine bed, not the fabric being moved.  It is the same colour as the fabric, so the
    colour mask alone merges the two.  `tol` is wider than the one used for the fabric: the layer
    must be found out to the edge of the frame, where the colour drifts a little.
    """
    acc, n = None, 0
    ref = None if fabric_bgr is None else bgr_to_lab(fabric_bgr)
    for f in frames:
        m = (auto_fabric_mask(f) if ref is None else color_match_mask(bgr_to_lab_image(f), ref, tol)).astype(np.float32)
        acc = m if acc is None else acc + m
        n += 1
    static = (acc / max(n, 1) >= frac).astype(np.uint8)
    static = cv2.morphologyEx(static, cv2.MORPH_OPEN, _ellipse(open_px))
    n_cc, lab, stats, _ = cv2.connectedComponentsWithStats(static, connectivity=8)
    if n_cc > 2:                                            # keep the layer itself, drop stray matches
        static = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return _smooth_top_layer(static)


def _smooth_top_layer(static, k=121):
    """A layer hanging from the top image border has a straight lower edge, but where the fabric often
    lies over it the frequency map makes that edge ragged.  Replace the lower edge by the median of its
    profile (window k px).  Columns the layer does not reach from the top are left as they are."""
    h, w = static.shape
    top = static[0] > 0
    run = np.argmax(static == 0, axis=0)                    # first empty row below the top
    run[static.all(axis=0)] = h
    bottom = np.where(top, run, 0).astype(np.float32)
    pad = k // 2
    padded = np.pad(bottom, pad, mode="edge")
    smooth = np.array([np.median(padded[i:i + k]) for i in range(w)])
    out = static.copy()
    rows = np.arange(h)[:, None]
    out[(rows < smooth[None, :]) & top[None, :]] = 1
    out[(rows >= smooth[None, :]) & top[None, :]] = 0
    return out


def occluder_from_static(static, grow=16):
    """Pixels near the static layer: a fabric edge ending here is hidden under / merged with it."""
    return cv2.dilate(static, _ellipse(2 * grow + 1))


def segment_fabric(bgr, fabric_bgr=None, tol=25, open_px=7, close_px=15, min_frac=0.01, static=None, static_grow=7):
    """Return (contour (N,2) float32 | None, mask uint8 0/255, raw colour mask uint8 0/1).

    fabric_bgr: picked fabric colour.  Without it (default) the fabric is found as the warmer class of
    the frame (auto_fabric_mask), which does not depend on the lighting.  `static` (see build_static_mask) is cut out of the mask first.  The colour mask is opened (cuts thin leaks / threads), closed
    (bridges texture and thin seams) and the largest blob is kept, with its holes filled.
    """
    h, w = bgr.shape[:2]
    lab = bgr_to_lab_image(bgr)
    if fabric_bgr is None:
        raw = auto_fabric_mask(bgr)
    else:
        raw = color_match_mask(lab, bgr_to_lab(fabric_bgr), tol)
    m = raw
    if static is not None:                          # the moving fabric is what is NOT the fixed layer
        m = m & (1 - cv2.dilate(static, _ellipse(2 * static_grow + 1)))
    if open_px > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _ellipse(open_px))
    if close_px > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _ellipse(close_px))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   # external: holes are filled
    if not cnts:
        return None, np.zeros((h, w), np.uint8), raw
    big = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(big) < min_frac * h * w:
        return None, np.zeros((h, w), np.uint8), raw
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [big], -1, 255, cv2.FILLED)
    return big.reshape(-1, 2).astype(np.float32), mask, raw


STATIC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static_mask.png")


def save_static(static, path=STATIC_PATH):
    cv2.imencode(".png", static * 255)[1].tofile(path)


def load_static(path=STATIC_PATH):
    """The saved fixed-layer mask (uint8 0/1) or None."""
    if not os.path.isfile(path):
        return None
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    return None if img is None else (img > 127).astype(np.uint8)


def fit_static(static, w, h):
    """The fixed-layer mask at the size of the current image (nearest neighbour), or None."""
    if static is None:
        return None
    if static.shape != (h, w):
        static = cv2.resize(static, (w, h), interpolation=cv2.INTER_NEAREST)
    return static
