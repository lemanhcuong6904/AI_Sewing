"""DexiNed edge detection + fabric contour extraction (standalone)."""
import os

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ONNX_PATH = os.path.join(ROOT, "models", "dexined", "dexined.onnx")

_session = None


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        if not os.path.exists(ONNX_PATH):
            raise FileNotFoundError(f"DexiNed ONNX not found: {ONNX_PATH}")
        avail = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        _session = ort.InferenceSession(ONNX_PATH, providers=providers)
    return _session


def infer_edge_map(bgr, max_side=1024):
    """Return a float32 edge probability map (H, W) in [0, 1] at the input resolution."""
    h, w = bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    img = cv2.resize(bgr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else bgr
    ih, iw = img.shape[:2]
    ph, pw = (-ih) % 16, (-iw) % 16          # DexiNed needs sizes divisible by 16
    img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    tensor = np.ascontiguousarray(img.astype(np.float32).transpose(2, 0, 1)[None])
    sess = _get_session()
    out = sess.run(None, {sess.get_inputs()[0].name: tensor})[-1]
    prob = 1.0 / (1.0 + np.exp(-np.squeeze(out)))
    prob = prob[:ih, :iw].astype(np.float32)
    if (ih, iw) != (h, w):
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    return prob


def bgr_to_lab_image(bgr, blur=2.0):
    """Lightly blurred float Lab image (L 0-100, a/b about -127..127) used for colour matching."""
    img = cv2.GaussianBlur(bgr, (0, 0), blur) if blur > 0 else bgr
    return cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def bgr_to_lab(bgr_triplet):
    px = np.array(bgr_triplet, np.float32).reshape(1, 1, 3) / 255.0
    return cv2.cvtColor(px, cv2.COLOR_BGR2Lab).reshape(3)


def sample_color(bgr, x, y, radius=6):
    """Median BGR of a small patch around (x, y): robust to noise / fabric texture."""
    h, w = bgr.shape[:2]
    x, y = int(round(x)), int(round(y))
    patch = bgr[max(0, y - radius):min(h, y + radius + 1), max(0, x - radius):min(w, x + radius + 1)]
    return np.median(patch.reshape(-1, 3), axis=0)


def color_match_mask(lab_img, lab_ref, tol, l_weight=None):
    """uint8 mask of pixels whose colour is within `tol` of the reference.

    Lightness is down-weighted so that a bright highlight or a shadow on the fabric
    still matches; the chroma (a, b) does most of the work.  But a colour with almost no
    chroma (white, grey, black fabric) carries no information in a, b: it differs from the
    background mostly in lightness.  So by default the lightness weight grows from 0.3
    (saturated colour) to 1.0 (neutral colour).
    """
    ref = np.asarray(lab_ref, np.float32)
    if l_weight is None:
        chroma = float(np.hypot(ref[1], ref[2]))
        l_weight = 0.3 + 0.7 * float(np.clip(1.0 - chroma / 25.0, 0.0, 1.0))
    d = lab_img - ref
    dist = np.sqrt(l_weight * d[..., 0] ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
    return (dist <= tol).astype(np.uint8)


def _fill_small_holes(mask, max_frac=0.05):
    """Fill enclosed holes smaller than `max_frac` of the image (printed dots, thin lines, ...)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(1 - mask, connectivity=4)
    h, w = mask.shape
    x, y = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    touches = (x == 0) | (y == 0) | (x + stats[:, cv2.CC_STAT_WIDTH] == w) | (y + stats[:, cv2.CC_STAT_HEIGHT] == h)
    fill = ~touches & (stats[:, cv2.CC_STAT_AREA] <= max_frac * h * w)
    fill[0] = False
    out = mask.copy()
    out[fill[lab]] = 1
    return out


def inner_fabric_region(color_match, close_frac=0.025, inner_frac=0.012, work_max=640, min_blob=0.25):
    """uint8 mask of the pixels deep inside the fabric.

    The colour mask is closed (bridging printed lines / folds, which have a different colour but
    sit inside the fabric), small holes are filled, and the result is eroded so that the true
    fabric boundary stays outside of it.  DexiNed edges falling inside this region are inner
    edges (printed stripes, seams, folds): they are not fabric boundaries.
    Sizes are fractions of min(image h, w).
    """
    h, w = color_match.shape
    s = min(1.0, work_max / float(max(h, w)))
    cm = (color_match > 0).astype(np.uint8)
    if s < 1:
        cm = (cv2.resize(cm * 255, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA) >= 128
              ).astype(np.uint8)
    # Only the big colour blobs (the fabric pieces) may define the inner region.  Small blobs of the
    # same colour (a bright highlight on a nearby object) must not be bridged to the fabric.
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cm, connectivity=8)
    if n > 1:
        area = stats[:, cv2.CC_STAT_AREA].copy()
        area[0] = 0
        cm = (area >= min_blob * area.max())[lab].astype(np.uint8)
    m = min(cm.shape)
    rc, ri = max(3, int(round(close_frac * m))), max(2, int(round(inner_frac * m)))

    def ell(r):
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

    closed = _fill_small_holes(cv2.morphologyEx(cm, cv2.MORPH_CLOSE, ell(rc)))
    inner = cv2.erode(closed, ell(ri))
    if s < 1:
        inner = cv2.resize(inner, (w, h), interpolation=cv2.INTER_NEAREST)
    return inner


def internal_edges(edge_prob, lab_img, thresh=0.25, max_grad=2.0):
    """uint8 mask of the weak edges: printed stripes, seams, folds inside the fabric.

    A fabric boundary is a strong colour step (fabric / background): the Lab gradient (Sobel, in
    delta-E per pixel) along it is large (median ~7 on the sample frames), while a faint printed line
    is weak (median ~0.5).  The gradient is averaged along the edge, so a single noisy pixel does not
    decide.  Gradients are normalised to a 480 px frame, so `max_grad` does not depend on the resolution.
    """
    h, w = edge_prob.shape
    sc = min(h, w) / 480.0
    ls = cv2.GaussianBlur(lab_img.astype(np.float32), (0, 0), 1.5 * sc)
    g2 = np.zeros((h, w), np.float32)
    for ch in range(3):
        gx = cv2.Sobel(ls[..., ch], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ls[..., ch], cv2.CV_32F, 0, 1, ksize=3)
        g2 += gx ** 2 + gy ** 2
    g = np.sqrt(g2) / 8.0 * sc
    edge = edge_prob >= thresh
    k = (9, 9)
    avg = cv2.blur(np.where(edge, g, 0).astype(np.float32), k) / (cv2.blur(edge.astype(np.float32), k) + 1e-6)
    return (edge & (avg < max_grad)).astype(np.uint8)


def extract_fabric(edge_prob, thresh=0.2, edge_dilate=5, open_px=35, seed=None, color_match=None, lab_img=None):
    """Edge map -> (contour (N,2) float32 in px, filled mask uint8 0/255, info).

    The fabric interior has (almost) no edges, and is enclosed by the fabric
    boundary edge.  So: threshold the edge map, drop the inner edges (weak gradient: with
    `lab_img`; or lying deep inside the colour mask: with `color_match`), thicken edges to close gaps,
    take the connected *non-edge* regions, and open them to cut thin leaks
    into the noisy background.  Which region is the fabric is decided by, in order:
      1. `seed` (x, y): the region containing that point (manual pick);
      2. `color_match`: the region with the most pixels matching the user's fabric
         colour, weighted by purity (matched count x matched fraction), so neither
         the fabric size nor the image border matter;
      3. fallback: the largest region that does not mostly lie along the image border.
    Returns (contour, mask, info); info["frac"] is the fraction of the chosen region
    matching the fabric colour (None without `color_match`).  (None, None, info) if nothing is found.
    """
    info = {"frac": None}
    h, w = edge_prob.shape
    edge = (edge_prob >= thresh).astype(np.uint8)
    if lab_img is not None:
        # printed stripes / seams / folds are weak colour steps, unlike the fabric boundary
        edge[internal_edges(edge_prob, lab_img, thresh) > 0] = 0
    elif color_match is not None and np.mean(color_match > 0) < 0.6:
        # without the Lab image: bridge the stripes with the colour mask instead.  Skipped when the colour
        # mask covers most of the image (tolerance too high): then it says nothing about the fabric.
        edge[inner_fabric_region(color_match) > 0] = 0
    if edge_dilate > 1:
        edge = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_dilate,) * 2))
    free = 1 - edge
    if open_px > 1:
        free = cv2.morphologyEx(free, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px,) * 2))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    if n <= 1:
        return None, None, info
    pick = 0
    if seed is not None:
        sx, sy = int(round(seed[0])), int(round(seed[1]))
        if 0 <= sx < w and 0 <= sy < h:
            pick = int(lab[sy, sx])
    if pick == 0 and color_match is not None:
        cnt = np.bincount(lab[color_match > 0], minlength=n).astype(np.float64)
        area = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
        score = cnt * cnt / np.maximum(area, 1.0)
        score[0] = 0.0
        score[area < 0.005 * h * w] = 0.0            # ignore specks
        if score.max() > 0:
            pick = int(np.argmax(score))
    if pick == 0:
        border = np.zeros_like(lab, dtype=bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        for i in np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1:
            touch = np.count_nonzero(border & (lab == i)) / float(2 * (h + w))
            if touch < 0.5:                  # background hugs the whole border
                pick = int(i)
                break
        if pick == 0:
            pick = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (lab == pick).astype(np.uint8) * 255
    # edges were thickened -> grow the region back so it sits on the true boundary
    if edge_dilate > 1:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_dilate - 1,) * 2))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, None, info
    if color_match is not None:
        info["frac"] = float(np.mean(color_match[mask > 0] > 0))
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    return c, mask, info
