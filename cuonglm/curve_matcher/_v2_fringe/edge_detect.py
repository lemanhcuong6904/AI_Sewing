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


def color_match_mask(lab_img, lab_ref, tol, l_weight=0.3):
    """uint8 mask of pixels whose colour is within `tol` of the reference.

    Lightness is down-weighted so that a bright highlight or a shadow on the fabric
    still matches; the chroma (a, b) does most of the work.
    """
    d = lab_img - np.asarray(lab_ref, np.float32)
    dist = np.sqrt(l_weight * d[..., 0] ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
    return (dist <= tol).astype(np.uint8)




def _ellipse(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _fill_holes(mask, max_frac=0.05):
    """Fill enclosed holes (printed dots, stitching, ...) smaller than `max_frac` of the image.

    The size limit matters: a ring of matching pixels around a big empty area (e.g. a coloured
    frame) must not be 'filled' into a giant blob.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(1 - mask, connectivity=4)
    h, w = mask.shape
    x, y = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    touches = (x == 0) | (y == 0) | (x + stats[:, cv2.CC_STAT_WIDTH] == w) | (y + stats[:, cv2.CC_STAT_HEIGHT] == h)
    fill = ~touches & (stats[:, cv2.CC_STAT_AREA] <= max_frac * h * w)
    fill[0] = False
    out = mask.copy()
    out[fill[lab]] = 1
    return out


def _pick_component(mask, seed):
    """Component of `mask` containing `seed` (x, y), else the largest one."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    if n <= 1:
        return None
    pick = 0
    if seed is not None:
        sx, sy = int(round(seed[0])), int(round(seed[1]))
        if 0 <= sx < mask.shape[1] and 0 <= sy < mask.shape[0]:
            pick = int(lab[sy, sx])
    if pick == 0:
        pick = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == pick).astype(np.uint8)


def _snap_to_edges(mask, prob, rs):
    """Let the boundary settle on the DexiNed edge within +-rs px of the colour mask.

    Watershed with 'surely fabric' = mask eroded by rs and 'surely not' = outside the mask
    dilated by rs.  Everything farther than rs from the colour boundary (inner edges, fringe)
    is out of reach of the edge map, so only the true boundary can attract it.
    """
    if rs < 1:
        return mask
    k = _ellipse(rs)
    fg = cv2.erode(mask, k)
    if not fg.any():
        return mask
    markers = np.zeros(mask.shape, np.int32)
    markers[cv2.dilate(mask, k) == 0] = 1
    markers[fg > 0] = 2
    img = cv2.cvtColor((np.clip(prob, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.watershed(img, markers)
    out = (markers == 2).astype(np.uint8)
    return out if out.any() else mask


def extract_fabric(edge_prob, thresh=0.2, close_frac=0.012, fringe_frac=0.03, seed=None,
                   color_match=None, snap_frac=0.01, work_max=720):
    """Edge map (+ optional fabric-colour mask) -> (contour (N,2) float32 px, mask uint8 0/255, info).

    Sizes are fractions of min(image height, width), so the same settings work for any resolution.

    With `color_match` (colour-first, the reliable path):
      1. fill the small holes (printed dots, stitching);
      2. open by `fringe_frac`: removes loose threads / fringe and thin leaks (before closing,
         otherwise closing would weld the threads into blobs);
      3. close by `close_frac`: joins fabric pieces split by printed lines and fills bays where a
         printed dot touches the boundary;
      4. keep the component containing `seed`, else the largest;
      5. snap the boundary onto the DexiNed edge (within `snap_frac`).
    Without it, the fabric is the region between DexiNed edges (see the fallback below).

    info["frac"] = fraction of the chosen region that matches the fabric colour (None without
    `color_match`).  Returns (None, None, info) if nothing is found.  Heavy work is done at a
    reduced resolution (`work_max`); the returned contour/mask are at the input resolution.
    """
    info = {"frac": None}
    h, w = edge_prob.shape
    s = min(1.0, work_max / float(max(h, w)))
    hs, ws = max(1, int(round(h * s))), max(1, int(round(w * s)))
    prob = cv2.resize(edge_prob, (ws, hs), interpolation=cv2.INTER_AREA) if s < 1 else edge_prob
    cm = None
    if color_match is not None:
        cm = cv2.resize(color_match.astype(np.uint8) * 255, (ws, hs), interpolation=cv2.INTER_AREA) \
            if s < 1 else color_match.astype(np.uint8) * 255
        cm = (cm >= 128).astype(np.uint8)
    seed_s = None if seed is None else (seed[0] * s, seed[1] * s)
    m = min(hs, ws)
    r_close, r_open, r_snap = (int(round(f * m)) for f in (close_frac, fringe_frac, snap_frac))

    if cm is not None:
        mask = _fill_holes(cm)                           # printed dots, stitching
        if r_open > 0:                                   # cut fringe BEFORE closing can merge the threads
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(r_open))
        if r_close > 0:                                  # join pieces split by lines, fill boundary bays
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(r_close))
        mask = _pick_component(_fill_holes(mask), seed_s)
        if mask is None:
            return None, None, info
        mask = _snap_to_edges(mask, prob, r_snap)
    else:
        # Fallback without a colour: regions between DexiNed edges, the biggest one that does not
        # mostly lie along the image border (that would be the background).
        edge = cv2.dilate((prob >= thresh).astype(np.uint8), _ellipse(2))
        free = 1 - edge
        if r_open > 0:
            free = cv2.morphologyEx(free, cv2.MORPH_OPEN, _ellipse(r_open))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
        if n <= 1:
            return None, None, info
        pick = 0
        if seed_s is not None:
            sx, sy = int(round(seed_s[0])), int(round(seed_s[1]))
            if 0 <= sx < ws and 0 <= sy < hs:
                pick = int(lab[sy, sx])
        if pick == 0:
            border = np.zeros_like(lab, dtype=bool)
            border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
            for i in np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1:
                if np.count_nonzero(border & (lab == i)) / float(2 * (hs + ws)) < 0.5:
                    pick = int(i)
                    break
            if pick == 0:
                pick = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = cv2.dilate((lab == pick).astype(np.uint8), _ellipse(1))   # edges were thickened

    if cm is not None:
        info["frac"] = float(np.mean(cm[mask > 0] > 0))
    full = mask * 255
    if s < 1:
        full = (cv2.resize(full, (w, h), interpolation=cv2.INTER_LINEAR) >= 128).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, None, info
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    return c, full, info
