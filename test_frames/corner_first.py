"""Per-frame detection of the bottom-left right-angle corner of the fabric (Harris + fabric colour).

Fixed camera => the corner is a right angle whose arms run along U1 (right) and U2 (up) within +-theta_max,
with the fabric lying up-right of it.  Steps, for ONE frame (no other frame is needed):

  1. Harris corner detector on the smoothed lightness -> many candidate points (letters on the box, wood
     grain, hands ... all produce corners);
  2. fabric-colour filter: on a small circle around the candidate, the pixels that look like the fabric
     colour must form ONE arc of about 90 deg and the rest of the circle must not look like the fabric.
     That keeps only corners of the FABRIC and gives the approximate directions of its two edges;
  3. the two edges: edge pixels (Canny) lying on a ray leaving the candidate, searched around each
     arc end; each arm must be long, and the colour on its two sides must differ (fabric inside);
  4. the arms must be perpendicular (~90 deg); the best candidate wins (arm lengths x fabric coverage);
  5. joint refinement: one orientation for both edges, perpendicular lines, robust offsets.
"""
import os
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cuonglm.curve_matcher.corner import Corner, _intersect, _wrap_deg, rot2  # noqa: E402
from cuonglm.curve_matcher.edge_detect import bgr_to_lab, bgr_to_lab_image, sample_color  # noqa: E402

__all__ = ["Detection", "detect", "bgr_to_lab", "bgr_to_lab_image", "sample_color", "color_distance"]


@dataclass
class Detection:
    corner: object                     # corner.Corner or None
    mask: np.ndarray                   # fabric cut-out (uint8 0/255)
    why: str = ""
    harris: list = field(default_factory=list)        # all Harris candidates (x, y)
    kept: list = field(default_factory=list)          # candidates that passed the fabric-colour filter
    n_candidates: int = 0


def _ang(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def color_distance(lab, ref_lab, l_weight=None):
    """Distance of every pixel to the fabric colour.

    l_weight: weight of the lightness difference (the chroma difference always has weight 1).  By default it
    grows for neutral colours (white fabric on a dark mat: only the lightness tells them apart).  A neutral
    DARK fabric on saturated wood is the opposite case: use a small weight so that the chroma decides.
    """
    ref = np.asarray(ref_lab, np.float32)
    if l_weight is None:
        chroma = float(np.hypot(ref[1], ref[2]))
        l_weight = 0.3 + 0.7 * float(np.clip(1.0 - chroma / 25.0, 0.0, 1.0))
    wl = float(l_weight)
    d = lab - ref
    return np.sqrt(wl * d[..., 0] ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)


def harris_points(lab, n_max=2500, min_dist=7, sigma=1.2):
    """Harris corner candidates (non-maximum suppressed), strongest first."""
    l8 = np.clip(lab[..., 0] * 2.55, 0, 255).astype(np.float32)
    l8 = cv2.GaussianBlur(l8, (0, 0), sigma)
    r = cv2.cornerHarris(l8, 5, 3, 0.04)
    k = 2 * min_dist + 1
    peak = (r == cv2.dilate(r, np.ones((k, k), np.uint8))) & (r > 1e-4 * r.max())
    ys, xs = np.nonzero(peak)
    if len(xs) > n_max:
        top = np.argpartition(-r[ys, xs], n_max)[:n_max]
        ys, xs = ys[top], xs[top]
    order = np.argsort(-r[ys, xs])
    return np.stack([xs[order], ys[order]], 1).astype(np.float32), r


def edge_points(lab, low=28, high=60, sigma=1.5):
    l8 = np.clip(lab[..., 0] * 2.55, 0, 255).astype(np.uint8)
    l8 = cv2.GaussianBlur(l8, (0, 0), sigma)
    return cv2.Canny(l8, low, high) > 0


def _best_arc(flags):
    """Longest cyclic run of True in a 1-D boolean array. Returns (start_index, length) or (0, 0)."""
    n = len(flags)
    if flags.all():
        return 0, n
    if not flags.any():
        return 0, 0
    f = np.concatenate([flags, flags])
    best_s, best_l, run_s, run_l = 0, 0, 0, 0
    for i in range(2 * n):
        if f[i]:
            if run_l == 0:
                run_s = i
            run_l += 1
            if run_l > best_l:
                best_l, best_s = run_l, run_s
        else:
            run_l = 0
    return best_s % n, min(best_l, n)


def _ray_arm(E, p, alpha_deg, span, step, rmax, tau, gap, start_max):
    """Best ray from p with direction near alpha: the one whose Canny pixels are the longest contiguous run.

    Returns (alpha, length, points) or None.
    """
    rel = E - p
    d2 = (rel ** 2).sum(1)
    near = d2 <= rmax * rmax
    rel, Q = rel[near], E[near]
    if len(rel) < 15:
        return None
    best = None
    for al in np.arange(alpha_deg - span, alpha_deg + span + 1e-6, step):
        c, s = np.cos(np.radians(al)), np.sin(np.radians(al))
        along = rel[:, 0] * c + rel[:, 1] * s
        perp = np.abs(-rel[:, 0] * s + rel[:, 1] * c)
        sel = (perp < tau) & (along > 3.0)
        if sel.sum() < 12:
            continue
        t = np.sort(along[sel])
        if t[0] > start_max:
            continue
        br = np.flatnonzero(np.diff(t) > gap)
        last = t[br[0]] if len(br) else t[-1]
        length = float(last)
        cnt = int((t <= last).sum())
        key = (length, cnt)
        if best is None or key > best[0]:
            best = (key, al, length, Q[sel][along[sel] <= last])
    if best is None:
        return None
    return best[1], best[2], best[3]


def detect(bgr, ref_lab, tol, u1, u2, prior=None, prior_w=0.25, theta_max=50.0, angle_tol=10.0,
           min_arm_frac=0.2, lab=None, tol_in=None, contrast=4.0, l_weight=None,
           tau=4.5, start_max=16.0, long_h=0.8, canny=(15, 40), min_quality=0.45):
    """Find the bottom-left corner of the fabric in one frame.

    tol: fabric colour tolerance (cut-out mask);  tol_in: tolerance of the colour tests (default 1.35 * tol).
    Returns a Detection; its corner is None (with a reason in `why`) when nothing valid is found.
    """
    h, w = bgr.shape[:2]
    m = float(min(h, w))
    if lab is None:
        lab = bgr_to_lab_image(bgr, blur=max(2.0, m / 330.0))
    tol_in = 1.35 * tol if tol_in is None else tol_in
    dmap = color_distance(lab, ref_lab, l_weight)
    M = (dmap <= tol).astype(np.uint8)
    det = Detection(None, M * 255)
    fab_like = dmap <= tol_in

    pts, _ = harris_points(lab)
    pts = pts[(pts[:, 0] > 6) & (pts[:, 0] < w - 7) & (pts[:, 1] > 6) & (pts[:, 1] < h - 7)]
    det.harris = [tuple(p) for p in pts]
    if len(pts) == 0:
        det.why = "No Harris corners in the frame."
        return det

    # ---- 2. fabric-colour filter on a circle around each candidate
    n_ang = 48
    angs = np.arange(n_ang) * (360.0 / n_ang)
    cs = np.stack([np.cos(np.radians(angs)), np.sin(np.radians(angs))], 1)         # (n_ang, 2)
    flags = np.ones((len(pts), n_ang), bool)
    for r in (0.016 * m, 0.027 * m):
        q = pts[:, None, :] + r * cs[None, :, :]
        x = np.clip(np.round(q[..., 0]).astype(int), 0, w - 1)
        y = np.clip(np.round(q[..., 1]).astype(int), 0, h - 1)
        flags &= fab_like[y, x]
    keep = []
    for i in range(len(pts)):
        s, ln = _best_arc(flags[i])
        total = int(flags[i].sum())
        arc_deg = ln * (360.0 / n_ang)
        if 55.0 <= arc_deg <= 130.0 and total - ln <= 2:                  # one arc of fabric, nothing else
            a_start = angs[s] - 0.5 * 360.0 / n_ang
            a_end = angs[(s + ln - 1) % n_ang] + 0.5 * 360.0 / n_ang
            if a_end < a_start:
                a_end += 360.0
            keep.append((i, a_start, a_end))
    det.kept = [tuple(pts[i]) for i, _, _ in keep]
    if not keep:
        det.why = (f"{len(pts)} Harris corners, none has a ~90 deg arc of fabric colour around it: "
                   "pick the fabric colour / adjust the tolerance, or the corner is hidden.")
        return det

    # ---- 3./4. the two edges of every kept candidate
    # edge pixels = Canny edges + boundary of the fabric-colour mask (continuous even along a fringed edge)
    Mc = cv2.morphologyEx(M, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    bnd = (Mc > 0) & (cv2.erode(Mc, np.ones((3, 3), np.uint8)) == 0)
    bnd[:2, :] = bnd[-2:, :] = False
    bnd[:, :2] = bnd[:, -2:] = False
    E = np.argwhere(edge_points(lab, *canny) | bnd)[:, ::-1].astype(np.float32)           # (x, y)
    rmax = 0.45 * m
    gap = 0.03 * m + 6.0
    min_arm = min_arm_frac * m
    delta = max(5.0, 0.011 * m)
    u1, u2 = np.asarray(u1, float), np.asarray(u2, float)
    cands = []
    rej = {}
    for i, a0, a1 in keep:
        p = pts[i].astype(np.float64)
        # arc runs clockwise (increasing angle, y down) from the vertical arm to the horizontal arm
        a_h = _wrapang(a1)
        a_v = _wrapang(a0)
        if abs(a_h) > theta_max + 12.0:
            rej['rotation'] = rej.get('rotation', 0) + 1
            continue
        arm_h = _ray_arm(E, p.astype(np.float32), a_h, 25.0, 0.5, rmax, tau, gap, start_max)
        arm_v = _ray_arm(E, p.astype(np.float32), a_v, 25.0, 0.5, rmax, tau, gap, start_max)
        if arm_h is None or arm_v is None:
            rej['no edge'] = rej.get('no edge', 0) + 1
            continue
        ah, len_h, ph = arm_h
        av, len_v, pv = arm_v
        ang = _wrapang(ah - av)
        if abs(ang - 90.0) > angle_tol or min(len_h, len_v) < min_arm:
            rej['angle/short arm'] = rej.get('angle/short arm', 0) + 1
            if os.environ.get('DBG'): print('   cand', p.astype(int), 'ah %.1f av %.1f lens %.0f %.0f' % (ah, av, len_h, len_v), 'ang %.1f' % ang)
            continue
        dh = np.array([np.cos(np.radians(ah)), np.sin(np.radians(ah))])
        dv = np.array([np.cos(np.radians(av)), np.sin(np.radians(av))])
        # fabric inside (up-right quadrant), different outside, along both arms
        th_c = _wrap_deg(_ang(u1) - _ang(dh))
        if abs(th_c) > theta_max:
            rej['rotation'] = rej.get('rotation', 0) + 1
            continue
        far_h, far_v = p + dh * len_h, p + dv * len_v
        cut = lambda q: q[0] < 8 or q[1] < 8 or q[0] > w - 9 or q[1] > h - 9
        if len_h < long_h * len_v and not cut(far_v):
            rej['short horizontal arm'] = rej.get('short horizontal arm', 0) + 1
            continue
        sup = []
        for d, ln, nin in ((dh, len_h, dv), (dv, len_v, dh)):
            ts = np.arange(8.0, ln, 4.0)
            if len(ts) < 3:
                sup.append(0.0)
                continue
            P = p[None, :] + ts[:, None] * d[None, :]
            xi = lambda Q: (np.clip(np.round(Q[:, 0]).astype(int), 0, w - 1), np.clip(np.round(Q[:, 1]).astype(int), 0, h - 1))
            ix, iy = xi(P + delta * nin)
            ox, oy = xi(P - delta * nin)
            din, dout = dmap[iy, ix], dmap[oy, ox]
            sup.append(float(np.mean((din <= tol_in) & (dout >= din + contrast))))
        quality = 0.5 * (sup[0] + sup[1])
        if quality < min_quality:
            rej['colour sides'] = rej.get('colour sides', 0) + 1
            continue
        gx, gy = np.meshgrid(np.linspace(0.1, 1, 12), np.linspace(0.1, 1, 12))
        Q = p[None, None, :] + (gx * min(len_h, 0.6 * m))[..., None] * dh + (gy * min(len_v, 0.6 * m))[..., None] * dv
        qx = np.clip(np.round(Q[..., 0]).astype(int), 0, w - 1)
        qy = np.clip(np.round(Q[..., 1]).astype(int), 0, h - 1)
        frac = float(fab_like[qy, qx].mean())
        if frac < 0.3:
            rej['quadrant'] = rej.get('quadrant', 0) + 1
            continue
        score = (min(len_h, 0.5 * m) + min(len_v, 0.5 * m)) * (0.4 + frac) * (0.5 + quality)
        if prior is not None:
            score -= prior_w * float(np.linalg.norm(p - np.asarray(prior)))
        cands.append((score, p, ah, av, len_h, len_v, ph, pv, ang, quality))
    det.n_candidates = len(cands)
    if not cands:
        det.why = (f"{len(keep)} fabric-like corner(s), but no pair of long perpendicular edges "
                   f"(arms >= {min_arm:.0f} px, fabric up-right) [{rej}]: the corner is probably hidden or out of view.")
        return det
    cands.sort(key=lambda c: -c[0])
    score, p, ah, av, len_h, len_v, ph, pv, ang, quality = cands[0]

    # ---- 5. joint refinement: shared orientation, perpendicular lines, robust offsets
    dh = np.array([np.cos(np.radians(ah)), np.sin(np.radians(ah))])
    dv = np.array([np.cos(np.radians(av)), np.sin(np.radians(av))])
    th_h = _wrap_deg(_ang(u1) - _ang(dh))
    th_v = _wrap_deg(_ang(u2) - _ang(dv))

    def rms(P, d):
        n = np.array([-d[1], d[0]])
        return float(np.sqrt(np.mean(((P - P.mean(0)) @ n) ** 2))) if len(P) > 3 else 5.0

    wh, wv = len_h / (rms(ph, dh) + 0.5) ** 2, len_v / (rms(pv, dv) + 0.5) ** 2
    th_auto = _wrap_deg(th_h + _wrap_deg(th_v - th_h) * wv / (wh + wv))
    R_back = rot2(np.radians(-th_auto))
    uh, uv = R_back @ u1, R_back @ u2
    nh, nv = np.array([-uh[1], uh[0]]), np.array([-uv[1], uv[0]])
    ha, hb = ph.astype(np.float64), pv.astype(np.float64)
    off_h, off_v = float(np.median(ha @ nh)), float(np.median(hb @ nv))
    c_r = _intersect(nh * off_h, uh, nv * off_v, uv)
    if c_r is None:
        c_r = p
    res_h = float(np.sqrt(np.mean((ha @ nh - off_h) ** 2)))
    res_v = float(np.sqrt(np.mean((hb @ nv - off_v) ** 2)))
    cd = Corner(c_r, uh, uv, ang, th_h, th_v, th_auto, float(len_h), float(len_v), res_h, res_v)
    cd.raw_point = p
    cd.others = [c[1] for c in cands[1:6]]
    det.corner = cd

    # cut-out of the fabric: fabric-coloured pixels up-right of the two edges, connected to the corner
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rel = np.stack([xx - c_r[0], yy - c_r[1]], -1)
    sgn_h = np.sign(float(uv @ nh)) or 1.0
    sgn_v = np.sign(float(uh @ nv)) or 1.0
    quad = ((rel @ nh) * sgn_h >= -2.0) & ((rel @ nv) * sgn_v >= -2.0)
    mq = (M & quad).astype(np.uint8)
    mq = cv2.morphologyEx(mq, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    seed_pt = c_r + 14 * uh + 14 * uv
    n_cc, lab_cc, _, _ = cv2.connectedComponentsWithStats(mq, connectivity=4)
    sx, sy = int(round(seed_pt[0])), int(round(seed_pt[1]))
    if 0 <= sx < w and 0 <= sy < h and lab_cc[sy, sx] > 0:
        mq = (lab_cc == lab_cc[sy, sx]).astype(np.uint8)
    det.mask = mq * 255
    return det


def _wrapang(a):
    return (a + 180.0) % 360.0 - 180.0
