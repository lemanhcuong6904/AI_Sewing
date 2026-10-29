"""Find the (approximately right-angle) bottom-left corner of the fabric and plan the move.

Conventions (image coordinates, y points down):
  * U1 = (1, 0) is the horizontal axis (right), U2 = (0, -1) the vertical axis (up on screen).
  * The target is a right angle at B: one arm along +U1, one arm along +U2, i.e. the fabric
    occupies the area to the right of and above B.  So the fabric corner we look for is the one
    whose two edges, after a rotation, run along +U1 and +U2: the bottom-left corner.
  * theta is applied as R(theta) with R = [[c, -s], [s, c]]: positive = clockwise on screen.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np


def rot2(theta_rad):
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s], [s, c]])


def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def _ang_deg(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _pca_line(P):
    p0 = P.mean(0)
    _, _, vt = np.linalg.svd(P - p0, full_matrices=False)
    return p0, vt[0]


def _fit_line(pts):
    """Robust line fit: PCA, then repeatedly drop the points far from the line (fringe, threads).

    Returns a point on the line, a unit direction and the rms distance of the kept points.
    """
    P = np.asarray(pts, np.float64)
    p0, d = _pca_line(P)
    keep = np.ones(len(P), bool)
    for _ in range(5):
        dist = (P - p0) @ np.array([-d[1], d[0]])
        med = np.median(dist)
        s = max(1.0, 1.4826 * np.median(np.abs(dist - med)))
        k = np.abs(dist - med) <= 2.5 * s
        if k.sum() < max(8, 0.5 * len(P)) or k.sum() == keep.sum():
            break
        keep = k
        p0, d = _pca_line(P[keep])
    dist = (P[keep] - p0) @ np.array([-d[1], d[0]])
    return p0, d, float(np.sqrt(np.mean(dist ** 2)))


def _arm(c, p0, d, pt, ray, tol=6.0, gap=25.0, start_gap=40.0):
    """Contour points lying on the line (p0, d), contiguous from the corner `pt` along `ray`.

    Gaps up to `gap` px (a small bump from a nearby object, a thread) do not interrupt the edge.
    Returns (length along ray, points sorted by distance from the corner).
    """
    n = np.array([-d[1], d[0]])
    near = c[np.abs((c - p0) @ n) <= tol]
    t = (near - pt) @ ray
    keep = t >= -10.0
    near, t = near[keep], t[keep]
    order = np.argsort(t)
    near, t = near[order], t[order]
    if len(t) == 0 or t[0] > start_gap:                # nothing on the line close to the corner
        return 0.0, near[:0]
    last, cnt = float(t[0]), 0                         # the corner itself may be rounded / fringed
    for ti in t:
        if ti - last > gap:
            break
        last = max(last, float(ti))
        cnt += 1
    return float(last), near[:cnt]


def _intersect(p0a, da, p0b, db):
    den = _cross(da, db)
    if abs(den) < 1e-9:
        return None
    s = _cross(p0b - p0a, db) / den
    return p0a + s * da


@dataclass
class Corner:
    point: np.ndarray                  # intersection of the two fitted edge lines (px)
    h_dir: np.ndarray                  # unit vector from the corner along the 'horizontal' edge
    v_dir: np.ndarray                  # unit vector from the corner along the 'vertical' edge
    angle_deg: float                   # interior angle between the two edges
    theta_h: float                     # rotation that aligns h_dir with U1 (deg)
    theta_v: float                     # rotation that aligns v_dir with U2 (deg)
    theta: float                       # the one used, according to rot_alignment (deg)
    h_len: float                       # real length of the edges from the corner (px)
    v_len: float
    h_rmse: float                      # line-fit residuals (px)
    v_rmse: float
    others: list = field(default_factory=list)    # other right-angle corners (px), for display
    h_fit: float = 0.0                             # length of the stretch that defines each line (px, drawn)
    v_fit: float = 0.0
    raw_point: object = None                       # plain intersection of the two fitted lines (px)
    cut: bool = False                              # an edge is short because the image border cuts it off


def find_bottom_left_corner(contour, image_size, u1, u2, rot_alignment="auto",
                            right_tol_deg=12.0, max_theta_deg=50.0, border_px=3.0,
                            min_edge_frac=0.22, corner_border_px=8.0, log=None, eps_frac=0.012,
                            max_fit_px=4.0, long_h=0.8, fit_tol_px=3.0, arm_fit_frac=0.3, min_cut_edge_frac=0.1,
                            occluder=None, long_h_min_theta=0.0):
    """contour: (N,2) closed fabric contour.  image_size = (w, h).

    Returns (Corner, None) or (None, reason).

    The contour is cut into straight edges; two consecutive long edges meeting at roughly 90
    (+-right_tol_deg) form a corner candidate.  Candidates are rejected when an edge is short
    (< min_edge_frac of the image's short side: a corner hidden behind an object, or only partly
    in view) or when the corner lies on the image border (the fabric is cut off there).

    For each candidate a rectangle model is fitted: the common orientation comes from both
    edges (weighted by length / fit residual, so a fringed edge counts less), the two lines are
    forced to be perpendicular and positioned at the robust (median) offset of their edge points.
    rot_alignment: 'auto' (weighted mean of both edges), 'horizontal', 'vertical' or 'mean'
    (these three use the raw, unconstrained edge directions).

    An edge whose far end lies on the image border is cut off (the fabric only partly entered the
    view): it may be shorter, down to min_cut_edge_frac of the short side.  Such a corner is only
    returned when no corner with two complete edges exists.  `occluder` (uint8 image-sized mask) marks
    regions that hide the fabric like the image border does (e.g. a fixed layer of the same colour):
    contour edges lying in it are ignored, and an edge ending in it counts as cut.

    The horizontal arm must be at least long_h times the vertical one (landscape fabric), but only when
    |theta| >= long_h_min_theta (default 0: always).  The alternative reading of a rectangle (the corner
    turned by 90) is out of reach for small theta, so a portrait piece needs no such rule there.
    """
    c = np.asarray(contour, np.float64)
    w, h = image_size
    m = float(min(w, h))

    def cut(p):                                        # the point lies on the image border / in the occluder
        if (p[0] < corner_border_px or p[1] < corner_border_px or
                p[0] > w - 1 - corner_border_px or p[1] > h - 1 - corner_border_px):
            return True
        return bool(occluder is not None and 0 <= p[0] < w and 0 <= p[1] < h
                    and occluder[int(p[1]), int(p[0])] > 0)
    if len(c) < 20:
        return None, "The fabric contour is too short."
    c32 = c.astype(np.float32).reshape(-1, 1, 2)
    peri = cv2.arcLength(c32, True)
    approx = cv2.approxPolyDP(c32, eps_frac * peri, True).reshape(-1, 2).astype(np.float64)
    k = len(approx)
    if k < 3:
        return None, "The fabric outline has fewer than 3 straight edges."
    idx = np.array([int(np.argmin(((c - v) ** 2).sum(1))) for v in approx])
    area = 0.5 * float(np.sum(c[:, 0] * np.roll(c[:, 1], -1) - np.roll(c[:, 0], -1) * c[:, 1]))

    # ---- straight edges between consecutive polygon vertices
    edges = []
    for e in range(k):
        i0, i1 = idx[e], idx[(e + 1) % k]
        pts = c[i0:i1 + 1] if i1 >= i0 else np.vstack([c[i0:], c[:i1 + 1]])
        a, b = approx[e], approx[(e + 1) % k]
        chord = float(np.linalg.norm(b - a))
        hid = (pts[:, 0] < border_px) | (pts[:, 1] < border_px) | (pts[:, 0] > w - 1 - border_px) | (pts[:, 1] > h - 1 - border_px)
        if occluder is not None:
            hid |= occluder[np.clip(pts[:, 1].astype(int), 0, h - 1), np.clip(pts[:, 0].astype(int), 0, w - 1)] > 0
        on_border = float(np.mean(hid)) > 0.5
        edges.append(dict(pts=pts, a=a, b=b, chord=chord, dir=(b - a) / max(chord, 1e-9), border=on_border))
    l_min = max(15.0, 0.08 * m)
    l_min_cut = max(10.0, 0.5 * min_cut_edge_frac * m)          # an edge that runs into the border / occluder
    is_long = [ed["chord"] >= (l_min_cut if (cut(ed["a"]) or cut(ed["b"])) else l_min) for ed in edges]
    gap_max = 0.3 * m
    min_edge = min_edge_frac * m

    def core_of(ed, trim_frac=0.12):
        pts = ed["pts"]
        t = int(trim_frac * len(pts))                  # skip the ends: rounded corners, fringe
        return pts[t:len(pts) - t] if len(pts) - 2 * t >= 10 else pts

    def line_of(ed, at_end):
        """Line of an edge, fitted on the stretch next to the corner (the end of the edge if at_end,
        else its start).  A long edge may bend slightly or be partly hidden, so the fit starts from
        the whole edge and falls back to shorter stretches until the points are straight."""
        pts = core_of(ed)
        if at_end:
            pts = pts[::-1]                            # now pts[0] is the corner end
        dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        fit = None
        for L in [f * m for f in (arm_fit_frac, 0.2, 0.12)]:
            sel = pts[dist <= L]
            if len(sel) < 10:
                break
            p0, d, rmse = _fit_line(sel)
            n = np.array([-d[1], d[0]])
            fit = (p0, d, rmse)
            if float(np.mean(np.abs((sel - p0) @ n) <= fit_tol_px)) >= 0.9:
                break
        if fit is None:                                # very short edge: use all of it
            fit = _fit_line(pts)
        if np.dot(fit[1], ed["dir"]) < 0:              # orient along the direction of travel
            fit = (fit[0], -fit[1], fit[2])
        return fit

    s_ref = np.sign(_cross(u1, u2)) or 1.0
    cands = []
    cnt = {"right": 0, "short": 0, "border": 0}

    def try_pair(e, f):
        pa, da, ra = line_of(edges[e], True)
        pb, db, rb = line_of(edges[f], False)
        pt = _intersect(pa, da, pb, db)
        if pt is None or np.min(np.linalg.norm(c - pt, axis=1)) > 0.1 * m:
            if log is not None:
                log.append(f"edges {e}->{f}: lines meet far from the contour")
            return                                   # the lines meet far from the contour
        if _cross(da, db) * area <= 0:
            return                                   # concave corner
        r1, r2 = -da, db                               # rays leaving the corner along each edge
        ang = float(np.degrees(np.arccos(np.clip(np.dot(r1, r2), -1.0, 1.0))))
        if abs(ang - 90.0) > right_tol_deg:
            if log is not None:
                log.append(f"corner ({pt[0]:.0f},{pt[1]:.0f}): angle {ang:.1f} not within {right_tol_deg} of 90")
            return
        cnt['right'] += 1
        # the real extent of each edge: contour points on its line, bridging small bumps
        len1, pts1 = _arm(c, pa, da, pt, r1)
        len2, pts2 = _arm(c, pb, db, pt, r2)
        if len(pts1) < 15:                             # fringed edge: fall back to the polygon edge
            pts1 = edges[e]["pts"]
        if len(pts2) < 15:
            pts2 = edges[f]["pts"]
        len1 = max(len1, float(np.linalg.norm(edges[e]["a"] - pt)))
        len2 = max(len2, float(np.linalg.norm(edges[f]["b"] - pt)))
        need1 = min_cut_edge_frac * m if cut(pt + r1 * len1) else min_edge   # a cut edge is only partly seen
        need2 = min_cut_edge_frac * m if cut(pt + r2 * len2) else min_edge
        if len1 < need1 or len2 < need2:
            cnt['short'] += 1
            if log is not None:
                log.append(f"corner ({pt[0]:.0f},{pt[1]:.0f}): short edge {min(len1, len2):.0f}px")
            return                                   # an edge is too short to define the corner
        if (pt[0] < corner_border_px or pt[1] < corner_border_px or
                pt[0] > w - 1 - corner_border_px or pt[1] > h - 1 - corner_border_px):
            cnt['border'] += 1
            if log is not None:
                log.append(f"corner ({pt[0]:.0f},{pt[1]:.0f}): on the image border")
            return                                   # the fabric is cut off by the image border
        # which ray is the 'horizontal' one: the fabric must lie between h and v, as between U1 and U2
        if _cross(r1, r2) * s_ref > 0:
            hv, vv, h_len, v_len, h_pts, v_pts, h_r, v_r = r1, r2, len1, len2, pts1, pts2, ra, rb
        else:
            hv, vv, h_len, v_len, h_pts, v_pts, h_r, v_r = r2, r1, len2, len1, pts2, pts1, rb, ra
        h_far, v_far = pt + hv * h_len, pt + vv * v_len
        fit_max = arm_fit_frac * m                     # only the stretch next to the corner defines the lines
        h_pts = h_pts[(h_pts - pt) @ hv <= fit_max]
        v_pts = v_pts[(v_pts - pt) @ vv <= fit_max]
        th_h = _wrap_deg(_ang_deg(u1) - _ang_deg(hv))
        th_v = _wrap_deg(_ang_deg(u2) - _ang_deg(vv))

        def trim(P, frac=0.15):                        # drop the ends (rounded corner, far end)
            t = int(frac * len(P))
            return P[t:len(P) - t] if len(P) - 2 * t >= 8 else P
        h_pts, v_pts = trim(h_pts), trim(v_pts)

        # ---- rectangle model: one orientation for both edges, perpendicular lines, robust offsets
        wh, wv = h_len / (h_r + 0.5) ** 2, v_len / (v_r + 0.5) ** 2
        dv = _wrap_deg(th_v - th_h)
        th_auto = _wrap_deg(th_h + dv * wv / (wh + wv))
        if rot_alignment == "horizontal":
            th = th_h
        elif rot_alignment == "vertical":
            th = th_v
        elif rot_alignment == "mean":
            th = _wrap_deg(th_h + dv / 2.0)
        else:
            th = th_auto
        R_back = rot2(np.radians(-th_auto))            # direction of the edges once aligned: u1/u2 rotated back
        uh, uv = R_back @ np.asarray(u1, float), R_back @ np.asarray(u2, float)
        nh, nv = np.array([-uh[1], uh[0]]), np.array([-uv[1], uv[0]])
        off_h = float(np.median(h_pts @ nh))
        off_v = float(np.median(v_pts @ nv))
        pt_r = _intersect(nh * off_h, uh, nv * off_v, uv)      # point on each line: normal * offset
        if pt_r is None:
            pt_r = pt
        res_h = float(np.sqrt(np.mean((h_pts @ nh - off_h) ** 2)))
        res_v = float(np.sqrt(np.mean((v_pts @ nv - off_v) ** 2)))
        # an arm whose far end is on the image border is cut off: its real length is unknown
        h_cut, v_cut = bool(cut(h_far)), bool(cut(v_far))
        if max(res_h, res_v) > max_fit_px:
            if log is not None:
                log.append(f"corner ({pt_r[0]:.0f},{pt_r[1]:.0f}): edge not straight (fit {res_h:.1f}/{res_v:.1f}px)")
            return
        # square-fabric convention: the horizontal arm is the long side (landscape).  An arm that is
        # cut by the image border is at least as long as what we see.  If it is the horizontal arm
        # and it already looks long enough, the corner is fine.  If it is the vertical arm, the true
        # vertical side may be the longer one, which only matters when the two readings of a
        # rectangle compete (|theta| > 35).
        if h_len < long_h * v_len and abs(th_auto) >= long_h_min_theta:
            if log is not None:
                log.append(f"corner ({pt_r[0]:.0f},{pt_r[1]:.0f}): horizontal arm {h_len:.0f}px is the short side")
            return
        if v_cut and abs(th_auto) > 35.0:
            if log is not None:
                log.append(f"corner ({pt_r[0]:.0f},{pt_r[1]:.0f}): theta {th_auto:+.0f} and the vertical arm is cut by the image border")
            return
        cd = Corner(pt_r, uh, uv, ang, th_h, th_v, th, h_len, v_len, res_h, res_v)
        cd.raw_point = pt
        cd.cut = bool(h_cut or v_cut) and min(h_len, v_len) < min_edge
        cd.h_fit, cd.v_fit = min(h_len, fit_max), min(v_len, fit_max)
        if log is not None:
            log.append(f"corner ({pt_r[0]:.0f},{pt_r[1]:.0f}) ACCEPTED angle {ang:.1f} thH {th_h:+.1f} thV {th_v:+.1f} "
                       f"auto {th_auto:+.1f} len {h_len:.0f}/{v_len:.0f}")
        cands.append(cd)

    # Pairs of long straight edges: e then f along the contour, with at most gap_max px of contour in
    # between (rounding, fringe, or a small bump where a nearby object touches the fabric).
    for e in range(k):
        if not is_long[e] or edges[e]["border"]:
            continue
        f, gap = (e + 1) % k, 0.0
        while f != e and gap <= gap_max:
            if is_long[f] and not edges[f]["border"]:
                try_pair(e, f)
            gap += edges[f]["chord"]
            f = (f + 1) % k
    n_right, n_short, n_border = cnt["right"], cnt["short"], cnt["border"]

    if not cands:
        why = (f"{n_right} right-angle corner(s) were found, but " if n_right else
               "No right-angle corner found on the fabric outline; ")
        reasons = []
        if n_short:
            reasons.append(f"{n_short} had an edge shorter than {min_edge_frac * m:.0f} px "
                           f"(hidden behind an object or partly out of view)")
        if n_border:
            reasons.append(f"{n_border} lay on the image border (the fabric is cut off)")
        if not n_right:
            return None, ("No right-angle corner (90 +- %g deg) with two long straight edges on the "
                          "fabric outline: the corner is probably hidden or out of view." % right_tol_deg)
        return None, why + "; ".join(reasons) + "."
    ok = [cd for cd in cands if abs(cd.theta) <= max_theta_deg]
    if not ok:
        extra = []
        if n_short:
            extra.append(f"{n_short} more rejected for a short edge")
        if n_border:
            extra.append(f"{n_border} more on the image border")
        return None, (f"{len(cands)} valid right-angle corner(s) found, but none can reach the bottom-left "
                      f"position with a rotation under {max_theta_deg:g} deg" + (" (" + ", ".join(extra) + ")" if extra else "") + ".")
    full = [cd for cd in ok if not cd.cut]             # corners with two complete edges come first
    ok = full or ok
    best = min(ok, key=lambda cd: abs(cd.theta))
    if abs(best.theta) > 35.0:
        # near +-45 deg two corners are about equally close to 'bottom-left' (the rectangle can be
        # read in either orientation): prefer the one whose horizontal edge is the longer one
        near = [cd for cd in ok if abs(abs(cd.theta) - abs(best.theta)) <= 8.0]
        best = max(near, key=lambda cd: cd.h_len)
    best.others = [cd.point for cd in cands if cd is not best]
    return best, None


@dataclass
class Plan:
    corner: Corner
    theta: float                       # deg
    t_px: np.ndarray                   # translation applied BEFORE the rotation (px, image axes)
    tx_mm: float                       # along U1
    ty_mm: float                       # along U2
    basis: np.ndarray                  # B
    rot_center: np.ndarray             # the point the rotation is about
    rot_center_name: str
    warnings: list = field(default_factory=list)


def plan_translate_then_rotate(corner, basis, rot_center, rot_center_name, u1, u2, px_per_cm,
                               max_rot_deg=10.0):
    """Translate by t first, then rotate by theta about `rot_center`, so that the corner lands on B:

        R(theta) (C + t - Pc) + Pc = B      =>      t = R^-1 (B - Pc) + Pc - C
    """
    R = rot2(np.radians(corner.theta))
    t = R.T @ (basis - rot_center) + rot_center - corner.point
    tx_mm = float(t @ u1) / px_per_cm[0] * 10.0
    ty_mm = float(t @ u2) / px_per_cm[1] * 10.0
    warns = []
    if abs(corner.theta) > max_rot_deg:
        warns.append(f"Rotation of {abs(corner.theta):.1f} deg exceeds the actuator limit ({max_rot_deg:g} deg).")
    if abs(corner.angle_deg - 90.0) > 5.0:
        warns.append(f"The fabric corner is {corner.angle_deg:.1f} deg, not 90: the two edges cannot both "
                     f"align (theta_h = {corner.theta_h:+.1f}, theta_v = {corner.theta_v:+.1f}).")
    if min(corner.h_len, corner.v_len) < 30.0:
        warns.append("One of the edges next to the corner is short (< 30 px): the corner is less reliable.")
    if max(corner.h_rmse, corner.v_rmse) > 3.0:
        warns.append(f"The edge is not straight (line-fit error {max(corner.h_rmse, corner.v_rmse):.1f} px).")
    return Plan(corner, float(corner.theta), t, tx_mm, ty_mm, np.asarray(basis, float),
                np.asarray(rot_center, float), rot_center_name, warns)


def apply_plan(plan, pts, trans_f=1.0, rot_f=1.0):
    """Move points: translate first (fraction trans_f), then rotate (fraction rot_f) about the centre."""
    pts = np.asarray(pts, float)
    R = rot2(np.radians(plan.theta * rot_f))
    return (pts + plan.t_px * trans_f - plan.rot_center) @ R.T + plan.rot_center
