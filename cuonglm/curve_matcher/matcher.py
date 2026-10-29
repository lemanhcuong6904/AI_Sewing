"""Match a hand-drawn curve to a sub-segment of a closed contour with a rigid transform.

Convention: the transform moves the FABRIC onto the drawn curve,
    p' = R(theta) (p - P) + P + (tx, ty)
Angles are in image coordinates (y points down), so positive theta is
clockwise on screen.  Everything is in pixels.
"""
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------- helpers
def resample(pts, step, closed):
    """Resample a polyline at uniform arc-length spacing."""
    pts = np.asarray(pts, np.float64)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    seg = np.hypot(*np.diff(pts, axis=0).T)
    keep = np.concatenate([[True], seg > 1e-9])          # drop duplicate points
    pts = pts[keep]
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    n = max(int(round(total / step)), 2)
    if closed:
        u = np.arange(n) * (total / n)
    else:
        u = np.linspace(0, total, n)
    return np.stack([np.interp(u, s, pts[:, 0]), np.interp(u, s, pts[:, 1])], 1)


def smooth(pts, sigma, closed):
    """Gaussian smoothing in sample units (circular if closed, else edge-padded)."""
    if sigma <= 0:
        return pts
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    mode = "wrap" if closed else "edge"
    out = np.empty_like(pts)
    for d in range(2):
        out[:, d] = np.convolve(np.pad(pts[:, d], r, mode=mode), k, mode="valid")
    return out


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def kabsch(P, Q):
    """Rigid R, t (no reflection) minimising |R P + t - Q|."""
    mp, mq = P.mean(0), Q.mean(0)
    Pc, Qc = P - mp, Q - mq
    a = np.sum(Pc * Qc)
    b = np.sum(Pc[:, 0] * Qc[:, 1] - Pc[:, 1] * Qc[:, 0])
    R = rot(np.arctan2(b, a))
    return R, mq - R @ mp


def nearest(A, B):
    """For each row of A the index/distance of the closest row of B."""
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    idx = d2.argmin(1)
    return idx, np.sqrt(d2[np.arange(len(A)), idx])


# ------------------------------------------------------------------ result
@dataclass
class MatchResult:
    theta_deg: float
    tx: float
    ty: float
    pivot: np.ndarray
    R: np.ndarray                 # total transform: p' = R p + t
    t: np.ndarray
    rmse: float
    max_err: float
    inlier_ratio: float
    seg_start: int
    seg_len_px: float
    rank: int                     # 0 = best candidate
    score_ratio: float            # window-fit error relative to the best candidate (1.0 = best)
    contour: np.ndarray           # resampled contour used (M, 2)
    seg_idx: np.ndarray           # indices into `contour` of the matched segment
    template: np.ndarray          # resampled drawn curve, oriented as it was matched (n, 2)
    warnings: list = field(default_factory=list)


def _pivot_point(mode, seg_pts, image_size):
    if mode == "image" and image_size is not None:
        return np.array([image_size[0] / 2.0, image_size[1] / 2.0])
    return seg_pts.mean(0)


# -------------------------------------------------------------------- main
def match_curves(contour, drawn, step=2.0, eps=4.0, pivot="segment", image_size=None, top_k=5,
                 border=3.0):
    """Return up to `top_k` distinct candidate matches, best first.

    contour: (N,2) closed fabric contour.  drawn: (K,2) open polyline.
    """
    C = smooth(resample(contour, step, True), 1.5, True)
    T = smooth(resample(drawn, step, False), 1.5, False)
    M, n = len(C), len(T)
    if n < 8:
        raise ValueError("Drawn curve is too short.")
    if n >= M:
        raise ValueError("Drawn curve is longer than the whole fabric contour.")

    shared = []
    chord = T[-1] - T[0]
    cl = np.linalg.norm(chord)
    dv = T - T[0]
    dev = np.abs(chord[0] * dv[:, 1] - chord[1] * dv[:, 0]).max() / cl if cl > 1e-6 else 0.0
    if cl < 1e-6 or dev / (n * step) < 0.08:
        shared.append("Drawn curve is nearly straight: several edge segments will fit it equally well.")

    # ---- 1) sliding-window Kabsch over every start index, both drawing directions
    idx = (np.arange(M)[:, None] + np.arange(n)[None, :]) % M
    Wc = C[idx]
    Wc = Wc - Wc.mean(1, keepdims=True)                   # (M, n, 2)
    err = np.empty((2, M))
    for rev in (0, 1):
        Tt = T[::-1] if rev else T
        Tc = Tt - Tt.mean(0)
        a = (Wc * Tc[None]).sum((1, 2))
        b = (Wc[:, :, 0] * Tc[None, :, 1] - Wc[:, :, 1] * Tc[None, :, 0]).sum(1)
        th = np.arctan2(b, a)
        c, s = np.cos(th)[:, None], np.sin(th)[:, None]
        rx = c * Wc[:, :, 0] - s * Wc[:, :, 1]
        ry = s * Wc[:, :, 0] + c * Wc[:, :, 1]
        err[rev] = ((rx - Tc[None, :, 0]) ** 2 + (ry - Tc[None, :, 1]) ** 2).mean(1)

    # contour that runs along the image border is where the camera cut the fabric off,
    # not a real fabric edge: never match against it
    if image_size is not None:
        w_, h_ = image_size
        bad = ((C[:, 0] < border) | (C[:, 1] < border) |
               (C[:, 0] > w_ - 1 - border) | (C[:, 1] > h_ - 1 - border))
        if bad.any():
            err[:, bad[idx].any(1)] = np.inf
        if not np.isfinite(err).any():
            raise ValueError("The fabric edge is cut off by the image border everywhere that could fit this curve.")

    # ---- 2) distinct candidates (non-maximum suppression, > n/2 apart on the contour)
    order = np.argsort(err.ravel())
    cands = []
    for o in order:
        rev, i = divmod(int(o), M)
        if all(min(abs(i - j), M - abs(i - j)) > n // 2 for _, j, _ in cands):
            if not np.isfinite(err[rev, i]):
                break
            cands.append((float(err[rev, i]), i, rev))
            if len(cands) == top_k:
                break
    e_best = max(cands[0][0], 1e-9)

    results = []
    for rank, (e, i0, rev) in enumerate(cands):
        Tt = T[::-1] if rev else T
        R, t = kabsch(C[idx[i0]], Tt)                     # contour window -> template
        Ra, ta = R.T, -R.T @ t                            # template -> contour frame

        # ---- 3) trimmed ICP (template -> contour) to absorb the imperfect drawing
        def chamfer(Ra_, ta_):
            _, d_ = nearest(Tt @ Ra_.T + ta_, C)
            return np.sqrt(np.mean(np.sort(d_)[: int(0.9 * n)] ** 2))

        base = chamfer(Ra, ta)
        Rb, tb = Ra, ta
        for _ in range(60):
            X = Tt @ Rb.T + tb
            j, d = nearest(X, C)
            keep = d <= np.quantile(d, 0.9)
            Ri, ti = kabsch(X[keep], C[j[keep]])
            Rn, tn = Ri @ Rb, Ri @ tb + ti
            moved = np.abs(Rn - Rb).max() + np.abs(tn - tb).max()
            Rb, tb = Rn, tn
            if moved < 1e-4:
                break
        if chamfer(Rb, tb) <= base:
            Ra, ta = Rb, tb

        # ---- 4) final transform contour -> template, and metrics
        R, t = Ra.T, -Ra.T @ ta
        j, d = nearest(Tt @ Ra.T + ta, C)
        rmse = float(np.sqrt(np.mean(d ** 2)))
        inl = float(np.mean(d <= eps))
        warns = list(shared)
        if inl < 0.8:
            warns.append(f"Only {inl:.0%} of the drawn curve lies within {eps:g}px of the edge.")

        # matched contour segment = smallest circular arc covering the nearest indices
        u = np.unique(j[d <= max(eps, np.median(d) * 2)])
        if len(u) == 0:
            u = np.unique(j)
        gaps = np.diff(np.concatenate([u, [u[0] + M]]))
        g = int(gaps.argmax())
        start = int(u[(g + 1) % len(u)])
        length = M - int(gaps[g]) + 1
        seg_idx = (start + np.arange(length)) % M

        # ---- 5) express as rotation (about pivot) then translation
        P = _pivot_point(pivot, C[seg_idx], image_size)
        th = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        tr = R @ P + t - P
        results.append(MatchResult(th, float(tr[0]), float(tr[1]), P, R, t, rmse, float(d.max()),
                                   inl, start, float(length * step), rank,
                                   float(np.sqrt(e / e_best)), C, seg_idx, Tt, warns))

    if len(results) > 1 and results[1].score_ratio < 1.25:
        results[0].warnings.append(
            f"Ambiguous: candidate 2 fits almost as well (x{results[1].score_ratio:.2f}). Check the alternatives.")
    return results


def match_curve(*args, **kwargs):
    """Best candidate only."""
    return match_curves(*args, **kwargs)[0]
