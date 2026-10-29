"""
tracking_helper.py
==================
Temporal stabilization utilities for line detection outputs.

The main class LineStabilizer implements a two-stage filter:
  1. Voting in Hough space (rho, theta) to find the dominant/consensus cluster
     across N frames.  This removes noisy outlier lines that jump far from the
     true edge (background hallucinations).
  2. Averaging the endpoints of every line that belongs to the winning cluster.
     This gives an unbiased, smooth estimate of the true physical edge.

Both stages are run independently for the horizontal and vertical reference
lines selected by _perform_line_detection().
"""

from __future__ import annotations
from typing import Optional, List
import numpy as np
import cv2 as cv


def _line_to_hough(line: List[float]):
    """
    Convert a line segment [x1, y1, x2, y2] to its (rho, theta_deg) Hough
    representation where:
        theta  = direction angle in [0, 180) degrees
        rho    = signed perpendicular distance of the mid-point from the origin
                 along the direction perpendicular to the line.
    """
    x1, y1, x2, y2 = line
    dx, dy = x2 - x1, y2 - y1
    theta_deg = float(np.degrees(np.arctan2(dy, dx)) % 180.0)

    # Perpendicular direction (theta + 90°)
    perp_rad = np.radians(theta_deg + 90.0)
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    rho = mx * np.cos(perp_rad) + my * np.sin(perp_rad)

    return rho, theta_deg


def _canonical_angle_diff(a: float, b: float) -> float:
    """Angular difference in [0, 90] degrees, since lines are undirected."""
    diff = abs(a - b) % 180.0
    if diff > 90.0:
        diff = 180.0 - diff
    return diff


class LineStabilizer:
    """
    Stabilizes the selected horizontal and vertical line detections over a
    rolling window of N frames.

    Strategy (Option 3 from the user's request):
        Stage 1 – Hough-space voting:
            Cluster all buffered lines by (rho, theta).  Keep only lines that
            belong to the largest cluster (the "consensus").  This removes
            frames where the model hallucinated a line far in the background.

        Stage 2 – Endpoint averaging:
            Average the four endpoint coordinates of all lines in the consensus
            cluster.  This gives a smooth, unbiased estimate of the true edge.

    Parameters
    ----------
    window : int
        Number of frames to accumulate before computing the stabilized output.
        A larger window gives more stability but more latency.
    hough_d_thresh : float
        Maximum rho difference (pixels) to consider two lines as belonging
        to the same physical edge.
    hough_theta_thresh : float
        Maximum angle difference (degrees) for the same grouping decision.
    min_cluster_ratio : float
        A cluster must contain at least this fraction of the valid buffer to
        be accepted.  Lines from tiny clusters (< ratio * n_valid) are treated
        as outliers.
    """

    def __init__(
        self,
        window: int = 15,
        hough_d_thresh: float = 7.0, # 12.0,
        hough_theta_thresh: float = 2.0, # 6.0,
        min_cluster_ratio: float = 0.1,
    ):
        self.window = window
        self.hough_d_thresh = hough_d_thresh
        self.hough_theta_thresh = hough_theta_thresh
        self.min_cluster_ratio = min_cluster_ratio

        self._h_buffer: List[Optional[List[float]]] = []
        self._v_buffer: List[Optional[List[float]]] = []
        self._h_weights: List[float] = []
        self._v_weights: List[float] = []

        self._last_h: Optional[List[float]] = None
        self._last_v: Optional[List[float]] = None

    def reset(self):
        """Clear all buffers to restart stabilization (e.g., after physical movement)."""
        self._h_buffer.clear()
        self._v_buffer.clear()
        self._h_weights.clear()
        self._v_weights.clear()
        self._last_h = None
        self._last_v = None

    # ------------------------------------------------------------------
    def update(
        self,
        h_line: Optional[List[float]],
        v_line: Optional[List[float]],
        h_weight: float = 1.0,
        v_weight: float = 1.0,
    ) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """
        Ingests the latest detected horizontal and vertical line candidates along with quality weights
        into the rolling buffer, updates the history, and computes the consensus stabilized line pair.

        Parameters:
            h_line: Optional[List[float]]
                Endpoints [x1, y1, x2, y2] of the candidate horizontal fabric edge for the current frame,
                or None if no horizontal edge was detected.
            v_line: Optional[List[float]]
                Endpoints [x1, y1, x2, y2] of the candidate vertical fabric edge for the current frame,
                or None if no vertical edge was detected.
            h_weight: float
                Confidence / quality weighting multiplier for h_line (e.g. Sobel gradient edge score).
            v_weight: float
                Confidence / quality weighting multiplier for v_line (e.g. Sobel gradient edge score).

        Returns:
            Tuple[Optional[List[float]], Optional[List[float]]]:
                Stabilized (h_line, v_line) pair with smoothed [x1, y1, x2, y2] coordinates.
                Returns (None, None) during the initial warmup phase while the rolling buffer
                is accumulating `window` frames. If a stabilization cycle temporarily fails,
                returns the last known good stabilized line pair.
        """
        # ── Step 1: Append newest candidates and maintain rolling window size ──
        self._h_buffer.append(h_line)
        self._v_buffer.append(v_line)
        self._h_weights.append(h_weight)
        self._v_weights.append(v_weight)

        if len(self._h_buffer) > self.window:
            self._h_buffer.pop(0)
            self._v_buffer.pop(0)
            self._h_weights.pop(0)
            self._v_weights.pop(0)

        # ── Step 2: Warmup check ──────────────────────────────────────────
        if len(self._h_buffer) < self.window:
            # Not enough history yet — wait to fill the window history
            return None, None

        # ── Step 3: Run independent Hough-clustering stabilization ────────
        stab_h = self._stabilize(self._h_buffer, self._h_weights)
        stab_v = self._stabilize(self._v_buffer, self._v_weights)

        # Keep last known good output as fallback across transient detection drops
        if stab_h is not None:
            self._last_h = stab_h
        if stab_v is not None:
            self._last_v = stab_v

        return (
            stab_h if stab_h is not None else self._last_h,
            stab_v if stab_v is not None else self._last_v,
        )

    # ------------------------------------------------------------------
    def _stabilize(self, buffer: List[Optional[List[float]]], weights: List[float]) -> Optional[List[float]]:
        """
        Two-stage consensus filter for a single axis (H or V).

        Clusters line detections in Hough parameter space (rho, theta) to eliminate spurious
        background hallucinations, then calculates a weighted average of line endpoints across
        the winning consensus cluster.

        Parameters:
            buffer: List[Optional[List[float]]]
                Rolling window history of line segments [x1, y1, x2, y2] or None.
            weights: List[float]
                Confidence weights corresponding to each entry in `buffer`.

        Returns:
            Optional[List[float]]:
                Stabilized line segment [x1, y1, x2, y2], or None if no valid line candidates exist.
        """
        valid_indices = [i for i, l in enumerate(buffer) if l is not None]
        if not valid_indices:
            return None
        if len(valid_indices) == 1:
            return buffer[valid_indices[0]]

        valid_lines = [buffer[i] for i in valid_indices]
        valid_weights = [weights[i] for i in valid_indices]

        # ── Stage 1: Hough-space greedy clustering ─────────────────────────
        # Convert each segment to normal distance (rho) and angle (theta) from image origin
        hough = [_line_to_hough(l) for l in valid_lines]
        assigned = [False] * len(valid_lines)
        clusters: List[List[int]] = []

        for i in range(len(valid_lines)):
            if assigned[i]:
                continue
            cluster = [i]
            assigned[i] = True
            rho_i, theta_i = hough[i]

            for j in range(i + 1, len(valid_lines)):
                if assigned[j]:
                    continue
                rho_j, theta_j = hough[j]

                # Group if lines are close in distance (rho) and parallel in orientation (theta)
                if (
                    abs(rho_i - rho_j) < self.hough_d_thresh
                    and _canonical_angle_diff(theta_i, theta_j) < self.hough_theta_thresh
                ):
                    cluster.append(j)
                    assigned[j] = True

            clusters.append(cluster)

        # Select the dominant consensus cluster containing the most detection votes
        largest = max(clusters, key=len)

        # Reject if the largest cluster is too sparse (majority of detections were erratic noise)
        if len(largest) < max(1, self.min_cluster_ratio * len(valid_lines)):
            # Fall back to returning the raw latest line rather than dropping completely
            return buffer[-1]

        # ── Stage 2: Endpoint averaging of the consensus cluster ──────────
        lines_in_cluster = np.array([valid_lines[i] for i in largest], dtype=np.float32)
        cluster_weights = np.array([valid_weights[i] for i in largest], dtype=np.float32)
        
        # Orient all lines to point in the same direction as the reference line.
        # Since detected line segments are undirected ([x1,y1,x2,y2] vs [x2,y2,x1,y1]),
        # flipping lines with a negative dot product prevents endpoint cancellation during averaging.
        ref_line = lines_in_cluster[0]
        ref_dx = ref_line[2] - ref_line[0]
        ref_dy = ref_line[3] - ref_line[1]
        
        for i in range(1, len(lines_in_cluster)):
            line = lines_in_cluster[i]
            dx = line[2] - line[0]
            dy = line[3] - line[1]
            if dx * ref_dx + dy * ref_dy < 0:
                # Flip endpoints to match reference direction
                lines_in_cluster[i] = [line[2], line[3], line[0], line[1]]
                
        # Guard against division by zero if all candidate weights are zero
        sum_weights = cluster_weights.sum()
        if sum_weights < 1e-6:
            cluster_weights = np.ones_like(cluster_weights)
            
        averaged = np.average(lines_in_cluster, axis=0, weights=cluster_weights).tolist()
        return averaged


# ---------------------------------------------------------------------------
# Gradient-based edge strength scoring
# ---------------------------------------------------------------------------

class EdgeGradientScorer:
    """
    Scores how strongly each candidate line aligns with a true image edge
    by measuring the Sobel gradient magnitude along the line segment.

    Motivation
    ----------
    In the fabric adjustor pipeline, line detectors sometimes produce
    "floating" lines in uniform background regions that happen to satisfy
    geometric constraints (angle, distance to origin) but do not correspond
    to any real fabric-background boundary. These floating lines have weak
    or zero image gradients underneath them, whereas true fabric edges sit
    on strong intensity transitions.

    This scorer computes a per-line "edge strength" value that can be used
    as an additional scoring term in `_select_best_lines()` to penalize
    lines that lack gradient support.

    How it works
    ------------
    1. **Precompute gradient map** (once per frame):
       Apply Sobel filters in X and Y, then compute the gradient magnitude
       map: ``mag = sqrt(Gx² + Gy²)``. This highlights all edges in the
       image regardless of orientation.

    2. **Sample along each line**:
       For each candidate line segment, sample ``n_samples`` evenly spaced
       points along its length. Look up the gradient magnitude at each
       sample point. The line's edge strength is the **mean** of these
       sampled gradient values.

    3. **Normalize**:
       Optionally normalize scores to [0, 1] by dividing by the maximum
       gradient magnitude in the frame, making the score comparable across
       different lighting conditions.

    Hyperparameters
    ---------------
    ksize : int, default=3
        Sobel kernel size. Controls the spatial scale of gradient detection.
        - **3** (default): Detects fine edges at 1–2 pixel scale. Good for
          sharp fabric boundaries. This is the standard choice.
        - **5**: Detects slightly broader edges (2–4 pixel scale). More
          robust to blur/noise but may respond to gradual shading.
        - **7**: Very broad edge detection. Use only if the image is very
          noisy or heavily blurred.
        **Rule of thumb**: Use 3 unless the fabric boundary appears soft/blurred
        in the grayscale image, in which case try 5.

    n_samples : int, default=20
        Number of evenly spaced points to sample along each line segment.
        - **Lower values** (5–10): Faster but may miss gradients if the
          line only partially overlaps a real edge.
        - **Higher values** (20–40): More accurate average, captures the
          full edge profile. Diminishing returns beyond ~30.
        - The computational cost is O(n_samples) per line, which is trivial
          compared to the Sobel convolution.
        **Rule of thumb**: 15–25 is a good range. Increase if lines are very
        long (>200 px); decrease if they are short (<50 px).

    normalize : bool, default=True
        Whether to divide each line's score by the maximum gradient value
        in the frame. When True:
        - Scores are in [0.0, 1.0], making them easier to weight against
          other scoring terms (distance, angle) regardless of lighting.
        - A score of 1.0 means "this line sits on the strongest edge in
          the frame"; 0.0 means "zero gradient = uniform region".
        When False:
        - Scores are raw gradient magnitudes (typically 0–1000+ for uint8
          images with Sobel ksize=3). Useful for debugging but harder to
          combine with other score components.
        **Rule of thumb**: Keep True for production use.

    Usage Example
    -------------
    >>> scorer = EdgeGradientScorer(ksize=3, n_samples=20, normalize=True)
    >>>
    >>> # Inside the frame processing loop:
    >>> scorer.precompute(gray)  # gray = preprocessed grayscale frame
    >>>
    >>> # For each candidate line [x1, y1, x2, y2]:
    >>> strength = scorer.score_line(line)
    >>> # strength ∈ [0.0, 1.0] if normalize=True
    >>>
    >>> # Or score all lines at once:
    >>> strengths = scorer.score_lines(lines_2d)
    >>> # strengths is a numpy array of shape (N,)
    """

    def __init__(
        self,
        ksize: int = 3,
        n_samples: int = 20,
        normalize: bool = True,
    ):
        self.ksize = ksize
        self.n_samples = n_samples
        self.normalize = normalize

        # Internal state (set by precompute)
        self._grad_mag: Optional[np.ndarray] = None
        self._max_mag: float = 1.0

    def precompute(self, gray: np.ndarray) -> None:
        """
        Compute the Sobel gradient magnitude map for the current frame.

        Must be called once per frame, before calling score_line() or
        score_lines(). The gradient map is cached internally until the
        next call to precompute().

        Parameters
        ----------
        gray : np.ndarray
            Grayscale image (H, W), dtype uint8 or float32.
            This should be the **preprocessed** grayscale image (after
            smoothing and contrast enhancement) for best results, since
            those steps reduce texture noise while preserving real edges.
        """
        gx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=self.ksize)
        gy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=self.ksize)
        self._grad_mag = np.sqrt(gx * gx + gy * gy)
        self._max_mag = float(self._grad_mag.max()) if self._grad_mag.max() > 0 else 1.0

    def score_lines(self, lines: np.ndarray) -> np.ndarray:
        """
        Computes edge gradient strength scores for an array of candidate line segments.

        Samples Sobel gradient magnitude values at evenly-spaced intervals along each segment's length
        and computes the average intensity. Lines with strong edge gradient support receive higher
        scores, penalizing spurious detections that fall onto uniform fabric or background areas.

        Parameters:
            lines: np.ndarray
                Array of line segments (dtype: float32 or float64, shape: [N, 4]),
                where each row is [x1, y1, x2, y2].

        Returns:
            np.ndarray:
                Per-line gradient strength scores (dtype: float64, shape: [N,]).
                If `normalize=True`, values are bounded within [0.0, 1.0].
        """
        # Guard against uncomputed gradient map or empty line array
        if self._grad_mag is None or lines is None or len(lines) == 0:
            return np.zeros(len(lines) if lines is not None else 0, dtype=np.float64)

        h, w = self._grad_mag.shape[:2]
        n = self.n_samples
        scores = np.zeros(len(lines), dtype=np.float64)

        # ── Step 1: Precompute parametric sampling points t in [0.0, 1.0] ──
        if n > 1:
            t_vals = np.linspace(0.0, 1.0, n)
        else:
            t_vals = np.array([0.5])

        # ── Step 2: Sample gradient magnitude along each line segment ──────
        for idx in range(len(lines)):
            x1, y1, x2, y2 = lines[idx]
            dx, dy = x2 - x1, y2 - y1

            total = 0.0
            count = 0
            for t in t_vals:
                px = int(round(x1 + t * dx))
                py = int(round(y1 + t * dy))
                # Bounds check to avoid accessing pixels outside the frame/ROI
                if 0 <= px < w and 0 <= py < h:
                    total += self._grad_mag[py, px]
                    count += 1

            if count > 0:
                scores[idx] = total / count

        # ── Step 3: Normalize scores to [0.0, 1.0] relative to frame maximum ──
        if self.normalize:
            scores /= self._max_mag

        return scores

