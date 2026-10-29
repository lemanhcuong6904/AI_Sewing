import os
import time
import math
from datetime import datetime
from typing import Tuple, Optional, List, Union
import cv2 as cv
import numpy as np
import json

from utils.camera import get_camera
from utils.line_detectors.detector import (
    DetectorMode,
    preprocessing_adjustor,
    postprocess_lines,
    keep_lines_near_background,
    filter_dominant_angles,
    filter_noisy_lines_by_area,
    map_noisy_areas_to_roi,
    detect_lines,
    ensemble_detect_lines,
    _ENSEMBLE_COLOURS,
    _FUSED_COLOUR,
    preprocessing_adjustor_batch,
    detect_lines_batch,
)
from utils.helper import (
    draw_lines,
    _draw_lines_colour,
    visualize_and_save_frame_adjustor,
)
from utils.constants import (
    DEFAULT_X_BASIS,
    DEFAULT_Y_BASIS,
    U1_VECTOR,
    U2_VECTOR,
    AXIS_LENGTH_PX,
    TRANSLATION_DEADBAND_CM,
    ROTATION_DEADBAND_DEG,
    PX_PER_CM_X,
    PX_PER_CM_Y,
    DEFAULT_PIVOT,
)
from utils.viz_adjustor import draw_adjustor_overlay
from utils.tracking_helper import LineStabilizer, EdgeGradientScorer


class LineDetectionAdjustor:
    """
    OOP wrapper for the M-LSD / classical line detection fabric adjustment pipeline.

    Usage:
        adjustor = LineDetectionAdjustor(x_basis=DEFAULT_X_BASIS, y_basis=DEFAULT_Y_BASIS, ...)
        for control_signals in adjustor.run(visualize=False):
            ...  # control_signals is None during cooldown, else ((tx,ty),(ah,av))
    """

    def __init__(
        self,
        x_basis: Optional[float] = None,
        y_basis: Optional[float] = None,
        img_size=None,
        enhance_contrast: bool = False,
        smooth_texture: bool = True,
        detector_mode: DetectorMode = DetectorMode.MLSD,
        bin_num: int = 32,
        mask_thresh: int = 30,
        use_gray: bool = False,
        velocity: Optional[List] = None,
        speed_state: Optional[List] = None,
        cooldown_time: float = 0.0,
        batch_size: int = 5,
        crop_roi: Optional[Tuple[int, int, int, int]] = None,
        alpha_prior: float = 90.0,
        angle_tolerance: float = 8.0,
        gradient_weight: float = 10.0,
        length_weighting: str = 'linear',
        line_selection_mode: str = 'gradient',
        pivot: Optional[Tuple[float, float]] = None,
        rot_alignment: str = 'horizontal',
        compensate_rotation: bool = True,
        max_rot_deg: float = 5.0,
        # noisy_areas: Optional[List[Tuple[float, float, float, float]]] = None,
    ):
        self.x_basis = x_basis if x_basis is not None else DEFAULT_X_BASIS
        self.y_basis = y_basis if y_basis is not None else DEFAULT_Y_BASIS
        self.pivot = pivot if pivot is not None else DEFAULT_PIVOT
        self.rot_alignment = rot_alignment
        self.compensate_rotation = compensate_rotation
        self.max_rot_deg = max_rot_deg
        # Exclusion zones defined in fixed original camera space (cm)
        # Fixtures: The hand (~381 px, 90 px) and needle plate hole (~264 px, 142.5 px)
        self.noisy_areas = [
            (385.0 / PX_PER_CM_X, 60.0 / PX_PER_CM_Y, 2.5, 12.0, 12.0, 'horizontal'), # the hand (~9.77 cm, ~2.57 cm)
            (385.0 / PX_PER_CM_X, 60.0 / PX_PER_CM_Y, 2.5, 12.0, 1.7, 'vertical'),    # the upper hand
            (264.0 / PX_PER_CM_X, 129 / PX_PER_CM_Y, 1.7, 5.5, 8.0, 'horizontal'),  # the hole (~6.77 cm, ~4.07 cm)
            # (264.0 / PX_PER_CM_X, 129 / PX_PER_CM_Y, 1.7, 5.5, 2, 'vertical'), # the hole
            # (443.4 / PX_PER_CM_X, 254.5 / PX_PER_CM_Y, 0.6, 8.0, 8.0, 'horizontal'),# the bottom hole
            # (6.08, 3.96, 1.51, 0.36, 3.5, 'vertical'),
        ]
        self.batch_size = batch_size
        self.img_size = img_size
        self.enhance_contrast = enhance_contrast
        self.smooth_texture = smooth_texture
        self.detector_mode = detector_mode
        self.mask_thresh = mask_thresh
        self.cooldown_time = cooldown_time
        self.crop_roi = crop_roi
        self.alpha_prior = alpha_prior
        self.angle_tolerance = angle_tolerance
        self.gradient_weight = gradient_weight
        self.length_weighting = length_weighting
        self.line_selection_mode = line_selection_mode  # 'angle' or 'gradient'
        self._edge_scorer = EdgeGradientScorer(ksize=3, n_samples=20, normalize=True)

    def update_pivot(self, delta_px_x: float, delta_px_y: float) -> Tuple[float, float]:
        """Translate the rotation center (pivot) by the given pixel displacement.

        Args:
            delta_px_x: Horizontal shift in original camera pixels.
            delta_px_y: Vertical shift in original camera pixels.

        Returns:
            Tuple[float, float]: Updated (px, py) pivot coordinates.
        """
        px, py = self.pivot
        self.pivot = (px + delta_px_x, py + delta_px_y)
        return self.pivot

    def reset_pivot(self, pivot: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
        """Reset rotation center (pivot) back to the home coordinate DEFAULT_PIVOT.

        Args:
            pivot: Optional override for the home coordinates. Defaults to DEFAULT_PIVOT.

        Returns:
            Tuple[float, float]: Reset (px, py) pivot coordinates.
        """
        self.pivot = pivot if pivot is not None else DEFAULT_PIVOT
        return self.pivot

    # ------------------------------------------------------------------
    # Static helpers (kept as static so module-level aliases work cleanly)
    # ------------------------------------------------------------------

    @staticmethod
    def _perform_line_detection(
        frame: np.ndarray,
        img_size: Optional[Tuple[int, int]],
        enhance_contrast: bool,
        smooth_texture: bool,
        detector_mode: DetectorMode,
        mask_thresh: int,
        x_basis: Optional[float] = None,
        y_basis: Optional[float] = None,
        crop_roi: Optional[tuple] = None,
        noisy_areas: Optional[list] = None,
        alpha_prior: float = 90.0,
        angle_tolerance: float = 5.0,
        gradient_weight: float = 3.0,
        length_weighting: str = 'linear',
        edge_scorer: Optional['EdgeGradientScorer'] = None,
        line_selection_mode: str = 'gradient',
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[List[float]], Optional[List[float]], Optional[List[np.ndarray]]]:
        """Perform end-to-end frame preprocessing, line detection, and best line pair extraction.

        Executes the single-frame line processing pipeline:
          1. Crops and pre-processes input frame to grayscale and normalized ROI.
          2. Runs selected line detector backend (M-LSD, LSD, FLD, EDLines, or Ensemble).
          3. Rejects false edges via background proximity thresholding and noise exclusion masks.
          4. Computes edge gradient magnitude scores and prunes lines via dominant-angle clustering.
          5. Remaps surviving lines back into the full-resolution sensor coordinate frame.
          6. Selects the optimal (horizontal, vertical) line pair forming the fabric corner.

        Args:
            frame: np.ndarray (dtype: uint8, shape: [H, W, 3])
                Raw RGB/BGR video frame captured from camera.
            img_size: Optional[Tuple[int, int]]
                Target resolution (width, height) to resize ROI for detector inference. If None,
                retains cropped dimensions.
            enhance_contrast: bool
                Whether to apply CLAHE contrast enhancement during preprocessing.
            smooth_texture: bool
                Whether to apply Gaussian texture smoothing to suppress fabric weave patterns.
            detector_mode: DetectorMode
                Active line detector backend enum (e.g. MLSD, LSD, ENSEMBLE).
            mask_thresh: int
                Luminance threshold used by keep_lines_near_background to isolate fabric boundary.
            x_basis: Optional[float]
                Origin X coordinate (in full frame pixels) used to score proximity of lines.
            y_basis: Optional[float]
                Origin Y coordinate (in full frame pixels) used to score proximity of lines.
            crop_roi: Optional[Tuple[int, int, int, int]]
                Bounding box (sx, ex, sy, ey) defining region of interest in the input frame.
            noisy_areas: Optional[List[Tuple[float, float, float, float, float, str]]]
                List of physical exclusion zones (x_cm, y_cm, h_cm, w_cm, length_thr, orientation)
                to filter out mechanical fixtures (e.g. sewing foot, needles).
            alpha_prior: float
                Expected corner angle in degrees between horizontal and vertical edges (default: 90.0°).
            angle_tolerance: float
                Allowable angular deviation (degrees) from alpha_prior before applying penalty.
            gradient_weight: float
                Relative weighting multiplier of Sobel edge gradient magnitude in line scoring.
            length_weighting: str
                Weighting strategy used during dominant angle clustering ('linear' or 'none').
            edge_scorer: Optional[EdgeGradientScorer]
                Pre-instantiated Sobel gradient scorer used to evaluate edge strength along lines.
            line_selection_mode: str
                Line pairing criterion: 'gradient' (edge strength priority) or 'angle' (angle priority).

        Returns:
            Tuple containing:
                gray: np.ndarray (dtype: uint8, shape: [H_roi, W_roi])
                    Preprocessed grayscale ROI image used for detection.
                img: np.ndarray (dtype: uint8, shape: [H_roi, W_roi, 3])
                    Preprocessed BGR ROI image.
                lines: Optional[np.ndarray] (dtype: float32, shape: [N, 4])
                    All surviving detected line segments [x1, y1, x2, y2] mapped to full-frame coordinates.
                best_h_line: Optional[List[float]]
                    Coordinates [x1, y1, x2, y2] of the selected horizontal fabric edge, or None.
                best_v_line: Optional[List[float]]
                    Coordinates [x1, y1, x2, y2] of the selected vertical fabric edge, or None.
                per_det: Optional[List[np.ndarray]]
                    Intermediate detections per sub-detector (populated only when detector_mode == ENSEMBLE).
        """
        # ── Step 1: ROI cropping, grayscale conversion, and contrast enhancement ──
        # offset contains (sx, sy, scale_x, scale_y) to remap ROI coords back to full frame
        gray, img, offset = preprocessing_adjustor(
            frame,
            img_size=img_size,
            crop_roi=crop_roi,
            enhance_contrast=enhance_contrast,
            smooth_texture=smooth_texture,
        )

        per_det = None
        if detector_mode == DetectorMode.ENSEMBLE:
            lines, scores, per_det = ensemble_detect_lines(gray)
        else:
            # ── Step 2: Detection and baseline geometric pruning ──
            lines = detect_lines(gray, detector_mode, img)
            if detector_mode != DetectorMode.MLSD:
                lines = postprocess_lines(lines)
            # Retain only line segments that border the fabric silhouette boundary
            lines = keep_lines_near_background(lines, gray, mask_thresh=30, mode="adjustor")

            # Filter out lines falling into pre-calibrated machine fixture zones
            if detector_mode == DetectorMode.MLSD and noisy_areas:
                lines = filter_noisy_lines_by_area(lines, noisy_areas, offset=offset)

            # ── Step 3: Gradient scoring and dominant angle clustering in ROI space ──
            grad_scores = None
            if edge_scorer is not None and lines is not None and len(lines) > 0:
                edge_scorer.precompute(gray)
                grad_scores = edge_scorer.score_lines(lines.reshape(-1, 4))

            # Filter lines that do not align with dominant fabric boundary angle clusters
            lines = filter_dominant_angles(lines, gradient_scores=grad_scores, length_weighting=length_weighting, offset=offset)

            # Re-score surviving lines (indices changed after dominant angle filtering)
            grad_scores = None
            if edge_scorer is not None and lines is not None and len(lines) > 0:
                grad_scores = edge_scorer.score_lines(lines.reshape(-1, 4))

            if lines is not None:
                lines = lines.astype(np.float32)
            
            # ── Step 4: Rescale line endpoints from ROI coords to full-frame sensor coords ──
            if lines is not None and len(lines) > 0:
                lines[:, 0] = lines[:, 0] * offset[2] + offset[0]
                lines[:, 1] = lines[:, 1] * offset[3] + offset[1]
                lines[:, 2] = lines[:, 2] * offset[2] + offset[0]
                lines[:, 3] = lines[:, 3] * offset[3] + offset[1]

        # ── Step 5: Select the optimal corner pair (H and V edges) ──
        best_h_line, best_v_line = LineDetectionAdjustor._select_best_lines(
            lines, img.shape, x_basis, y_basis, U1_VECTOR, U2_VECTOR,
            alpha_prior=alpha_prior, angle_tolerance=angle_tolerance,
            gradient_weight=gradient_weight, gradient_scores=grad_scores,
            line_selection_mode=line_selection_mode,
        )

        return gray, img, lines, best_h_line, best_v_line, per_det

    @staticmethod
    def _select_best_lines(
        lines, img_shape, x_basis=None, y_basis=None, u1_vector=None, u2_vector=None,
        alpha_prior: float = 90.0, angle_tolerance: float = 5.0,
        gradient_weight: float = 23.0, gradient_scores: Optional[np.ndarray] = None,
        return_scores: bool = False,
        line_selection_mode: str = 'gradient',
    ):
        """
        Dispatcher method that routes to either angle or gradient line selection.
        
        Args:
            line_selection_mode: 'angle' for original implementation or 
                                'gradient' for gradient-focused approach
        """
        if line_selection_mode == 'angle':
            return LineDetectionAdjustor._select_best_lines_angle_based(
                lines, img_shape, x_basis, y_basis, u1_vector, u2_vector,
                alpha_prior, angle_tolerance, gradient_weight, gradient_scores, return_scores
            )
        else:  # default to gradient
            return LineDetectionAdjustor._select_best_lines_gradient_priority(
                lines, img_shape, x_basis, y_basis, u1_vector, u2_vector,
                alpha_prior, angle_tolerance, gradient_weight, gradient_scores, return_scores
            )

    @staticmethod
    def _select_best_lines_angle_based(
        lines, img_shape, x_basis=None, y_basis=None, u1_vector=None, u2_vector=None,
        alpha_prior: float = 90.0, angle_tolerance: float = 5.0,
        gradient_weight: float = 23.0, gradient_scores: Optional[np.ndarray] = None,
        return_scores: bool = False,
    ):
        """
        Original angle-based line selection implementation.
        
        Prioritizes lines that form angles close to the prior angle constraint.
        Uses gradient scores as a bonus term, not as the primary criterion.
        This approach can stall when fabric edges deviate from the expected angle.
        """
        if u1_vector is None:
            u1_vector = U1_VECTOR
        if u2_vector is None:
            u2_vector = U2_VECTOR

        best_h_line = None
        best_v_line = None
        best_h_score = 1.0
        best_v_score = 1.0

        if lines is not None and len(lines) > 0:
            lines_2d = lines.reshape(-1, 4)
            dx = lines_2d[:, 2] - lines_2d[:, 0]
            dy = lines_2d[:, 3] - lines_2d[:, 1]
            angles = np.arctan2(dy, dx)

            lengths = np.hypot(dx, dy)
            lengths[lengths == 0] = 1e-6
            dir_x = dx / lengths
            dir_y = dy / lengths
            
            # Absolute dot products with U1 and U2
            dot_u1 = np.abs(dir_x * u1_vector[0] + dir_y * u1_vector[1])
            dot_u2 = np.abs(dir_x * u2_vector[0] + dir_y * u2_vector[1])

            H, W = img_shape[:2]
            best_score = -float('inf')
            n_lines = len(lines_2d)

            mean_x = (lines_2d[:, 0] + lines_2d[:, 2]) / 2.0
            mean_y = (lines_2d[:, 1] + lines_2d[:, 3]) / 2.0
            
            origin = np.array([x_basis if x_basis is not None else 0.0, 
                               y_basis if y_basis is not None else 0.0])

            for i in range(n_lines):
                for j in range(i + 1, n_lines):
                    angle_diff = angles[i] - angles[j]
                    diff_deg = abs(np.degrees(angle_diff)) % 180.0
                    if diff_deg > 90.0:
                        diff_deg = 180.0 - diff_deg
                        
                    # STRICT angle constraint: hard skip if deviation exceeds tolerance
                    if abs(diff_deg - alpha_prior) > angle_tolerance:
                        continue

                    # Assign U1 and U2 lines based on which pair configuration has better total alignment
                    if dot_u1[i] + dot_u2[j] > dot_u1[j] + dot_u2[i]:
                        u1_idx, u2_idx = i, j
                    else:
                        u1_idx, u2_idx = j, i

                    # Ensure candidate lines are genuinely aligned with physical basis vectors U1 (horizontal) and U2 (vertical)
                    if dot_u1[u1_idx] < 0.80 or dot_u2[u2_idx] < 0.80:
                        continue

                    # Distance of U1 line to X-axis (measured along U2)
                    r_u1 = np.array([mean_x[u1_idx], mean_y[u1_idx]], dtype=np.float64) - origin
                    dist_u1_to_x = abs(float(np.dot(r_u1, u2_vector)))
                    
                    # Distance of U2 line to Y-axis (measured along U1)
                    r_u2 = np.array([mean_x[u2_idx], mean_y[u2_idx]], dtype=np.float64) - origin
                    dist_u2_to_y = abs(float(np.dot(r_u2, u1_vector)))

                    # Distance-based score with strong angle penalty
                    angle_penalty = (abs(diff_deg - alpha_prior) / angle_tolerance) * 5.0
                    score = -(dist_u1_to_x / H) - (dist_u2_to_y / W) - angle_penalty

                    # Gradient edge strength bonus: reward lines that sit on real image edges
                    if gradient_scores is not None and len(gradient_scores) > 0:
                        grad_bonus = 0.0
                        if u1_idx < len(gradient_scores):
                            grad_bonus += gradient_scores[u1_idx]
                        if u2_idx < len(gradient_scores):
                            grad_bonus += gradient_scores[u2_idx]
                        grad_bonus /= 2.0
                        score += grad_bonus * gradient_weight

                    if score > best_score:
                        best_score = score
                        best_h_line = lines_2d[u1_idx].tolist()
                        best_v_line = lines_2d[u2_idx].tolist()
                        if return_scores and gradient_scores is not None:
                            best_h_score = gradient_scores[u1_idx] if u1_idx < len(gradient_scores) else 0.0
                            best_v_score = gradient_scores[u2_idx] if u2_idx < len(gradient_scores) else 0.0

        if best_h_line is not None:
            len_h = np.hypot(best_h_line[2] - best_h_line[0], best_h_line[3] - best_h_line[1]) / PX_PER_CM_X
            if len_h < 1.5:  # Lowered threshold to prevent valid lines from being dropped
                best_h_line = None
                
        if best_v_line is not None:
            len_v = np.hypot(best_v_line[2] - best_v_line[0], best_v_line[3] - best_v_line[1]) / PX_PER_CM_Y
            if len_v < 1.5:  # Lowered threshold to prevent valid lines from being dropped
                best_v_line = None
                
        if best_h_line is None or best_v_line is None:
            if return_scores:
                return None, None, 0.0, 0.0
            return None, None
            
        if return_scores:
            return best_h_line, best_v_line, best_h_score, best_v_score
        return best_h_line, best_v_line

    @staticmethod
    def _select_best_lines_gradient_priority(
        lines: Optional[np.ndarray],
        img_shape: Tuple[int, ...],
        x_basis: Optional[float] = None,
        y_basis: Optional[float] = None,
        u1_vector: Optional[np.ndarray] = None,
        u2_vector: Optional[np.ndarray] = None,
        alpha_prior: float = 90.0,
        angle_tolerance: float = 5.0,
        gradient_weight: float = 23.0,
        gradient_scores: Optional[np.ndarray] = None,
        return_scores: bool = False,
    ):
        """Select the best horizontal and vertical line pair forming the fabric corner.

        Evaluates all candidate line pairs using a multi-criteria optimization function:
          - Edge gradient strength (primary: strongly favors true fabric contrast boundaries)
          - Geometric proximity to the corner origin (secondary: penalizes distant artifacts)
          - Angle deviation from expected prior angle alpha_prior (tertiary: penalizes skew)
          - Dot-product alignment with physical basis vectors U1 and U2

        Args:
            lines: Optional[np.ndarray] (dtype: float32, shape: [N, 4])
                Candidate line segments [x1, y1, x2, y2] in full-frame pixel coordinates.
            img_shape: Tuple[int, ...]
                Dimensions of the image frame (H, W, ...).
            x_basis: Optional[float]
                Reference corner origin X coordinate in pixels.
            y_basis: Optional[float]
                Reference corner origin Y coordinate in pixels.
            u1_vector: Optional[np.ndarray] (shape: [2,])
                Direction vector for horizontal reference axis (defaults to U1_VECTOR).
            u2_vector: Optional[np.ndarray] (shape: [2,])
                Direction vector for vertical reference axis (defaults to U2_VECTOR).
            alpha_prior: float
                Expected corner angle in degrees between the two intersecting edges.
            angle_tolerance: float
                Allowable angular deviation (degrees) before applying smooth angular penalty.
            gradient_weight: float
                Scaling factor for edge gradient strength in the composite score.
            gradient_scores: Optional[np.ndarray] (shape: [N,])
                Precomputed Sobel edge gradient magnitude score for each candidate line segment.
            return_scores: bool
                If True, returns individual line gradient scores alongside line coordinates.

        Returns:
            If return_scores is False:
                best_h_line: Optional[List[float]] - Coordinates [x1, y1, x2, y2] of the horizontal line, or None.
                best_v_line: Optional[List[float]] - Coordinates [x1, y1, x2, y2] of the vertical line, or None.
            If return_scores is True:
                best_h_line: Optional[List[float]]
                best_v_line: Optional[List[float]]
                best_h_score: float - Edge gradient score for the horizontal line.
                best_v_score: float - Edge gradient score for the vertical line.
        """
        if u1_vector is None:
            u1_vector = U1_VECTOR
        if u2_vector is None:
            u2_vector = U2_VECTOR

        best_h_line = None
        best_v_line = None
        best_h_score = 1.0
        best_v_score = 1.0

        if lines is not None and len(lines) > 0:
            lines_2d = lines.reshape(-1, 4)
            dx = lines_2d[:, 2] - lines_2d[:, 0]
            dy = lines_2d[:, 3] - lines_2d[:, 1]
            angles = np.arctan2(dy, dx)

            lengths = np.hypot(dx, dy)
            lengths[lengths == 0] = 1e-6
            dir_x = dx / lengths
            dir_y = dy / lengths
            
            # Absolute dot products measuring alignment with target axes U1 and U2
            dot_u1 = np.abs(dir_x * u1_vector[0] + dir_y * u1_vector[1])
            dot_u2 = np.abs(dir_x * u2_vector[0] + dir_y * u2_vector[1])

            H, W = img_shape[:2]
            best_score = -float('inf')
            n_lines = len(lines_2d)

            mean_x = (lines_2d[:, 0] + lines_2d[:, 2]) / 2.0
            mean_y = (lines_2d[:, 1] + lines_2d[:, 3]) / 2.0
            
            origin = np.array([x_basis if x_basis is not None else 0.0, 
                               y_basis if y_basis is not None else 0.0])

            # ── Pairwise candidate evaluation loop ──
            for i in range(n_lines):
                for j in range(i + 1, n_lines):
                    angle_diff = angles[i] - angles[j]
                    diff_deg = abs(np.degrees(angle_diff)) % 180.0
                    if diff_deg > 90.0:
                        diff_deg = 180.0 - diff_deg
                    
                    # Enforce strict corner angle constraint: deviation from alpha_prior must not exceed angle_tolerance (±5°)
                    if abs(diff_deg - alpha_prior) > angle_tolerance:
                        continue

                    # Assign line i and line j to U1 (horizontal) vs U2 (vertical) based on maximal alignment
                    if dot_u1[i] + dot_u2[j] > dot_u1[j] + dot_u2[i]:
                        u1_idx, u2_idx = i, j
                    else:
                        u1_idx, u2_idx = j, i

                    # Ensure candidate lines are genuinely aligned with physical basis vectors U1 (horizontal) and U2 (vertical)
                    # dot >= 0.80 ensures alignment within ~36.8° of target axes, rejecting diagonal edges
                    if dot_u1[u1_idx] < 0.80 or dot_u2[u2_idx] < 0.80:
                        continue

                    # Distance of horizontal line midpoint to X-axis (measured along U2)
                    r_u1 = np.array([mean_x[u1_idx], mean_y[u1_idx]], dtype=np.float64) - origin
                    dist_u1_to_x = abs(float(np.dot(r_u1, u2_vector)))
                    
                    # Distance of vertical line midpoint to Y-axis (measured along U1)
                    r_u2 = np.array([mean_x[u2_idx], mean_y[u2_idx]], dtype=np.float64) - origin
                    dist_u2_to_y = abs(float(np.dot(r_u2, u1_vector)))

                    # ── Term 1: Primary score — Edge gradient magnitude ──
                    score = 0.0
                    if gradient_scores is not None and len(gradient_scores) >= 2:
                        grad_u1 = gradient_scores[u1_idx] if u1_idx < len(gradient_scores) else 0.0
                        grad_u2 = gradient_scores[u2_idx] if u2_idx < len(gradient_scores) else 0.0
                        mean_gradient = (grad_u1 + grad_u2) / 2.0
                        score += mean_gradient * gradient_weight
                    
                    # ── Term 2: Distance penalty — penalize lines far from corner origin ──
                    dist_penalty = -(dist_u1_to_x / H) * 2.0 - (dist_u2_to_y / W) * 2.0
                    score += dist_penalty
                    
                    # ── Term 3: Angle deviation penalty — smooth penalty from alpha_prior ──
                    angle_deviation = abs(diff_deg - alpha_prior)
                    if angle_deviation > angle_tolerance:
                        normalized_deviation = angle_deviation / 45.0
                        angle_penalty = -min(normalized_deviation * 1.5, 3.0)
                        score += angle_penalty
                    
                    # ── Term 4: Alignment bonus — small boost for lines parallel to U1 and U2 ──
                    alignment_bonus = (dot_u1[u1_idx] + dot_u2[u2_idx]) * 0.5
                    score += alignment_bonus

                    if score > best_score:
                        best_score = score
                        best_h_line = lines_2d[u1_idx].tolist()
                        best_v_line = lines_2d[u2_idx].tolist()
                        if return_scores and gradient_scores is not None:
                            best_h_score = gradient_scores[u1_idx] if u1_idx < len(gradient_scores) else 0.0
                            best_v_score = gradient_scores[u2_idx] if u2_idx < len(gradient_scores) else 0.0

        # ── Minimum physical length validation (at least 1.5 cm) ──
        if best_h_line is not None:
            len_h = np.hypot(best_h_line[2] - best_h_line[0], best_h_line[3] - best_h_line[1]) / PX_PER_CM_X
            if len_h < 1.5:
                best_h_line = None
                
        if best_v_line is not None:
            len_v = np.hypot(best_v_line[2] - best_v_line[0], best_v_line[3] - best_v_line[1]) / PX_PER_CM_Y
            if len_v < 1.5:
                best_v_line = None
                
        if best_h_line is None or best_v_line is None:
            if return_scores:
                return None, None, 0.0, 0.0
            return None, None
            
        if return_scores:
            return best_h_line, best_v_line, best_h_score, best_v_score
        return best_h_line, best_v_line

    @staticmethod
    def _calculate_control_signals(
        x_basis: float,
        y_basis: float,
        x_1: Union[List[float], np.ndarray],
        y_2: Union[List[float], np.ndarray],
        pivot: Optional[Tuple[float, float]] = DEFAULT_PIVOT,
        rot_alignment: str = 'horizontal',
        compensate_rotation: bool = True,
        max_rot_deg: float = 10.0,
        rot_deadband_deg: float = ROTATION_DEADBAND_DEG,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Calculate Cartesian translation vector and rotation angles from the fabric corner.

        Computes the 2D intersection point (I1, I2) of the horizontal line (x_1) and
        vertical line (y_2) using Cramer's determinant rule.

        When `compensate_rotation=True` and `pivot` is provided:
        Accounts for the kinematic coupling between hand rotation and corner translation.
        Because the robot hand clamps and rotates the fabric around its pivot point P,
        applying rotation angle theta causes the corner I to swing along an arc around P.
        To ensure the corner reaches the target basis (x_basis, y_basis) after simultaneous
        rotation and translation, the translation vector is modified based on the predicted
        post-rotation corner position I_rot.

        Args:
            x_basis: float
                Target corner origin X coordinate in pixels.
            y_basis: float
                Target corner origin Y coordinate in pixels.
            x_1: Union[List[float], np.ndarray] (shape: [4,])
                Coordinates [x1, y1, x2, y2] of the horizontal fabric edge.
            y_2: Union[List[float], np.ndarray] (shape: [4,])
                Coordinates [x1, y1, x2, y2] of the vertical fabric edge.
            pivot: Optional[Tuple[float, float]]
                Coordinates (px, py) of the robot hand center of rotation in image pixels.
                Defaults to DEFAULT_PIVOT (385.0, 83.0).
            rot_alignment: str
                Which rotation angle governs the rotation: 'horizontal' (angle_r1),
                'vertical' (angle_r2), or 'mean'. Defaults to 'horizontal'.
            compensate_rotation: bool
                Whether to adjust the translation vector to compensate for rotation swing.
                Defaults to True.
            max_rot_deg: float
                Maximum rotation angle limit in degrees (clamped to hardware limits).
                Defaults to 10.0.
            rot_deadband_deg: float
                Rotation deadband in degrees. Displacements below this are treated as zero rotation.
                Defaults to ROTATION_DEADBAND_DEG (0.1).

        Returns:
            Tuple[Tuple[float, float], Tuple[float, float]]:
                translation_vector: Tuple[float, float]
                    (tx, ty) modified displacement in pixels along U1_VECTOR (horizontal) and
                    U2_VECTOR (vertical) compensating for rotation.
                rotation_angles: Tuple[float, float]
                    (angle_r1, angle_r2) signed angular errors in degrees for the
                    horizontal and vertical edges relative to U1 and U2.
        """
        x1_a, y1_a, x2_a, y2_a = x_1
        x1_b, y1_b, x2_b, y2_b = y_2

        # ── 1. Calculate 2D intersection (I1, I2) via Cramer's rule determinant ──
        denom = (x1_a - x2_a) * (y1_b - y2_b) - (y1_a - y2_a) * (x1_b - x2_b)
        if abs(denom) < 1e-5:
            # Fallback for parallel or collinear segments
            I1, I2 = x_basis, y_basis
        else:
            I1 = ((x1_a*y2_a - y1_a*x2_a)*(x1_b - x2_b) - (x1_a - x2_a)*(x1_b*y2_b - y1_b*x2_b)) / denom
            I2 = ((x1_a*y2_a - y1_a*x2_a)*(y1_b - y2_b) - (y1_a - y2_a)*(x1_b*y2_b - y1_b*x2_b)) / denom

        u1 = U1_VECTOR
        u2 = U2_VECTOR

        v_x1 = np.array([x2_a - x1_a, y2_a - y1_a], dtype=np.float64)
        v_y2 = np.array([x2_b - x1_b, y2_b - y1_b], dtype=np.float64)

        # ── 2. Compute signed rotation angle wrapped to [-90°, +90°] ──
        def signed_angle_between_deg(v, u):
            theta_v = np.arctan2(v[1], v[0])
            theta_u = np.arctan2(u[1], u[0])
            angle = theta_v - theta_u
            angle = (angle + np.pi) % (2 * np.pi) - np.pi
            if angle > np.pi / 2:
                angle -= np.pi
            elif angle < -np.pi / 2:
                angle += np.pi
            return math.degrees(angle)

        angle_r1 = -signed_angle_between_deg(v_x1, u1)
        angle_r2 = -signed_angle_between_deg(v_y2, u2)

        # ── 3. Modify translation vector to adapt to rotation around pivot ──
        if compensate_rotation and pivot is not None:
            if rot_alignment == 'vertical':
                rot_deg = angle_r2
            elif rot_alignment == 'mean':
                rot_deg = (angle_r1 + angle_r2) / 2.0
            else:
                rot_deg = angle_r1

            # Enforce deadband and physical bounds matching actuator
            if abs(rot_deg) < rot_deadband_deg:
                effective_rot_deg = 0.0
            else:
                effective_rot_deg = max(-max_rot_deg, min(max_rot_deg, rot_deg))

            if effective_rot_deg != 0.0:
                theta_rad = math.radians(effective_rot_deg)
                cos_t = math.cos(theta_rad)
                sin_t = math.sin(theta_rad)

                p_vec = np.array(pivot, dtype=np.float64)
                corner_vec = np.array([I1, I2], dtype=np.float64)
                r_vec = corner_vec - p_vec

                # 2D rotation matrix: R(theta) * r
                rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
                corner_rot = p_vec + rot_mat @ r_vec
                I1_rot, I2_rot = float(corner_rot[0]), float(corner_rot[1])
            else:
                I1_rot, I2_rot = I1, I2
        else:
            I1_rot, I2_rot = I1, I2

        # ── 4. Project corner error vector onto physical basis axes U1 and U2 ──
        v = np.array([I1_rot - x_basis, I2_rot - y_basis], dtype=np.float64)
        tx = float(np.dot(v, U1_VECTOR))
        ty = float(np.dot(v, U2_VECTOR))
        translation_vector = (tx, ty)

        return translation_vector, (angle_r1, angle_r2)

    # ------------------------------------------------------------------
    # Generator
    # ------------------------------------------------------------------

    def compute_single_signal(self, cam, target_frames: int = 150, visualize: bool = False):
        """
        Captures a burst of video frames, processes them in batches, tracks and stabilizes
        horizontal and vertical fabric edge lines across time, and calculates the single
        stabilized control signal for closed-loop adjustment.

        Parameters:
            cam: Union[Camera, FastCamera, cv.VideoCapture]
                Active camera stream object providing `.read()` returning `(ret, frame, has_new)`
                and optionally `.clear_buffer()`.
            target_frames: int
                Total number of valid frames to accumulate in the burst for temporal line stabilization
                (default: 150).
            visualize: bool
                Whether to render real-time detection overlays, bounding boxes, and side-by-side
                ROI debugging displays using OpenCV.

        Returns:
            Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
                Stabilized control signals `((tx, ty), (angle_r1, angle_r2))` where:
                    tx, ty: float
                        Translation displacement vector projected along the U1 and U2 axes in image pixels.
                    angle_r1, angle_r2: float
                        Rotational error angles of horizontal and vertical fabric edges relative to
                        the target coordinate axes in degrees.
                Returns None if camera capture fails or if valid corner edge pairs cannot be resolved.
        """
        if cam is None:
            return None

        # ── Step 1: Flush hardware frame queue ────────────────────────────
        # Clean the queue so that stale frames captured before the physical adjustment trigger
        # are discarded, ensuring the burst only analyzes the current fabric position.
        cam.clear_buffer() if hasattr(cam, 'clear_buffer') else None

        stabilizer = LineStabilizer(window=target_frames)
        last_best_h_line, last_best_v_line = None, None
        last_raw_lines = None
        last_roi_img = None
        last_roi_lines = None
        frames_processed = 0
        frame_buffer = []
        last_frame = None

        # ── Step 2: Accumulate frames in batches up to target_frames ──────
        while frames_processed < target_frames:
            ret, frame, has_new = cam.read()
            if not ret or frame is None:
                return None
            if not has_new:
                time.sleep(0.005)
                continue
            frame_buffer.append(frame)
            last_frame = frame

            # Process whenever a full batch is accumulated, or on the final trailing frames
            if len(frame_buffer) >= self.batch_size or (frames_processed + len(frame_buffer) == target_frames):
                grays, imgs, offsets = preprocessing_adjustor_batch(
                    frame_buffer, img_size=self.img_size, crop_roi=self.crop_roi,
                    enhance_contrast=self.enhance_contrast, smooth_texture=self.smooth_texture
                )

                if self.detector_mode == DetectorMode.ENSEMBLE:
                    lines_batch = []
                    for gray in grays:
                        lines, _, _ = ensemble_detect_lines(gray)
                        lines_batch.append(lines)
                else:
                    lines_batch = detect_lines_batch(grays, self.detector_mode, imgs)
                    lines_batch = [postprocess_lines(ls) if self.detector_mode != DetectorMode.MLSD else ls for ls in lines_batch]

                # ── Step 3: Process each frame in chunk & update temporal stabilizer ──
                for i in range(len(grays)):
                    gray, img, lines, offset = grays[i], imgs[i], lines_batch[i], offsets[i]
                    if self.detector_mode != DetectorMode.ENSEMBLE:
                        # Filter mechanical fixtures (e.g. presser foot, needle plate holes)
                        if self.detector_mode == DetectorMode.MLSD and self.noisy_areas:
                            lines = filter_noisy_lines_by_area(lines, self.noisy_areas, offset=offset)

                        # Precompute Sobel gradient map in ROI space to evaluate physical edge strength
                        grad_scores_roi = None
                        if lines is not None and len(lines) > 0:
                            self._edge_scorer.precompute(gray)
                            grad_scores_roi = self._edge_scorer.score_lines(lines.reshape(-1, 4))

                        # Group candidate lines by dominant orientation modes and prune outliers
                        lines = filter_dominant_angles(lines, gradient_scores=grad_scores_roi, length_weighting=self.length_weighting, offset=offset)

                        # Re-score surviving lines since candidate index order changed
                        grad_scores_roi = None
                        if lines is not None and len(lines) > 0:
                            grad_scores_roi = self._edge_scorer.score_lines(lines.reshape(-1, 4))
                    
                    if i == len(grays) - 1:
                        last_roi_img = img.copy()
                        last_roi_lines = np.copy(lines) if lines is not None else None

                    # Rescale lines from local ROI bounding box coordinates back to full sensor frame
                    if lines is not None and len(lines) > 0:
                        lines[:, 0] = lines[:, 0] * offset[2] + offset[0]
                        lines[:, 1] = lines[:, 1] * offset[3] + offset[1]
                        lines[:, 2] = lines[:, 2] * offset[2] + offset[0]
                        lines[:, 3] = lines[:, 3] * offset[3] + offset[1]
                    if i == len(grays) - 1:
                        last_raw_lines = np.copy(lines) if lines is not None and len(lines) > 0 else None
                        
                    # Find optimal orthogonal/non-orthogonal corner edge pair for this individual frame
                    best_h_line, best_v_line, h_score, v_score = self._select_best_lines(
                        lines, frame_buffer[i].shape, self.x_basis, self.y_basis, U1_VECTOR, U2_VECTOR,
                        alpha_prior=self.alpha_prior, angle_tolerance=self.angle_tolerance,
                        gradient_weight=self.gradient_weight, gradient_scores=grad_scores_roi,
                        return_scores=True, line_selection_mode=self.line_selection_mode
                    )                    
                    # Update temporal Hough-clustering stabilizer with current frame's candidates
                    stab_h_line, stab_v_line = stabilizer.update(best_h_line, best_v_line, h_score, v_score)
                    if i == len(grays) - 1:
                        last_best_h_line = stab_h_line
                        last_best_v_line = stab_v_line
                
                frames_processed += len(frame_buffer)
                frame_buffer.clear()

        # ── Step 4: Render debug visualization if requested ───────────────
        if visualize and last_frame is not None:
            display_frame = last_frame.copy()
            if last_raw_lines is not None:
                for line in last_raw_lines:
                    cv.line(display_frame, (int(line[0]), int(line[1])),
                            (int(line[2]), int(line[3])), (0, 255, 0), 2)
            
            if self.noisy_areas:
                for area in self.noisy_areas:
                    x_c, y_c, h_c, w_c = area[:4]
                    x_c_px = int(x_c * PX_PER_CM_X)
                    y_c_px = int(y_c * PX_PER_CM_Y)
                    h_px = int(h_c * PX_PER_CM_Y)
                    w_px = int(w_c * PX_PER_CM_X)
                    x1 = int(x_c_px - w_px / 2)
                    y1 = int(y_c_px - h_px / 2)
                    x2 = int(x_c_px + w_px / 2)
                    y2 = int(y_c_px + h_px / 2)
                    cv.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if last_best_h_line is not None:
                cv.line(display_frame, (int(last_best_h_line[0]), int(last_best_h_line[1])),
                        (int(last_best_h_line[2]), int(last_best_h_line[3])), (0, 255, 255), 2)
            if last_best_v_line is not None:
                cv.line(display_frame, (int(last_best_v_line[0]), int(last_best_v_line[1])),
                        (int(last_best_v_line[2]), int(last_best_v_line[3])), (255, 0, 255), 2)
            draw_adjustor_overlay(display_frame, (self.x_basis, self.y_basis), AXIS_LENGTH_PX)
            
            # Reconcile height mismatches between ROI and full frame with black borders before hconcat
            if last_roi_img is not None:
                roi_vis = last_roi_img.copy()
                if last_roi_lines is not None:
                    for line in last_roi_lines:
                        cv.line(roi_vis, (int(line[0]), int(line[1])),
                                (int(line[2]), int(line[3])), (0, 255, 0), 2)
                                
                if self.noisy_areas and 'offset' in locals() and offset is not None:
                    roi_areas = map_noisy_areas_to_roi(self.noisy_areas, offset)
                    for area in roi_areas:
                        x_c, y_c, h_c, w_c = area[:4]
                        x_c_px = int(x_c * PX_PER_CM_X)
                        y_c_px = int(y_c * PX_PER_CM_Y)
                        h_px = int(h_c * PX_PER_CM_Y)
                        w_px = int(w_c * PX_PER_CM_X)
                        x1 = int(x_c_px - w_px / 2)
                        y1 = int(y_c_px - h_px / 2)
                        x2 = int(x_c_px + w_px / 2)
                        y2 = int(y_c_px + h_px / 2)
                        cv.rectangle(roi_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                
                h_full, w_full = display_frame.shape[:2]
                h_roi, w_roi = roi_vis.shape[:2]
                
                if h_roi < h_full:
                    pad_bottom = h_full - h_roi
                    roi_vis_padded = cv.copyMakeBorder(roi_vis, 0, pad_bottom, 0, 0, cv.BORDER_CONSTANT, value=[0,0,0])
                    combined_vis = cv.hconcat([display_frame, roi_vis_padded])
                elif h_roi > h_full:
                    pad_bottom = h_roi - h_full
                    display_frame_padded = cv.copyMakeBorder(display_frame, 0, pad_bottom, 0, 0, cv.BORDER_CONSTANT, value=[0,0,0])
                    combined_vis = cv.hconcat([display_frame_padded, roi_vis])
                else:
                    combined_vis = cv.hconcat([display_frame, roi_vis])
                    
                cv.imshow("Line Detection", combined_vis)
            else:
                cv.imshow("Line Detection", display_frame)
                
            cv.waitKey(1)

        # ── Step 5: Calculate final control signal from stabilized corner lines ──
        if last_best_h_line is not None and last_best_v_line is not None:
            return self._calculate_control_signals(
                self.x_basis, self.y_basis, last_best_h_line, last_best_v_line,
                pivot=self.pivot,
                rot_alignment=self.rot_alignment,
                compensate_rotation=self.compensate_rotation,
                max_rot_deg=self.max_rot_deg,
            )
            
        return None

    def run(self, cam=None, target_frames: int = 150, visualize: bool = False, debug: bool = False):
        """
        Continuous generator function streaming control signals or debug visualizations from camera.

        In debug mode, runs an interactive window showing raw line detections. In production mode,
        accumulates batches of frames, performs temporal line stabilization, enforces cooldown periods
        between physical adjustments, suppresses micro-movements within deadbands, and yields control signals.

        Parameters:
            cam: Optional[Union[Camera, FastCamera, cv.VideoCapture]]
                Camera stream object. If None, automatically acquires camera via `get_camera()`.
            target_frames: int
                Rolling window size for the temporal `LineStabilizer` (default: 150 frames).
            visualize: bool
                If True, renders real-time tracking overlays and logs annotated frames to disk.
            debug: bool
                If True, displays raw detections interactively without yielding motor control signals.

        Yields:
            Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
                Stabilized control signals `((tx, ty), (ah, av))` where:
                    tx, ty: float
                        Translation displacement along U1 and U2 in image pixels.
                    ah, av: float
                        Rotational error angles of horizontal and vertical edges relative to axes in degrees.
                Yields None during cooldown periods, buffer accumulation, deadband suppression,
                or when valid corner edges are undetected.
        """
        if cam is None:
            cam = get_camera()
        if cam is None:
            return

        if debug:
            # ── Debug Mode: interactive raw line detection inspection without motor signals ──
            time.sleep(0.05)
            cv.namedWindow("Line Detection", cv.WINDOW_NORMAL)
            while True:
                ret, frame, has_new = cam.read()
                if not ret or frame is None:
                    break
                if not has_new:
                    time.sleep(0.005)
                    continue

                gray, img, lines, _, _, per_det = self._perform_line_detection(
                    frame, self.img_size, self.enhance_contrast, self.smooth_texture,
                    self.detector_mode, self.mask_thresh,
                    noisy_areas=self.noisy_areas,
                    alpha_prior=self.alpha_prior,
                    angle_tolerance=self.angle_tolerance,
                    gradient_weight=self.gradient_weight,
                    length_weighting=self.length_weighting,
                    edge_scorer=self._edge_scorer,
                    line_selection_mode=self.line_selection_mode,
                )

                drawn_img = img.copy()
                if self.detector_mode == DetectorMode.ENSEMBLE:
                    for colour, det_lines in zip(_ENSEMBLE_COLOURS, per_det):
                        drawn_img = _draw_lines_colour(drawn_img, det_lines, colour, 1)
                    drawn_img = _draw_lines_colour(drawn_img, lines, _FUSED_COLOUR, 2)
                else:
                    if lines is not None:
                        drawn_img = draw_lines(drawn_img, lines, self.detector_mode)
                cv.imshow("Line Detection", drawn_img)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

            cam.release()
            cv.destroyAllWindows()
            return

        # ── Signal Streaming Mode: temporal stabilization & motor signal generation ──
        time.sleep(4)

        ret, init_frame, _ = cam.read()
        if ret and init_frame is not None:
            _, _, _ = preprocessing_adjustor(
                init_frame,
                img_size=self.img_size,
                crop_roi=self.crop_roi,
                enhance_contrast=self.enhance_contrast,
                smooth_texture=self.smooth_texture,
            )

        control_signals = ((0.0, 0.0), (0.0, 0.0))
        stabilizer = LineStabilizer(window=target_frames)
        cooldown_end_time = 0.0

        session_capture_dir = None
        frame_count = 0
        if visualize:
            base_capture_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "captured_frames"
            )
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            session_capture_dir = os.path.join(base_capture_dir, current_time)
            os.makedirs(session_capture_dir, exist_ok=True)
            cv.namedWindow("Line Detection", cv.WINDOW_NORMAL)
            cv.namedWindow("Gray (Left) vs Line Detection (Right)", cv.WINDOW_NORMAL)
        last_yielded_tx = None
        last_yielded_ty = None

        frame_buffer = []
        
        while True:
            ret, frame, has_new = cam.read()
            if not ret or frame is None:
                break

            if not has_new:
                time.sleep(0.005) if not visualize else None
                if visualize:
                    cv.waitKey(1)
                yield None
                continue

            # ── Cooldown Check: suppress signal generation while motors are settling ──
            if time.time() < cooldown_end_time:
                if visualize:
                    display_frame = frame.copy()
                    draw_adjustor_overlay(display_frame, (self.x_basis, self.y_basis), AXIS_LENGTH_PX)
                    cv.putText(display_frame, "COOLDOWN", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv.imshow("Line Detection", display_frame)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break
                # Reset stabilizer history so past movement does not corrupt next stabilization cycle
                stabilizer.reset()
                yield None
                continue

            frame_buffer.append(frame)
            
            yielded_signal = None
            # ── Batch Preprocessing & Line Detection ────────────────────────
            if len(frame_buffer) >= self.batch_size:
                grays, imgs, offsets = preprocessing_adjustor_batch(
                    frame_buffer, img_size=self.img_size, crop_roi=self.crop_roi,
                    enhance_contrast=self.enhance_contrast, smooth_texture=self.smooth_texture
                )
                
                if self.detector_mode == DetectorMode.ENSEMBLE:
                    lines_batch = []
                    for gray in grays:
                        lines, _, _ = ensemble_detect_lines(gray)
                        lines_batch.append(lines)
                else:
                    lines_batch = detect_lines_batch(grays, self.detector_mode, imgs)
                    lines_batch = [postprocess_lines(ls) if self.detector_mode != DetectorMode.MLSD else ls for ls in lines_batch]
                        
                # ── Sequential Frame Evaluation & Stabilizer Ingestion ──────────
                last_best_h_line, last_best_v_line = None, None
                for i in range(len(grays)):
                    gray, img, lines, offset = grays[i], imgs[i], lines_batch[i], offsets[i]
                    if self.detector_mode != DetectorMode.ENSEMBLE:
                        # Exclude mechanical fixtures
                        if self.detector_mode == DetectorMode.MLSD and self.noisy_areas:
                            lines = filter_noisy_lines_by_area(lines, self.noisy_areas, offset=offset)

                        # Evaluate Sobel gradient magnitudes under candidate lines in local ROI
                        grad_scores_roi = None
                        if lines is not None and len(lines) > 0:
                            self._edge_scorer.precompute(gray)
                            grad_scores_roi = self._edge_scorer.score_lines(lines.reshape(-1, 4))

                        # Retain dominant angle clusters and re-score survivors
                        lines = filter_dominant_angles(lines, gradient_scores=grad_scores_roi, length_weighting=self.length_weighting, offset=offset)

                        grad_scores_roi = None
                        if lines is not None and len(lines) > 0:
                            grad_scores_roi = self._edge_scorer.score_lines(lines.reshape(-1, 4))
                    
                    # Remap line endpoints from local ROI coordinates to full frame sensor space
                    if lines is not None and len(lines) > 0:
                        lines[:, 0] = lines[:, 0] * offset[2] + offset[0]
                        lines[:, 1] = lines[:, 1] * offset[3] + offset[1]
                        lines[:, 2] = lines[:, 2] * offset[2] + offset[0]
                        lines[:, 3] = lines[:, 3] * offset[3] + offset[1]
                        
                    lines_batch[i] = lines
                        
                    # Find candidate corner pair for frame i
                    best_h_line, best_v_line, h_score, v_score = self._select_best_lines(
                        lines, frame_buffer[i].shape, self.x_basis, self.y_basis, U1_VECTOR, U2_VECTOR,
                        alpha_prior=self.alpha_prior, angle_tolerance=self.angle_tolerance,
                        gradient_weight=self.gradient_weight, gradient_scores=grad_scores_roi,
                        return_scores=True, line_selection_mode=self.line_selection_mode
                    )                    
                    # Sequentially push frame detections to temporal stabilizer
                    stab_h_line, stab_v_line = stabilizer.update(best_h_line, best_v_line, h_score, v_score)
                    if i == len(grays) - 1:
                        last_best_h_line = stab_h_line
                        last_best_v_line = stab_v_line
                        
                # ── Control Signal Computation & Deadband Filtering ────────────
                if last_best_h_line is not None and last_best_v_line is not None:
                    control_signals = self._calculate_control_signals(
                        self.x_basis, self.y_basis, last_best_h_line, last_best_v_line,
                        pivot=self.pivot,
                        rot_alignment=self.rot_alignment,
                        compensate_rotation=self.compensate_rotation,
                        max_rot_deg=self.max_rot_deg,
                    )
                    
                    (tx_px, ty_px), _ = control_signals
                    tx_cm = tx_px / PX_PER_CM_X
                    ty_cm = ty_px / PX_PER_CM_Y
                    
                    # Deadband check: avoid small erratic movements if position hasn't meaningfully changed
                    if last_yielded_tx is not None and last_yielded_ty is not None:
                        if abs(tx_cm - last_yielded_tx) < TRANSLATION_DEADBAND_CM and abs(ty_cm - last_yielded_ty) < TRANSLATION_DEADBAND_CM:
                            control_signals = None
                    
                    if control_signals is not None:
                        last_yielded_tx = tx_cm
                        last_yielded_ty = ty_cm
                        cooldown_end_time = time.time() + self.cooldown_time
                        
                    yielded_signal = control_signals

                if visualize:
                    # Visualise the final frame of the batch
                    visualize_and_save_frame_adjustor(
                        grays[-1], frame_buffer[-1], lines_batch[-1], last_best_h_line, last_best_v_line, self.detector_mode,
                        control_signals, self.x_basis, self.y_basis,
                        session_capture_dir, frame_count,
                    )
                    
                frame_count += len(frame_buffer)
                frame_buffer.clear()
                
                if visualize and cv.waitKey(1) & 0xFF == ord('q'):
                    break

            yield yielded_signal

        cam.release()
        cv.destroyAllWindows()


class NonOrthoLineAdjustor(LineDetectionAdjustor):
    """
    Subclass of LineDetectionAdjustor that handles arbitrary non-orthogonal fabric edges.
    It extracts a prior state (alpha angle) from a perfect 'prior' image, and uses it 
    to filter and track lines.
    """

    def __init__(self, prior_img_path: Optional[str] = None, angle_tolerance: float = 5.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha_prior = None
        self.prior_h_line = None
        self.prior_v_line = None
        self.angle_tolerance = angle_tolerance
        if prior_img_path is not None:
            self.extract_prior(prior_img_path=prior_img_path)

    def extract_prior(self, prior_img_path: Optional[str] = None, frame: Optional[np.ndarray] = None):
        """
        Extracts the baseline corner angle `alpha_prior` and reference edges from a calibrated prior template.

        The prior image assumes the fabric corner is placed at `(x_basis, y_basis)` with one of its
        edges aligned with the Oy axis (`U2_VECTOR`). The method inspects either a pre-saved JSON
        metadata file or analyzes the prior image by detecting line pairs, computing their intersection,
        and finding the pair that best aligns with the reference coordinate axes.

        Parameters:
            prior_img_path: Optional[str]
                Filepath to the prior template image (`.png`) or precomputed metadata (`.json`).
                Automatically falls back to searching inside the `mask_priors/` directory.
            frame: Optional[np.ndarray]
                Pre-loaded BGR image array (dtype: uint8, shape: [H, W, 3]) used if `prior_img_path` is None.

        Returns:
            None:
                Sets `self.alpha_prior` (float, degrees), `self.prior_h_line` (List[float]),
                and `self.prior_v_line` (List[float]) in-place.

        Raises:
            ValueError: If neither a valid image nor JSON metadata is provided, or if no valid
                        intersecting corner lines are detected near the basis origin.
        """
        # ── Step 1: Check for precomputed JSON metadata ───────────────────
        if prior_img_path is not None:
            base_path = prior_img_path
            if base_path.endswith('.png'):
                base_path = base_path[:-4]
            elif base_path.endswith('.json'):
                base_path = base_path[:-5]
                
            json_path = base_path + '.json'
            png_path = base_path + '.png'
            
            # Auto-fallback to mask_priors directory if not found in current directory
            if not os.path.exists(json_path) and not os.path.exists(png_path):
                alt_json = os.path.join("mask_priors", json_path)
                alt_png = os.path.join("mask_priors", png_path)
                if os.path.exists(alt_json) or os.path.exists(alt_png):
                    json_path = alt_json
                    png_path = alt_png
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                self.alpha_prior = data.get('alpha_prior')
                if self.alpha_prior is None:
                    raise ValueError(f"JSON file {json_path} does not contain 'alpha_prior'")
                print(f"[Prior Extraction] Loaded prior angle from JSON: {self.alpha_prior:.2f}°")
                return
            else:
                prior_img_path = png_path

        # ── Step 2: Load and preprocess image template ────────────────────
        if frame is None:
            if prior_img_path is None:
                raise ValueError("Must provide either prior_img_path or frame")
            frame = cv.imread(prior_img_path)
            if frame is None:
                raise ValueError(f"Could not read prior image at {prior_img_path}")

        gray, img, offset = preprocessing_adjustor(
            frame,
            img_size=self.img_size,
            crop_roi=self.crop_roi,
            enhance_contrast=self.enhance_contrast,
            smooth_texture=self.smooth_texture,
        )

        # ── Step 3: Detect candidate line segments ────────────────────────
        if self.detector_mode == DetectorMode.ENSEMBLE:
            lines, _, _ = ensemble_detect_lines(gray)
        else:
            lines = detect_lines(gray, self.detector_mode, img)
            if self.detector_mode != DetectorMode.MLSD:
                lines = postprocess_lines(lines)
            if self.detector_mode == DetectorMode.MLSD and hasattr(self, 'noisy_areas') and self.noisy_areas:
                lines = filter_noisy_lines_by_area(lines, self.noisy_areas, offset=offset)
            lines = filter_dominant_angles(lines, offset=offset)

        if lines is None or len(lines) == 0:
            raise ValueError("No lines detected in prior image!")
            
        if lines is not None and len(lines) > 0:
            lines[:, 0] = lines[:, 0] * offset[2] + offset[0]
            lines[:, 1] = lines[:, 1] * offset[3] + offset[1]
            lines[:, 2] = lines[:, 2] * offset[2] + offset[0]
            lines[:, 3] = lines[:, 3] * offset[3] + offset[1]

        # ── Step 4: Compute line directional unit vectors & Oy alignment ──
        lines_2d = lines.astype(np.float32).reshape(-1, 4)
        dx = lines_2d[:, 2] - lines_2d[:, 0]
        dy = lines_2d[:, 3] - lines_2d[:, 1]
        angles = np.arctan2(dy, dx)
        lengths = np.hypot(dx, dy)
        lengths[lengths == 0] = 1e-6
        dir_x = dx / lengths
        dir_y = dy / lengths
        
        # Absolute dot product measuring parallelism to the Oy axis (U2_VECTOR)
        lines_dir = np.column_stack([dir_x, dir_y])
        dot_u2 = np.abs(np.dot(lines_dir, U2_VECTOR))

        best_score = -float('inf')
        n_lines = len(lines_2d)
        origin = np.array([self.x_basis, self.y_basis], dtype=np.float64)

        # ── Step 5: Pairwise line evaluation to locate the corner ─────────
        for i in range(n_lines):
            for j in range(i + 1, n_lines):
                x1, y1, x2, y2 = lines_2d[i]
                x3, y3, x4, y4 = lines_2d[j]
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if denom == 0:
                    continue
                
                # Intersection point via Cramer's determinant
                ix = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
                iy = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
                
                dist_to_origin = np.hypot(ix - origin[0], iy - origin[1])
                if dist_to_origin > 300.0:
                    continue
                
                # Check alignment of the most vertical-ish line with U2
                u2_alignment = max(dot_u2[i], dot_u2[j])
                
                # Score combines vertical axis alignment, closeness to basis origin, and segment lengths
                score = (u2_alignment * 100.0) - (dist_to_origin * 2.0) + ((lengths[i] + lengths[j]) * 0.1)
                
                if score > best_score:
                    best_score = score
                    angle_diff = angles[i] - angles[j]
                    
                    # Store canonical angular difference in [0, 90] degrees
                    diff_deg = abs(np.degrees(angle_diff)) % 180.0
                    if diff_deg > 90.0:
                        diff_deg = 180.0 - diff_deg
                    self.alpha_prior = diff_deg
                    
                    if dot_u2[i] > dot_u2[j]:
                        self.prior_v_line = lines_2d[i].tolist()
                        self.prior_h_line = lines_2d[j].tolist()
                    else:
                        self.prior_v_line = lines_2d[j].tolist()
                        self.prior_h_line = lines_2d[i].tolist()

        if self.alpha_prior is None:
            raise ValueError("Could not find a valid intersecting pair in the prior image near the origin!")
            
        print(f"[Prior Extraction] Successfully extracted prior angle: {self.alpha_prior:.2f}°")


    @staticmethod
    def _calculate_control_signals(
        x_basis: float,
        y_basis: float,
        x_1,
        y_2,
        pivot: Optional[Tuple[float, float]] = DEFAULT_PIVOT,
        rot_alignment: str = 'vertical',
        compensate_rotation: bool = True,
        max_rot_deg: float = 10.0,
        rot_deadband_deg: float = ROTATION_DEADBAND_DEG,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Calculate control signals for non-orthogonal fabric pieces with rotation compensation.

        Determines the intersection point of two fabric edges, projects the translation error
        onto the U1 and U2 axes (adapting the translation vector to compensate for rotation around
        the hand pivot), and calculates the signed angular deviation of the vertical edge
        relative to U2_VECTOR.

        Parameters:
            x_basis: float
                Target corner X coordinate in full-frame image pixels.
            y_basis: float
                Target corner Y coordinate in full-frame image pixels.
            x_1: Union[List[float], Tuple[float, ...], np.ndarray]
                Coordinates [x1, y1, x2, y2] of the first fabric edge segment (horizontal candidate).
            y_2: Union[List[float], Tuple[float, ...], np.ndarray]
                Coordinates [x1, y1, x2, y2] of the second fabric edge segment (vertical candidate).
            pivot: Optional[Tuple[float, float]]
                Coordinates (px, py) of the robot hand center of rotation in image pixels.
                Defaults to DEFAULT_PIVOT.
            rot_alignment: str
                Ignored for NonOrtho (vertical edge relative to U2 always used). Defaults to 'vertical'.
            compensate_rotation: bool
                Whether to adjust the translation vector to compensate for rotation swing.
                Defaults to True.
            max_rot_deg: float
                Maximum rotation angle limit in degrees. Defaults to 10.0.
            rot_deadband_deg: float
                Rotation deadband in degrees. Defaults to ROTATION_DEADBAND_DEG (0.1).

        Returns:
            Tuple[Tuple[float, float], Tuple[float, float]]:
                translation_vector: (tx, ty)
                    Displacement error projected along the U1 and U2 axes in image pixels.
                rotation_angles: (0.0, angle_oy)
                    Rotation angles in degrees, where angle_oy is the signed angular error of the
                    vertical fabric edge relative to the Oy axis (U2_VECTOR).
        """
        x1_a, y1_a, x2_a, y2_a = x_1
        x1_b, y1_b, x2_b, y2_b = y_2

        # ── Step 1: Intersection of line segments via 2D determinant ───────
        denom = (x1_a - x2_a) * (y1_b - y2_b) - (y1_a - y2_a) * (x1_b - x2_b)
        if abs(denom) < 1e-5:
            I1, I2 = x_basis, y_basis
        else:
            I1 = ((x1_a*y2_a - y1_a*x2_a)*(x1_b - x2_b) - (x1_a - x2_a)*(x1_b*y2_b - y1_b*x2_b)) / denom
            I2 = ((x1_a*y2_a - y1_a*x2_a)*(y1_b - y2_b) - (y1_a - y2_a)*(x1_b*y2_b - y1_b*x2_b)) / denom

        # ── Step 2: Compute signed angle between vertical line and U2 ─────
        v_y2 = np.array([x2_b - x1_b, y2_b - y1_b], dtype=np.float64)

        def signed_angle_between_deg(v, u):
            theta_v = np.arctan2(v[1], v[0])
            theta_u = np.arctan2(u[1], u[0])
            angle = theta_v - theta_u
            angle = (angle + np.pi) % (2 * np.pi) - np.pi
            if angle > np.pi / 2:
                angle -= np.pi
            elif angle < -np.pi / 2:
                angle += np.pi
            return math.degrees(angle)

        angle_oy = -signed_angle_between_deg(v_y2, U2_VECTOR)

        # ── Step 3: Modify translation vector to adapt to rotation around pivot ──
        if compensate_rotation and pivot is not None:
            rot_deg = angle_oy
            if abs(rot_deg) < rot_deadband_deg:
                effective_rot_deg = 0.0
            else:
                effective_rot_deg = max(-max_rot_deg, min(max_rot_deg, rot_deg))

            if effective_rot_deg != 0.0:
                theta_rad = math.radians(effective_rot_deg)
                cos_t = math.cos(theta_rad)
                sin_t = math.sin(theta_rad)

                p_vec = np.array(pivot, dtype=np.float64)
                corner_vec = np.array([I1, I2], dtype=np.float64)
                r_vec = corner_vec - p_vec

                rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
                corner_rot = p_vec + rot_mat @ r_vec
                I1_rot, I2_rot = float(corner_rot[0]), float(corner_rot[1])
            else:
                I1_rot, I2_rot = I1, I2
        else:
            I1_rot, I2_rot = I1, I2

        # ── Step 4: Project displacement vector onto motor motion axes ─────
        v = np.array([I1_rot - x_basis, I2_rot - y_basis], dtype=np.float64)
        tx = float(np.dot(v, U1_VECTOR))
        ty = float(np.dot(v, U2_VECTOR))
        translation_vector = (tx, ty)

        return translation_vector, (0.0, angle_oy)

    @classmethod
    def _perform_line_detection_arbitrary(
        cls, adjustor_instance, frame, img_size, enhance_contrast, smooth_texture,
        detector_mode: DetectorMode, mask_thresh: int,
        x_basis: Optional[float] = None, y_basis: Optional[float] = None,
    ):
        """
        Instance-aware wrapper for evaluation scripts to use the loaded prior.
        """
        gray, img, offset = preprocessing_adjustor(
            frame,
            img_size=img_size,
            crop_roi=adjustor_instance.crop_roi if hasattr(adjustor_instance, 'crop_roi') else None,
            enhance_contrast=enhance_contrast,
            smooth_texture=smooth_texture,
        )

        per_det = None
        if detector_mode == DetectorMode.ENSEMBLE:
            lines, scores, per_det = ensemble_detect_lines(gray)
        else:
            lines = detect_lines(gray, detector_mode, img)
            if detector_mode != DetectorMode.MLSD:
                lines = postprocess_lines(lines)
            lines = keep_lines_near_background(lines, gray, mask_thresh=30, mode="adjustor")

            if detector_mode == DetectorMode.MLSD and hasattr(adjustor_instance, 'noisy_areas') and adjustor_instance.noisy_areas:
                lines = filter_noisy_lines_by_area(lines, adjustor_instance.noisy_areas, offset=offset)

            # Score in ROI space (lines are still in ROI coords before rescaling)
            grad_scores = None
            if lines is not None and len(lines) > 0:
                adjustor_instance._edge_scorer.precompute(gray)
                grad_scores = adjustor_instance._edge_scorer.score_lines(lines.reshape(-1, 4))

            lines = filter_dominant_angles(lines, gradient_scores=grad_scores, offset=offset)

            # Re-score survivors (index set changed after filtering)
            grad_scores = None
            if lines is not None and len(lines) > 0:
                grad_scores = adjustor_instance._edge_scorer.score_lines(lines.reshape(-1, 4))

            if lines is not None:
                lines = lines.astype(np.float32)
                
            if lines is not None and len(lines) > 0:
                lines[:, 0] = lines[:, 0] * offset[2] + offset[0]
                lines[:, 1] = lines[:, 1] * offset[3] + offset[1]
                lines[:, 2] = lines[:, 2] * offset[2] + offset[0]
                lines[:, 3] = lines[:, 3] * offset[3] + offset[1]

        best_h_line, best_v_line = adjustor_instance._select_best_lines(
            lines, frame.shape, x_basis, y_basis, U1_VECTOR, U2_VECTOR,
            alpha_prior=adjustor_instance.alpha_prior, angle_tolerance=adjustor_instance.angle_tolerance,
            gradient_weight=adjustor_instance.gradient_weight, gradient_scores=grad_scores,
            line_selection_mode=adjustor_instance.line_selection_mode
        )

        return gray, img, lines, best_h_line, best_v_line, per_det




# ---------------------------------------------------------------------------
# Module-level aliases (Option B): static methods accessible as free functions
# so evaluate_adjustor.py and other callers continue to work unchanged.
# ---------------------------------------------------------------------------

_perform_line_detection    = LineDetectionAdjustor._perform_line_detection
_calculate_control_signals = LineDetectionAdjustor._calculate_control_signals
