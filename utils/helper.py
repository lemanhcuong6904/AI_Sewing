from typing import Tuple, Optional, Any
from collections import defaultdict
import time
import os
import cv2 as cv
import numpy as np
from numba import njit
from utils.viz_adjustor import draw_adjustor_overlay, AXIS_LENGTH_PX

timings = defaultdict(float)
counts = defaultdict(int)

lsd = cv.createLineSegmentDetector(cv.LSD_REFINE_STD)
clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# Pre-computed sample points for _line_is_near_background()
_BG_SAMPLES = np.linspace(0, 1, 10)


def tic():
    return time.time()


def toc(name, start):
    global timings
    timings[name] += time.time() - start
    counts[name] += 1


def _get_mask(gray, threshold=210):
    if gray.dtype != np.uint8:
        gray = (gray * 255).astype(np.uint8)

    # print("gray min:", gray.min())
    # print("gray max:", gray.max())
    # print("unique:", np.unique(gray))
    
    mask = gray >= threshold if threshold > 200 else gray <= threshold

    # print("any(mask):", np.any(mask))
    # print("sum(mask):", np.sum(mask))

    return mask.astype(np.uint8)


def _line_length(line):
    x1, y1, x2, y2 = line
    return np.hypot(x2 - x1, y2 - y1)


def _endpoint_distance(l1, l2):
    pts1 = [(l1[0],l1[1]), (l1[2],l1[3])]
    pts2 = [(l2[0],l2[1]), (l2[2],l2[3])]

    min_dist = float("inf")

    for p1 in pts1:
        for p2 in pts2:
            d = np.hypot(p1[0]-p2[0], p1[1]-p2[1])
            min_dist = min(min_dist, d)

    return min_dist


@njit(cache=True)
def _line_is_near_background_sewer(line, mask, dist_thresh=4, samples=10):
    x1 = int(line[0])
    y1 = int(line[1])
    x2 = int(line[2])
    y2 = int(line[3])

    h = mask.shape[0]
    w = mask.shape[1]

    for i in range(samples):
        t = i / (samples - 1) if samples > 1 else 0.0

        x = int(x1 + t * (x2 - x1))
        y = int(y1 + t * (y2 - y1))

        for dx in range(1, dist_thresh + 1): # Only consider from the right side of the line
            for dy in range(1, dist_thresh + 1): # Only consider from the upper side of the line
                xx = x + dx
                yy = y - dy

                if 0 <= xx < w and 0 <= yy < h:
                    if mask[yy, xx]:
                        return True

    return False


@njit(cache=True)
def _line_is_near_background_adjustor(line, mask, dist_thresh=10, samples=30):
    x1 = int(line[0])
    y1 = int(line[1])
    x2 = int(line[2])
    y2 = int(line[3])

    h = mask.shape[0]
    w = mask.shape[1]
    
    bg_count = 0
    total_count = 0

    for i in range(samples):
        t = i / (samples - 1) if samples > 1 else 0.0

        x = int(x1 + t * (x2 - x1))
        y = int(y1 + t * (y2 - y1))

        for dx in range(1, dist_thresh + 1, 2):
            for dy in range(1, dist_thresh + 1, 2):
                xx = x + dx
                yy = y + dy

                if 0 <= xx < w and 0 <= yy < h:
                    if mask[yy, xx]:
                        bg_count += 1
                    total_count += 1

    if total_count == 0:
        return False
        
    bg_ratio = bg_count / total_count
    
    # A true edge will have a mix of background and fabric in its neighborhood.
    # If bg_ratio > 0.95, it is a noisy line purely in the background.
    # If bg_ratio < 0.05, it is a texture line purely inside the fabric.
    return 0.05 < bg_ratio < 0.95


def _get_control_point(lines, img_shape):
    def intersect_with_horizontal(line, y_target):
        x1, y1, x2, y2 = line

        # The intersect point P would have the form (x1 + t(x2-x1), y1 + t(y2-y1)) (Parametric form of a line)
        # Solve for t, given the equation y1 + t(y2-y1) = y_target

        # Avoid division by zero (horizontal line)
        if y1 == y2:
            return None

        t = (y_target - y1) / (y2 - y1) # Solve for t

        # Check if intersection is within segment
        if t < 0 or t > 1:
            return None

        x = x1 + t * (x2 - x1)

        return int(x)

    if lines is None or len(lines) == 0:
        return None, None

    H, W = img_shape[:2]
    y_ref1 = 3 * H // 4
    y_ref2 = H // 4 

    candidates1 = []
    candidates2 = []

    for line in lines:
        x1 = intersect_with_horizontal(line, y_ref1)
        if x1:
            candidates1.append(x1)
            
        x2 = intersect_with_horizontal(line, y_ref2)
        if x2:
            candidates2.append(x2)

    center_x = W // 2
    
    best_pt1 = None
    if len(candidates1) > 0:
        best_x1 = min(candidates1, key=lambda x: abs(x - center_x))
        best_pt1 = (int(best_x1), int(y_ref1))

    best_pt2 = None
    if len(candidates2) > 0:
        best_x2 = min(candidates2, key=lambda x: abs(x - center_x))
        best_pt2 = (int(best_x2), int(y_ref2))

    return best_pt1, best_pt2


def _get_control_point_weighted(lines, scores, img_shape):
    """
    Like _get_control_point but selects the scan-line intersection with the
    highest ensemble confidence score.  Falls back to closest-to-centre when
    scores are equal or not provided.

    Parameters
    ----------
    lines  : (N, 4) float32 array or None
    scores : (N,) float32 confidence array (same order as lines), or None
    """
    def intersect_with_horizontal(line, y_target):
        x1, y1, x2, y2 = line
        if y1 == y2:
            return None
        t = (y_target - y1) / (y2 - y1)
        if t < 0 or t > 1:
            return None
        return int(x1 + t * (x2 - x1))

    if lines is None or len(lines) == 0:
        return None, None

    H, W = img_shape[:2]
    y_ref1 = H // 4
    y_ref2 = 3 * H // 4
    center_x = W // 2

    candidates1 = []  # (x, score)
    candidates2 = []

    for i, line in enumerate(lines):
        score = float(scores[i]) if (scores is not None and i < len(scores)) else 0.0

        x1 = intersect_with_horizontal(line, y_ref1)
        if x1 is not None:
            candidates1.append((x1, score))
            
        x2 = intersect_with_horizontal(line, y_ref2)
        if x2 is not None:
            candidates2.append((x2, score))

    best_pt1 = None
    if len(candidates1) > 0:
        best_x1 = max(candidates1, key=lambda c: (c[1], -abs(c[0] - center_x)))[0]
        best_pt1 = (int(best_x1), int(y_ref1))

    best_pt2 = None
    if len(candidates2) > 0:
        best_x2 = max(candidates2, key=lambda c: (c[1], -abs(c[0] - center_x)))[0]
        best_pt2 = (int(best_x2), int(y_ref2))

    return best_pt1, best_pt2


def _compute_offset_roller(point1, point2, img_shape):
        H, W = img_shape[:2]
        center_x = W // 2
    
        err1 = (point1[0] - center_x) if point1 is not None else None
        err2 = (point2[0] - center_x) if point2 is not None else None

        return err1, err2


def get_channel_histograms(image, bin_num, use_gray=False):
    if use_gray:
        # Ensure image is grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image
            
        hist = cv.calcHist([gray], [0], None, [bin_num], [0, 256])
        # Normalize to probability distribution (sum of bins equals 1.0)
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist = hist / hist_sum
        return hist.flatten()
    else:
        hists = []
        # Expecting color image
        for i in range(3):
            hist = cv.calcHist([image], [i], None, [bin_num], [0, 256])
            # Normalize to percentage (sum of each channel's bins equals 1.0)
            hist_sum = hist.sum()
            if hist_sum > 0:
                hist = hist / hist_sum
            hists.append(hist)
        return np.concatenate(hists).flatten()


class Kalman1DTracker:
    def __init__(self, process_noise=1e-4, measurement_noise=1e-1, max_missed=2):
        self.kf = cv.KalmanFilter(2, 1, 0)
        # State: [error, velocity]
        self.kf.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        # Measurement matrix: we only measure error
        self.kf.measurementMatrix = np.array([[1, 0]], np.float32)
        
        self.kf.processNoiseCov = np.eye(2, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(1, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(2, dtype=np.float32)
        
        self.kf.statePost = np.zeros((2, 1), np.float32)
        
        self.initialized = False
        self.missed_frames = 0
        self.max_missed = max_missed

    def update(self, measurement):
        """
        Updates tracking and returns (tracked_value, is_predicted)
        """
        if measurement is None:
            self.missed_frames += 1
            if self.missed_frames > self.max_missed or not self.initialized:
                self.initialized = False
                return None, False
            
            # Predict only
            prediction = self.kf.predict()
            return float(prediction[0, 0]), True
        else:
            if not self.initialized:
                self.kf.statePost = np.array([[float(measurement)], [0.0]], np.float32)
                self.initialized = True
            
            self.missed_frames = 0
            # Predict and then update
            self.kf.predict()
            self.kf.correct(np.array([[float(measurement)]], np.float32))
            return float(measurement), False


# ---------------------------------------------------------------------------
# Output normalisation
# ---------------------------------------------------------------------------

def normalize_lines(lines) -> Optional[np.ndarray]:
    """
    Normalise any detector output to shape (N, 4) float32: [x1, y1, x2, y2].

    Handles:
      - LSD / FLD output: (N, 1, 4) or (N, 4)
      - EDLines output:   (N, 1, 4) where cols are [x1, y1, x2, y2]
    Returns None if input is None or empty.
    """
    if lines is None:
        return None
    lines = np.array(lines, dtype=np.float32)
    if lines.ndim == 3 and lines.shape[1] == 1:
        lines = lines.reshape(-1, 4)
    elif lines.ndim == 2 and lines.shape[1] == 4:
        pass  # already correct
    else:
        return None
    if len(lines) == 0:
        return None
    return lines


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_lines_colour(img: np.ndarray, lines: Optional[np.ndarray],
                       colour: Tuple[int, int, int], thickness: int = 1) -> np.ndarray:
    if lines is None or len(lines) == 0:
        return img
    out = img
    for x1, y1, x2, y2 in lines.astype(int):
        cv.line(out, (x1, y1), (x2, y2), colour, thickness, cv.LINE_AA)
    return out


def draw_lines(img: np.ndarray, lines: Optional[np.ndarray],
               mode) -> np.ndarray:
    """Draw line segments on img. Lines must be (N,4) float32."""
    from .line_detectors.detector import DetectorMode, fld, _FUSED_COLOUR
    if lines is None or len(lines) == 0:
        return img
    if mode == DetectorMode.LSD:
        return lsd.drawSegments(img, lines)
    elif mode == DetectorMode.FLD:
        return fld.drawSegments(img, lines)
    elif mode == DetectorMode.ENSEMBLE:
        # ENSEMBLE calls its own visualisation; this branch shouldn't normally be hit.
        return _draw_lines_colour(img, lines, _FUSED_COLOUR, 2)
    else:  # EDLINES
        return _draw_lines_colour(img, lines, (0, 255, 0), 1)


def visualize_and_save_frame_sewer(
    gray, img, lines, detector_mode,
    error1, error2, is_pred1, is_pred2, control_signals,
    session_capture_dir, frame_count
):
    drawn_img = img.copy()
    if lines is not None:
        drawn_img = draw_lines(drawn_img, lines, detector_mode)
        
    # Draw tracked errors
    y_ofs1 = 60
    y_ofs2 = 20

    def sig_to_char(sig):
        if sig == [False, False]: return 'num'
        if sig == [True, False]: return 'a'
        if sig == [True, True]: return 'b'
        return '-'

    c1 = sig_to_char(control_signals[0])
    c2 = sig_to_char(control_signals[1])

    text1 = f"{int(error1)} [{c1}]" if is_pred1 else f"{int(error1)} [{c1}]" if error1 is not None else f"NaN [{c1}]"
    text2 = f"{int(error2)} [{c2}]" if is_pred2 else f"{int(error2)} [{c2}]" if error2 is not None else f"NaN [{c2}]"
    
    color1 = (0, 165, 255) if is_pred1 else (0, 255, 0)
    color2 = (0, 165, 255) if is_pred2 else (0, 255, 0)
    
    cv.putText(drawn_img, text1, (10, y_ofs1), cv.FONT_HERSHEY_SIMPLEX, 0.3, color1, 1)
    cv.putText(drawn_img, text2, (10, y_ofs2), cv.FONT_HERSHEY_SIMPLEX, 0.3, color2, 1)
        
    gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    combined_vis = cv.hconcat([gray_bgr, drawn_img])
    
    # Zoom the display window (scale up 3x)
    zoomed_vis = cv.resize(combined_vis, None, fx=3.0, fy=3.0, interpolation=cv.INTER_NEAREST)

    # Save the captured frame with text/prediction for debugging
    frame_filename = os.path.join(session_capture_dir, f"frame_{frame_count:05d}.jpg")
    cv.imwrite(frame_filename, zoomed_vis)
    
    cv.imshow("Gray (Left) vs Line Detection (Right)", zoomed_vis)


def visualize_and_save_frame_adjustor(
    gray, img, lines, best_h_line, best_v_line, detector_mode,
    control_signals, x_basis, y_basis,
    session_capture_dir, frame_count
):
    drawn_img = img.copy()
    if lines is not None:
        drawn_img = draw_lines(drawn_img, lines, detector_mode)

    draw_adjustor_overlay(
        drawn_img,
        basis=(x_basis, y_basis),
        axis_length=AXIS_LENGTH_PX,
        best_h_line=best_h_line,
        best_v_line=best_v_line,
        control_signals=control_signals if (best_h_line is not None and best_v_line is not None) else None,
    )

    gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    
    # Pad gray_bgr to match drawn_img shape if they differ due to cropping
    if gray_bgr.shape != drawn_img.shape:
        padded_gray = np.zeros_like(drawn_img)
        # We don't have offset_x/offset_y here, but centering or top-left padding is fine for visualization
        h, w = gray_bgr.shape[:2]
        padded_gray[:h, :w] = gray_bgr
        gray_bgr = padded_gray

    combined_vis = cv.hconcat([gray_bgr, drawn_img])
    
    # Zoom the display window (scale up 3x)
    zoomed_vis = cv.resize(combined_vis, None, fx=1.0, fy=1.0, interpolation=cv.INTER_NEAREST)

    # Save the captured frame with text/prediction for debugging
    frame_filename = os.path.join(session_capture_dir, f"frame_{frame_count:05d}.jpg")
    cv.imwrite(frame_filename, zoomed_vis)
    # threading.Thread(target=cv.imwrite, args=(frame_filename, zoomed_vis)).start()
    cv.imshow("Gray (Left) vs Line Detection (Right)", zoomed_vis)
