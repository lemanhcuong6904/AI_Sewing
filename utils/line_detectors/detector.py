from typing import Tuple, Optional, List
from enum import Enum
from collections import defaultdict
import os
import cv2 as cv
import numpy as np
from numba import njit
try:
    import onnxruntime as ort
except ImportError:
    ort = None


from utils.helper import (
    _endpoint_distance,
    _get_mask,
    _line_is_near_background_sewer,
    _line_is_near_background_adjustor,
    _line_length,
    normalize_lines,
    _draw_lines_colour,
    draw_lines,
)
from utils.constants import PX_PER_CM_X, PX_PER_CM_Y


class DetectorMode(Enum):
    LSD     = "lsd"
    FLD     = "fld"
    EDLINES = "edlines"
    ENSEMBLE = "ensemble"
    MLSD    = "mlsd"
    DEXINED = "dexined"


# --- Detector instances ---
lsd = cv.createLineSegmentDetector(cv.LSD_REFINE_STD)

try:
    fld = cv.ximgproc.createFastLineDetector(
        length_threshold=10,
        distance_threshold=1.41421356,
        canny_th1=50,
        canny_th2=50,
        canny_aperture_size=3,
        do_merge=True,
    )
except AttributeError:
    print("[detector] Warning: cv2.ximgproc.createFastLineDetector not available. FLD will not work.")
    fld = None

# EdgeDrawing — params API (EdgeDrawing_Params / setParams) requires OpenCV >= 4.6.
# On OpenCV 4.5.x we just create the detector and rely on its built-in defaults.
try:
    _ed = cv.ximgproc.createEdgeDrawing()
except AttributeError:
    print("[detector] Warning: cv2.ximgproc.createEdgeDrawing not available. EDLines will not work.")
    _ed = None

clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# Colours for ensemble visualisation (BGR)
_ENSEMBLE_COLOURS = [
    (0,   200, 255),   # LSD     — amber
    (255, 180,   0),   # FLD     — cyan-blue
    (0,   255, 120),   # EDLines — green
]
_FUSED_COLOUR   = (0, 0, 255)   # fused lines — red




# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocessing_sewer(
        img,
        img_size=None,
        crop_roi=None,
        enhance_contrast=True,
        smooth_texture=True,
        # square_contrast=True,
    ):
    # ---- center crop ----
    if crop_roi is not None:
        sy, ey, sx, ex = crop_roi
        img = img[sy:ey, sx:ex]
    else:
        h, w = img.shape[:2]
        min_dim = min(h, w)
        crop_size_y = int(min_dim / 5)
        crop_size_x = int(min_dim / 2.5 * 0.87)
        start_x = int((w - crop_size_x) / 2)
        start_y = (h - crop_size_y) // 2
        img = img[start_y:start_y + crop_size_y,
                  start_x:start_x + crop_size_x]

    # ---- resize ----
    if img_size:
        img = cv.resize(img, img_size)

    # Downscale the image to speed up processing (1x pyrDown)
    img = cv.pyrDown(img)

    # convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # ---- Texture smoothing ----
    if smooth_texture:
        # omment out the old method so it's easy to fall back
        # gray = cv.bilateralFilter(gray, d=7, sigmaColor=75, sigmaSpace=75) 
        gray = cv.GaussianBlur(gray, (5, 5), 0)

    # ---- Contrast enhancement ----
    if enhance_contrast:
        gray = clahe.apply(gray)
        
    # ---- Non-linear Contrast Transformation (Squaring) ----
    # if square_contrast:
        # Scale to [0, 1], square to make dark pixels exponentially darker, and scale back to [0, 255]
        gray = ((gray.astype(np.float32) / 255.0) ** 2 * 255.0).astype(np.uint8)

    return gray, img


def preprocessing_adjustor(
        img,
        img_size=None,
        crop_roi=None,
        enhance_contrast=True,
        smooth_texture=True,
        # square_contrast=False,
    ):
    # ---- center crop ----
    offset_y, offset_x = 0, 0
    crop_h, crop_w = img.shape[:2]
    if crop_roi is not None:
        sx, ex, sy, ey = crop_roi
        img = img[sy:ey, sx:ex]
        offset_y = sy
        offset_x = sx
        crop_h, crop_w = ey - sy, ex - sx
    else:
        pass
    
    # Downscale the image to speed up processing (1x pyrDown)
    h, w = img.shape[:2]           # (H, W) — note: cv.resize needs (W, H)
    scale = 0
    for _ in range(scale):
        img = cv.pyrDown(img)

    # ---- resize back to original dimensions ----
    scale_x, scale_y = 1.0, 1.0
    if img_size:
        img = cv.resize(img, img_size)   # img_size already expected as (W, H)
        scale_x = crop_w / img_size[0]
        scale_y = crop_h / img_size[1]
    else:
        img = cv.resize(img, (w, h))     # ← fixed: was (h, w) which transposed the aspect ratio
    if h > 356:
        img = img[0:356, :]

    # convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # ---- Texture smoothing ----
    if smooth_texture:
        # omment out the old method so it's easy to fall back
        # gray = cv.bilateralFilter(gray, d=7, sigmaColor=75, sigmaSpace=75) 
        # gray = cv.GaussianBlur(gray, (5, 5), 0)
        img = cv.GaussianBlur(img, (5, 5), 0)

    # ---- Contrast enhancement ----
    if enhance_contrast:
        # gray = clahe.apply(gray)
        
    # ---- Non-linear Contrast Transformation (Squaring) ----
    # if square_contrast:
        # Scale to [0, 1], square to make dark pixels exponentially darker, and scale back to [0, 255]
        gray = ((gray.astype(np.float32) / 255.0) ** 3 * 255.0).astype(np.uint8)

    return gray, img, (offset_x, offset_y, scale_x, scale_y)

from concurrent.futures import ThreadPoolExecutor

def preprocessing_adjustor_batch(
        frames,
        img_size=None,
        crop_roi=None,
        enhance_contrast=True,
        smooth_texture=True,
    ):
    """
    Parallelize preprocessing_adjustor over a batch of frames using ThreadPoolExecutor.
    """
    def process_single(frame):
        return preprocessing_adjustor(
            frame, img_size=img_size, crop_roi=crop_roi,
            enhance_contrast=enhance_contrast, smooth_texture=smooth_texture
        )
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_single, frames))
        
    grays = [r[0] for r in results]
    imgs = [r[1] for r in results]
    offsets = [r[2] for r in results]
    return grays, imgs, offsets

# ---------------------------------------------------------------------------
# Deep Learning Setup
# ---------------------------------------------------------------------------

_mlsd_session = None
_tplsd_session = None

def _get_mlsd_session():
    global _mlsd_session
    if _mlsd_session is None:
        engine_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "m-lsd", "mlsd_tiny_512_fp32.engine")
        if os.path.exists(engine_path):
            try:
                from utils.line_detectors.trt_infer import TRTInfer
                _mlsd_session = TRTInfer(engine_path)
                print("[detector] M-LSD TensorRT engine loaded successfully.")
                return _mlsd_session
            except ImportError as e:
                print(f"[detector] Warning: Could not load TensorRT engine: {e}")
        
        if ort is not None:
            model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "m-lsd", "mlsd_tiny_512_fp32.onnx")
            if os.path.exists(model_path):
                _mlsd_session = ort.InferenceSession(model_path, providers=[
                    ('TensorrtExecutionProvider', {
                        'trt_engine_cache_enable': True,
                        'trt_engine_cache_path': './models/m-lsd',
                        'trt_fp16_enable': True
                    }),
                    'CUDAExecutionProvider',
                    'CPUExecutionProvider'
                ])
                print("[detector] M-LSD ONNX loaded successfully.")
            else:
                print(f"[detector] Warning: M-LSD model not found at {model_path}.")
    return _mlsd_session


def preprocessing_dl_from_processed_img(img_processed, input_shape=(512, 512)):
    # M-LSD requires 512x512, and 4 channels (RGBA)
    img_rgba = cv.cvtColor(img_processed, cv.COLOR_BGR2RGBA)
    resized_image = cv.resize(img_rgba, (input_shape[1], input_shape[0]), interpolation=cv.INTER_AREA)
    
    img_float = resized_image.astype(np.float32) / 127.5 - 1.0
    tensor = np.transpose(img_float, (2, 0, 1)) # CHW
    tensor = np.expand_dims(tensor, axis=0) # NCHW
    return tensor

def preprocessing_dl_from_processed_img_batch(imgs_processed, input_shape=(512, 512)):
    def process_single(img):
        img_rgba = cv.cvtColor(img, cv.COLOR_BGR2RGBA)
        resized_image = cv.resize(img_rgba, (input_shape[1], input_shape[0]), interpolation=cv.INTER_AREA)
        img_float = resized_image.astype(np.float32) / 127.5 - 1.0
        return np.transpose(img_float, (2, 0, 1))
        
    with ThreadPoolExecutor() as executor:
        tensors = list(executor.map(process_single, imgs_processed))
        
    return np.stack(tensors, axis=0)


# ---------------------------------------------------------------------------
# Single-detector dispatch
# ---------------------------------------------------------------------------

def detect_lines(gray: np.ndarray, mode: DetectorMode, img_for_dl: np.ndarray = None) -> Optional[np.ndarray]:
    """
    Run one detector on a grayscale image.
    For DL models, img_for_dl (BGR) is required.
    Returns lines normalised to (N, 4) float32, or None.
    """
    from utils.line_detectors.mlsd_postprocess import postprocess_mlsd_squares
    
    if mode == DetectorMode.MLSD:
        session = _get_mlsd_session()
        if session is None or img_for_dl is None: return None
        tensor = preprocessing_dl_from_processed_img(img_for_dl)
        if hasattr(session, 'run'):
            outputs = session.run(None, {session.get_inputs()[0].name: tensor})
        else:
            outputs = session(tensor)
        return postprocess_mlsd_squares(outputs, img_for_dl.shape[:2])
    elif mode == DetectorMode.LSD:
        raw = lsd.detect(gray)[0]
        return normalize_lines(raw)
    elif mode == DetectorMode.FLD:
        raw = fld.detect(gray)
        return normalize_lines(raw)
    elif mode == DetectorMode.EDLINES:
        _ed.detectEdges(gray)
        raw = _ed.detectLines()
        return normalize_lines(raw)
    return None

def detect_lines_batch(grays: List[np.ndarray], mode: DetectorMode, imgs_for_dl: List[np.ndarray] = None) -> List[Optional[np.ndarray]]:
    """
    Run detector on a batch of grayscale images.
    Optimized batched TensorRT execution for M-LSD. 
    Classical algorithms use ThreadPoolExecutor.
    """
    if mode == DetectorMode.MLSD:
        session = _get_mlsd_session()
        if session is None or imgs_for_dl is None or len(imgs_for_dl) == 0:
            return [None] * len(grays)
            
        from utils.line_detectors.mlsd_postprocess import postprocess_mlsd_squares_batch
        
        # Batch preprocessing
        batch_tensor = preprocessing_dl_from_processed_img_batch(imgs_for_dl)
        
        # Batch inference
        if hasattr(session, 'run'):
            outputs = session.run(None, {session.get_inputs()[0].name: batch_tensor})
        else:
            outputs = session(batch_tensor)
            
        # Batch postprocessing
        shapes = [img.shape[:2] for img in imgs_for_dl]
        return postprocess_mlsd_squares_batch(outputs, shapes)
        
    else:
        # Fallback to parallel per-frame detection for classical methods
        def process_single(i):
            img_dl = imgs_for_dl[i] if imgs_for_dl is not None else None
            return detect_lines(grays[i], mode, img_dl)
            
        with ThreadPoolExecutor() as executor:
            return list(executor.map(process_single, range(len(grays))))


# ---------------------------------------------------------------------------
# Background filter & postprocessing (unchanged)
# ---------------------------------------------------------------------------

def keep_lines_near_background(lines, gray, mask_thresh=30, mode="sewer"):
    if lines is None:
        return None
    lines = lines.reshape(-1, 4)
    mask = _get_mask(gray, mask_thresh)
    if mode == "adjustor":
        kept = [line for line in lines if _line_is_near_background_adjustor(line, mask)]
    else:
        kept = [line for line in lines if _line_is_near_background_sewer(line, mask)]
    if len(kept) == 0:
        return None
    return np.array(kept, dtype=np.float32)


@njit(cache=True)
def _filter_isolated_lines_jit(lines, dist_thresh, strong_line_len):
    n = lines.shape[0]
    keep_mask = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        l1 = lines[i]
        dx = l1[2] - l1[0]
        dy = l1[3] - l1[1]
        length = np.sqrt(dx * dx + dy * dy)
        if length >= strong_line_len:
            keep_mask[i] = True
            continue
        for j in range(n):
            if i == j:
                continue
            min_dist = np.inf
            for a in range(2):
                for b in range(2):
                    p1x = l1[a * 2]
                    p1y = l1[a * 2 + 1]
                    p2x = lines[j, b * 2]
                    p2y = lines[j, b * 2 + 1]
                    d = np.sqrt((p1x - p2x) ** 2 + (p1y - p2y) ** 2)
                    if d < min_dist:
                        min_dist = d
            if min_dist < dist_thresh:
                keep_mask[i] = True
                break
    return lines[keep_mask]


def filter_dominant_angles(lines, num_peaks=2, tolerance_deg=12.0, bin_size_deg=5.0,
                           gradient_scores=None, length_weighting='linear', offset=None):
    """
    Keep only lines whose orientations cluster around the top ``num_peaks``
    dominant angular directions in the image.

    Parameters
    ----------
    lines : np.ndarray of shape (N, 4)  [x1, y1, x2, y2]
    num_peaks : int
        Maximum number of dominant angle clusters to preserve.
    tolerance_deg : float
        A line is kept if its angle is within this many degrees of a peak.
    bin_size_deg : float
        Angular resolution of the histogram.
    gradient_scores : np.ndarray of shape (N,) or None
        Per-line gradient edge-strength scores in [0, 1].
        When provided, each line votes for its histogram bin proportionally
        to ``length × gradient_score`` rather than ``length`` alone.
        Lines in uniform background regions (score ≈ 0) contribute almost
        nothing to any histogram bin, preventing spurious dominant-angle peaks.
        If None, falls back to the original pure length-weighting.
    """
    if lines is None or len(lines) == 0:
        return lines
        
    lines_2d = lines.reshape(-1, 4)
    if offset is not None:
        dx = (lines_2d[:, 2] - lines_2d[:, 0]) * offset[2]
        dy = (lines_2d[:, 3] - lines_2d[:, 1]) * offset[3]
    else:
        dx = lines_2d[:, 2] - lines_2d[:, 0]
        dy = lines_2d[:, 3] - lines_2d[:, 1]
    lengths = np.hypot(dx, dy)
    
    # We only care about line orientation in [0, 180)
    angles = np.abs(np.arctan2(dy, dx) * 180.0 / np.pi) % 180.0
    
    # Apply length weighting
    if length_weighting == 'uniform':
        weighted_lengths = np.ones_like(lengths)
    elif length_weighting == 'sqrt':
        weighted_lengths = np.sqrt(lengths)
    elif length_weighting == 'log':
        weighted_lengths = np.log1p(lengths)
    else: # linear
        weighted_lengths = lengths

    # Build histogram weights: weighted length alone, or weighted length × gradient score
    if gradient_scores is not None and len(gradient_scores) == len(lines):
        hist_weights = weighted_lengths * np.asarray(gradient_scores, dtype=np.float64)
    else:
        hist_weights = weighted_lengths
    
    # Build weighted histogram
    bins = np.arange(0, 180 + bin_size_deg, bin_size_deg)
    hist, bin_edges = np.histogram(angles, bins=bins, weights=hist_weights)
    
    # Smooth the histogram (circular convolution) to avoid splitting peaks
    hist_smooth = np.convolve(np.pad(hist, (1, 1), mode='wrap'), [0.25, 0.5, 0.25], mode='valid')
    
    # Find local maxima (peaks)
    peaks = []
    n_bins = len(hist_smooth)
    for i in range(n_bins):
        prev_i = (i - 1) % n_bins
        next_i = (i + 1) % n_bins
        if hist_smooth[i] > hist_smooth[prev_i] and hist_smooth[i] >= hist_smooth[next_i]:
            peaks.append((hist_smooth[i], i))
            
    if not peaks:
        return lines
        
    # Sort peaks by weight (descending)
    peaks.sort(key=lambda x: x[0], reverse=True)
    
    # Select top dominant angles
    dominant_angles = []
    max_weight = peaks[0][0]
    for weight, bin_idx in peaks[:num_peaks]:
        # Only keep peak if it's at least 10% of the max peak to avoid keeping pure noise
        if weight > max_weight * 0.1: 
            angle = bin_edges[bin_idx] + bin_size_deg / 2.0
            dominant_angles.append(angle)
            
    if not dominant_angles:
        return lines
        
    # Filter lines: keep all lines geometrically close to a dominant angle,
    # regardless of their individual gradient score (that is handled downstream)
    keep_mask = np.zeros(len(lines), dtype=np.bool_)
    for i, angle in enumerate(angles):
        for dom_angle in dominant_angles:
            # Shortest circular distance
            diff = np.abs(angle - dom_angle)
            if diff > 90:
                diff = 180 - diff
            if diff <= tolerance_deg:
                keep_mask[i] = True
                break
                
    return lines[keep_mask]


def map_noisy_areas_to_roi(noisy_areas, offset):
    """
    Map noisy areas defined in original full-frame coordinates (in cm) into local ROI space.
    offset: Tuple (offset_x, offset_y, scale_x, scale_y) returned by preprocessing_adjustor.
    """
    if not noisy_areas or offset is None:
        return noisy_areas

    offset_x, offset_y, scale_x, scale_y = offset
    mapped_areas = []
    for area_config in noisy_areas:
        x_center_cm, y_center_cm, height_cm, width_cm = area_config[:4]
        
        # Convert original physical cm to original pixel space
        x_center_px = x_center_cm * PX_PER_CM_X
        y_center_px = y_center_cm * PX_PER_CM_Y
        width_px = width_cm * PX_PER_CM_X
        height_px = height_cm * PX_PER_CM_Y
        
        # Map original full-frame pixels to ROI pixel coordinates
        x_roi_px = (x_center_px - offset_x) / scale_x
        y_roi_px = (y_center_px - offset_y) / scale_y
        w_roi_px = width_px / scale_x
        h_roi_px = height_px / scale_y
        
        # Convert back to cm in ROI coordinate space
        x_roi_cm = x_roi_px / PX_PER_CM_X
        y_roi_cm = y_roi_px / PX_PER_CM_Y
        h_roi_cm = h_roi_px / PX_PER_CM_Y
        w_roi_cm = w_roi_px / PX_PER_CM_X
        
        mapped_area = (x_roi_cm, y_roi_cm, h_roi_cm, w_roi_cm) + area_config[4:]
        mapped_areas.append(mapped_area)
        
    return mapped_areas


def filter_noisy_lines_by_area(lines, noisy_areas=None, offset=None):
    """
    Filter out static noisy lines that fall inside specific areas, provided they are short.
    noisy_areas: list of tuples (x_center_cm, y_center_cm, height_cm, width_cm) or (x_center_cm, y_center_cm, height_cm, width_cm, max_length)
                 or (x_center_cm, y_center_cm, height_cm, width_cm, max_length, orientation).
                 Defined in original full-frame coordinate space.
    If max_length is not provided, defaults to 1.5 cm.
    orientation can be 'horizontal' or 'vertical'. If provided, only lines with matching orientation are filtered.
    A line is filtered out if its center or either endpoint falls within any of the noisy_areas 
    AND its length is less than or equal to max_length (in cm) AND its orientation matches (if specified).
    offset: Optional tuple (offset_x, offset_y, scale_x, scale_y) to map noisy_areas into local ROI coordinates before checking.
    """
    if lines is None or len(lines) == 0 or not noisy_areas:
        return lines
        
    lines_2d = lines.reshape(-1, 4)
    
    # Map noisy areas from original space to local ROI space
    roi_noisy_areas = map_noisy_areas_to_roi(noisy_areas, offset) if offset is not None else noisy_areas
    
    scale_x = offset[2] if offset is not None else 1.0
    scale_y = offset[3] if offset is not None else 1.0
    
    keep_mask = np.ones(len(lines_2d), dtype=np.bool_)
    
    # Calculate physical displacement in cm (accounting for potential ROI resizing via scale_x, scale_y)
    dx_cm = (lines_2d[:, 2] - lines_2d[:, 0]) * scale_x / PX_PER_CM_X
    dy_cm = (lines_2d[:, 3] - lines_2d[:, 1]) * scale_y / PX_PER_CM_Y
    lengths_cm = np.hypot(dx_cm, dy_cm)
    
    angles = np.abs(np.arctan2(dy_cm, dx_cm) * 180.0 / np.pi) % 180.0
    
    cx = (lines_2d[:, 0] + lines_2d[:, 2]) / 2.0
    cy = (lines_2d[:, 1] + lines_2d[:, 3]) / 2.0
    
    for i in range(len(lines_2d)):
        line_len = lengths_cm[i]
        angle = angles[i]
        
        for area_config in roi_noisy_areas:
            if len(area_config) >= 6:
                x_center_cm, y_center_cm, height_cm, width_cm, max_len, orientation = area_config[:6]
            elif len(area_config) >= 5:
                x_center_cm, y_center_cm, height_cm, width_cm, max_len = area_config[:5]
                orientation = None
            else:
                x_center_cm, y_center_cm, height_cm, width_cm = area_config[:4]
                max_len = 1.5  # default max length for noise in cm
                orientation = None
                
            width_px = width_cm * PX_PER_CM_X
            height_px = height_cm * PX_PER_CM_Y
            x_center_px = x_center_cm * PX_PER_CM_X
            y_center_px = y_center_cm * PX_PER_CM_Y
                
            x_min = x_center_px - width_px / 2.0
            x_max = x_center_px + width_px / 2.0
            y_min = y_center_px - height_px / 2.0
            y_max = y_center_px + height_px / 2.0

            in_area = (
                (x_min <= cx[i] <= x_max and y_min <= cy[i] <= y_max) or
                (x_min <= lines_2d[i, 0] <= x_max and y_min <= lines_2d[i, 1] <= y_max) or
                (x_min <= lines_2d[i, 2] <= x_max and y_min <= lines_2d[i, 3] <= y_max)
            )
            
            orientation_match = True
            if orientation == 'horizontal':
                if not (angle <= 45 or angle >= 135):
                    orientation_match = False
            elif orientation == 'vertical':
                if not (45 < angle < 135):
                    orientation_match = False
            
            if in_area and line_len <= max_len and orientation_match:
                keep_mask[i] = False
                break
                
    return lines_2d[keep_mask]


def postprocess_lines(
        lines,
        min_noise_len=5,
        strong_line_len=60,
        min_boundary_len=20,
        connect_dist=7
    ):
    def filter_short_lines(lines, min_length):
        if lines is None:
            return None
        lines = lines.reshape(-1, 4)
        dx = lines[:, 2] - lines[:, 0]
        dy = lines[:, 3] - lines[:, 1]
        lengths = np.hypot(dx, dy)
        return lines[lengths >= min_length]

    if lines is None:
        return None
    lines = lines.reshape(-1, 4)

    lines = filter_short_lines(lines, min_length=min_noise_len)
    if lines is None or len(lines) == 0:
        return None

    lines = _filter_isolated_lines_jit(
        lines.astype(np.float64),
        float(connect_dist),
        float(strong_line_len)
    )

    lines = filter_short_lines(lines, min_length=min_boundary_len)

    return lines


# ---------------------------------------------------------------------------
# Ensemble stage 1 — collinear merging (per-detector)
# ---------------------------------------------------------------------------

def merge_collinear_lines(
        lines: Optional[np.ndarray],
        angle_thresh_deg: float = 8.0,
        dist_thresh: float = 6.0,
) -> Optional[np.ndarray]:
    """
    Collapse collinear fragments produced by a single detector into one segment.

    Algorithm:
      1. Compute the dominant angle of each segment (0–180°).
      2. Bucket segments by angle (±angle_thresh_deg).
      3. Within each angle bucket, sub-group by perpendicular distance from
         the image origin (±dist_thresh px).
      4. Merge every sub-group into a single segment spanning its two most
         distant endpoints.

    Returns (N, 4) float32 or None.
    """
    if lines is None or len(lines) == 0:
        return lines

    lines = lines.reshape(-1, 4).astype(np.float32)

    # --- compute angle in [0, 180) for each segment ---
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    angles = np.degrees(np.arctan2(dy, dx)) % 180.0   # (N,)

    # --- compute perpendicular distance from origin (mid-point projection) ---
    mx = (lines[:, 0] + lines[:, 2]) / 2.0
    my = (lines[:, 1] + lines[:, 3]) / 2.0
    rads = np.radians(angles)
    perp = mx * np.cos(rads + np.pi / 2) + my * np.sin(rads + np.pi / 2)

    # --- bucket assignment ---
    angle_bucket = (angles / angle_thresh_deg).astype(int)
    perp_bucket  = (perp  / dist_thresh).astype(int)

    groups: dict = defaultdict(list)
    for i in range(len(lines)):
        key = (angle_bucket[i], perp_bucket[i])
        groups[key].append(i)

    merged = []
    for idxs in groups.values():
        if len(idxs) == 1:
            merged.append(lines[idxs[0]])
            continue

        # Gather all 4 endpoints for this group
        pts = []
        for i in idxs:
            pts.append((lines[i, 0], lines[i, 1]))
            pts.append((lines[i, 2], lines[i, 3]))

        # Project onto the mean direction to find the two extremes
        mean_angle = float(np.mean(angles[idxs]))
        rad = np.radians(mean_angle)
        ux, uy = np.cos(rad), np.sin(rad)

        projections = [ux * p[0] + uy * p[1] for p in pts]
        i_min = int(np.argmin(projections))
        i_max = int(np.argmax(projections))

        merged.append([pts[i_min][0], pts[i_min][1],
                       pts[i_max][0], pts[i_max][1]])

    if len(merged) == 0:
        return None
    return np.array(merged, dtype=np.float32)


# ---------------------------------------------------------------------------
# Ensemble stage 2 — cross-detector voting
# ---------------------------------------------------------------------------

def _lines_are_compatible(
        a: np.ndarray, b: np.ndarray,
        angle_thresh_deg: float, dist_thresh: float,
) -> bool:
    """Return True when two segments are roughly collinear and close."""
    # Angle test
    ax = a[2] - a[0]; ay = a[3] - a[1]
    bx = b[2] - b[0]; by = b[3] - b[1]
    ang_a = np.degrees(np.arctan2(ay, ax)) % 180.0
    ang_b = np.degrees(np.arctan2(by, bx)) % 180.0
    diff  = abs(ang_a - ang_b)
    if diff > 90:
        diff = 180.0 - diff
    if diff > angle_thresh_deg:
        return False

    # Perpendicular distance between mid-points along the normal of `a`
    rad = np.radians(ang_a + 90)
    nx, ny = np.cos(rad), np.sin(rad)
    mid_a = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    mid_b = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    perp_dist = abs(nx * (mid_b[0] - mid_a[0]) + ny * (mid_b[1] - mid_a[1]))
    return perp_dist <= dist_thresh


def _score_lines_with_supporters(
        primary: np.ndarray,
        supporter_sets: List[Optional[np.ndarray]],
        angle_thresh_deg: float = 8.0,
        dist_thresh: float = 8.0,
) -> np.ndarray:
    """
    Score each line in `primary` by length × (1 + number of supporter detectors that agree).
    Lines with no supporters still get their base length score.
    """
    dx = primary[:, 2] - primary[:, 0]
    dy = primary[:, 3] - primary[:, 1]
    lengths = np.hypot(dx, dy).astype(np.float32)

    agreement = np.zeros(len(primary), dtype=np.float32)
    for sup_lines in supporter_sets:
        if sup_lines is None or len(sup_lines) == 0:
            continue
        for i, line in enumerate(primary):
            for sline in sup_lines:
                if _lines_are_compatible(line, sline, angle_thresh_deg, dist_thresh):
                    agreement[i] += 1.0
                    break  # count each supporter detector at most once

    scores = lengths * (1.0 + agreement)
    order  = np.argsort(scores)[::-1]
    return scores[order], order   # return scores and sort order


# ---------------------------------------------------------------------------
# Ensemble orchestrator  —  priority cascade: LSD → FLD → EDLines
# ---------------------------------------------------------------------------

def ensemble_detect_lines(
        gray: np.ndarray,
        mask_thresh: int = 240,
        angle_thresh_deg: float = 8.0,
        dist_thresh_merge: float = 6.0,
        dist_thresh_vote:  float = 8.0,
) -> Tuple[Optional[np.ndarray], np.ndarray, List[Optional[np.ndarray]]]:
    """
    Priority-cascade ensemble: LSD → FLD → EDLines.

    Strategy
    --------
    • LSD is the primary detector (most accurate).
    • If LSD detects no lines, FLD is used as fallback.
    • If neither LSD nor FLD detects lines, EDLines is the last resort.
    • Whichever detector is chosen as primary, the other two act purely as
      *supporters*: they can raise the confidence score of a primary line but
      never replace or dilute it.  This prevents EDLines noise from corrupting
      the result.

    Returns
    -------
    primary_lines    : (M, 4) float32 — lines from the winning detector
    scores           : (M,) float32   — length × (1 + supporter agreement)
    per_detector_raw : list of 3 arrays [lsd, fld, ed] for visualisation
    """
    _MODES = [DetectorMode.LSD, DetectorMode.FLD, DetectorMode.EDLINES]
    merged: List[Optional[np.ndarray]] = []

    for mode in _MODES:
        raw = detect_lines(gray, mode)
        raw = postprocess_lines(raw)
        raw = keep_lines_near_background(raw, gray, mask_thresh)
        raw = merge_collinear_lines(raw, angle_thresh_deg, dist_thresh_merge)
        merged.append(raw)

    lsd_lines, fld_lines, ed_lines = merged

    # --- Priority cascade ---
    def has_lines(ls):
        return ls is not None and len(ls) > 0

    if has_lines(lsd_lines):
        primary    = lsd_lines
        supporters = [fld_lines, ed_lines]
    elif has_lines(fld_lines):
        primary    = fld_lines
        supporters = [ed_lines]
    elif has_lines(ed_lines):
        primary    = ed_lines
        supporters = []
    else:
        return None, np.array([], dtype=np.float32), merged

    scores, order = _score_lines_with_supporters(
        primary, supporters, angle_thresh_deg, dist_thresh_vote
    )

    return primary[order].astype(np.float32), scores, merged


# ---------------------------------------------------------------------------
# Public API — run_line_detector / run_line_detector_with_offset
# ---------------------------------------------------------------------------

def run_line_detector(
        img,
        mode: DetectorMode = DetectorMode.LSD,
        img_size=None,
        enhance_contrast=True,
        smooth_texture=True,
    ):
    gray, img = preprocessing_sewer(
        img,
        img_size=img_size,
        enhance_contrast=enhance_contrast,
        smooth_texture=smooth_texture,
    )

    if mode == DetectorMode.ENSEMBLE:
        fused, _, per_det = ensemble_detect_lines(gray)
        vis = img.copy()
        # Draw each detector's merged lines in its own colour
        for colour, det_lines in zip(_ENSEMBLE_COLOURS, per_det):
            vis = _draw_lines_colour(vis, det_lines, colour, 1)
        # Draw fused lines on top
        vis = _draw_lines_colour(vis, fused, _FUSED_COLOUR, 2)
        return vis

    lines = detect_lines(gray, mode, img)
    if mode != DetectorMode.MLSD:
        lines = postprocess_lines(lines)
        lines = keep_lines_near_background(lines, gray)
        lines = filter_dominant_angles(lines)

    drawn_img = img.copy()
    if lines is not None and len(lines) > 0:
        lines = lines.astype(np.float32)
        drawn_img = draw_lines(img.copy(), lines, mode)

    return drawn_img
