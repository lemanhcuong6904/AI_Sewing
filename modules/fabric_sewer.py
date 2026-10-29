import os
import time
from datetime import datetime
from typing import Tuple, Optional, List
import cv2 as cv
import numpy as np

from utils.camera import get_camera, compute_hist_distance, DEFAULT_CROP_ROI
from utils.line_detectors.detector import (
    DetectorMode,
    preprocessing_sewer,
    postprocess_lines,
    keep_lines_near_background,
    detect_lines,
    ensemble_detect_lines,
)
from utils.helper import (
    _get_control_point,
    _get_control_point_weighted,
    _compute_offset_roller,
    get_channel_histograms,
    Kalman1DTracker,
    visualize_and_save_frame_sewer,
)


def _perform_line_detection(frame, img_size, enhance_contrast, smooth_texture, detector_mode: DetectorMode, mask_thresh: int):
    """
    Modularized frame preprocessing and line detection logic.
    """
    gray, img = preprocessing_sewer(
         frame, 
         img_size=img_size, 
         crop_roi=DEFAULT_CROP_ROI,
         enhance_contrast=enhance_contrast, 
         smooth_texture=smooth_texture, 
    )

    per_det = None
    if detector_mode == DetectorMode.ENSEMBLE:
        lines, scores, per_det = ensemble_detect_lines(gray)
        control_pt1, control_pt2 = _get_control_point_weighted(lines, scores, img.shape)
    else:
        lines = detect_lines(gray, detector_mode, img)
        if detector_mode != DetectorMode.MLSD:
            lines = postprocess_lines(lines)
        lines = keep_lines_near_background(lines, gray, mask_thresh, mode="sewer")
        if lines is not None:
            lines = lines.astype(np.float32)
        control_pt1, control_pt2 = _get_control_point(lines, img.shape)
    return gray, img, lines, control_pt1, control_pt2, per_det


def _calculate_control_signals(
    control_pt1, control_pt2, img_shape,
    speed_state, velocity, step_overhead,
    tracker1, tracker2,
    hist_dist, hist_threshold, min_val, max_val
):
    """
    Modularized control signal calculation logic.
    """
    _speed_index = {'1': 0, '2': 1, '3': 2}
    
    # Compute control signal
    error1, error2 = _compute_offset_roller(control_pt1, control_pt2, img_shape)

    # Apply velocity-based error correction based on current speed/direction state
    # direction: 1=forward, 0=stopped (mode a), -1=reverse (mode b)
    i = _speed_index.get(str(speed_state[0]), 0)
    overhead = step_overhead * speed_state[1] * velocity[i]
    if error1 is not None:
        error1 = error1 + overhead
    if error2 is not None:
        error2 = error2 + overhead
    
    # Apply tracking to handle temporary misses
    error1, is_pred1 = tracker1.update(error1)
    error2, is_pred2 = tracker2.update(error2)

    # Compute control signal over interval
    def get_signal(err):
        if err is not None: # If there is a line
            return [err >= min_val, err >= max_val]
        elif hist_dist > hist_threshold: # If there is no line (the fabric is covering the whole view), use histogram distance
            return [True, True]
        else: # If there is no line and the fabric is not covering the whole view
            return [False, False]

    control_signals = (get_signal(error1), get_signal(error2))
    return control_signals, error1, error2, is_pred1, is_pred2


def run_cam_return_signal(
        interval: Tuple[int, int] = (-3, 3),
        img_size=(128, 128),
        enhance_contrast=True,
        smooth_texture=True,
        detector_mode: DetectorMode = DetectorMode.LSD,
        hist_threshold=50000.0,
        bin_num=32,
        dist_method="L2",
        mask_thresh: int = 30,
        use_gray: bool = False,
        velocity: Optional[List] = None,
        step_overhead: float = 0.0,
        speed_state: Optional[List] = None,
):
    """
    velocity : list of 3 floats  [v1, v2, v3]
        One value per speed level (index 0/1/2 for '1'/'2'/'3').
        The sign is flipped automatically for reverse mode, and
        zeroed automatically for stopped mode (mode 'a').
    step_overhead : float
        Scalar multiplier for the velocity-based error correction.
    speed_state : list[cmd_speed_char, direction]
        Shared mutable list updated by the caller in real-time.
        speed_state[0] in {'1','2','3'}.
        speed_state[1] is an int direction multiplier:
            1  = forward (rolling),  0 = stopped (mode 'a'),  -1 = reverse (mode 'b').
    """
    if velocity is None:
        velocity = [0.0, 0.0, 0.0]
    if speed_state is None:
        speed_state = ['1', 1]

    cam = get_camera()
    if cam is None:
        return
    
    time.sleep(4)

    init_hist = None
    ret, init_frame, _ = cam.read()
    if ret and init_frame is not None:
        init_gray, init_img = preprocessing_sewer(
             init_frame, 
             img_size=img_size, 
             crop_roi=DEFAULT_CROP_ROI,
             enhance_contrast=enhance_contrast, 
             smooth_texture=smooth_texture, 
        )
        init_target = init_gray if use_gray else init_img
        init_hist = get_channel_histograms(init_target, bin_num, use_gray=use_gray)

    min_val, max_val = interval
    control_signals = ([False, False], [False, False])
    
    tracker1 = Kalman1DTracker(max_missed=2)
    tracker2 = Kalman1DTracker(max_missed=2)

    while True:
        ret, frame, has_new = cam.read()
        if not ret or frame is None:
            break

        if not has_new:
            time.sleep(0.005)
            yield None, None
            continue

        gray, img, lines, control_pt1, control_pt2, _ = _perform_line_detection(
             frame, img_size, enhance_contrast, smooth_texture, detector_mode, mask_thresh
        )

        current_target = gray if use_gray else img
        current_hist = get_channel_histograms(current_target, bin_num, use_gray=use_gray)
        hist_dist = compute_hist_distance(init_hist, current_hist, method=dist_method)

        control_signals, _, _, _, _ = _calculate_control_signals(
            control_pt1, control_pt2, img.shape,
            speed_state, velocity, step_overhead,
            tracker1, tracker2,
            hist_dist, hist_threshold, min_val, max_val
        )

        yield control_signals

    cam.release()
    cv.destroyAllWindows()


def run_cam_return_signal_with_img(
        interval: Tuple[int, int] = (-3, 3),
        img_size=(128, 128),
        enhance_contrast=True,
        smooth_texture=True,
        detector_mode: DetectorMode = DetectorMode.LSD,
        hist_threshold=50000.0,
        bin_num=32,
        dist_method="L2",
        mask_thresh: int = 30,
        use_gray: bool = False,
        velocity: Optional[List] = None,
        step_overhead: float = 0.0,
        speed_state: Optional[List] = None,
):
    """
    velocity : list of 3 floats  [v1, v2, v3]
        One value per speed level (index 0/1/2 for '1'/'2'/'3').
        The sign is flipped automatically for reverse mode, and
        zeroed automatically for stopped mode (mode 'a').
    step_overhead : float
        Scalar multiplier for the velocity-based error correction.
    speed_state : list[cmd_speed_char, direction]
        Shared mutable list updated by the caller in real-time.
        speed_state[0] in {'1','2','3'}.
        speed_state[1] is an int direction multiplier:
            1  = forward (rolling),  0 = stopped (mode 'a'),  -1 = reverse (mode 'b').
    """
    if velocity is None:
        velocity = [0.0, 0.0, 0.0]
    if speed_state is None:
        speed_state = ['1', 1]

    cam = get_camera()
    if cam is None:
        return

    time.sleep(4)

    init_hist = None
    ret, init_frame, _ = cam.read()
    if ret and init_frame is not None:
        init_gray, init_img = preprocessing_sewer(
             init_frame, 
             img_size=img_size, 
             crop_roi=DEFAULT_CROP_ROI,
             enhance_contrast=enhance_contrast, 
             smooth_texture=smooth_texture, 
        )
        init_target = init_gray if use_gray else init_img
        init_hist = get_channel_histograms(init_target, bin_num, use_gray=use_gray)

    min_val, max_val = interval
    control_signals = ([False, False], [False, False])
    
    tracker1 = Kalman1DTracker(max_missed=2)
    tracker2 = Kalman1DTracker(max_missed=2)

    # Setup directory for saving captured frames
    base_capture_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "captured_frames")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_capture_dir = os.path.join(base_capture_dir, current_time)
    os.makedirs(session_capture_dir, exist_ok=True)
    frame_count = 0
    cv.namedWindow("Gray (Left) vs Line Detection (Right)", cv.WINDOW_NORMAL)

    while True:
        ret, frame, has_new = cam.read()
        if not ret or frame is None:
            break

        if not has_new:
            yield None, None
            cv.waitKey(1)
            continue

        gray, img, lines, control_pt1, control_pt2, _ = _perform_line_detection(
             frame, img_size, enhance_contrast, smooth_texture, detector_mode, mask_thresh
        )

        current_target = gray if use_gray else img
        current_hist = get_channel_histograms(current_target, bin_num, use_gray=use_gray)
        hist_dist = compute_hist_distance(init_hist, current_hist, method=dist_method)

        control_signals, error1, error2, is_pred1, is_pred2 = _calculate_control_signals(
            control_pt1, control_pt2, img.shape,
            speed_state, velocity, step_overhead,
            tracker1, tracker2,
            hist_dist, hist_threshold, min_val, max_val
        )

        yield control_signals

        # Visualization
        visualize_and_save_frame_sewer(
            gray, img, lines, detector_mode,
            error1, error2, is_pred1, is_pred2, control_signals,
            session_capture_dir, frame_count
        )
        frame_count += 1

        # Exit on 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv.destroyAllWindows()
