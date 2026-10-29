from typing import Tuple, Optional, Any
import cv2 as cv
import numpy as np
import time
from collections import defaultdict
import random as rng

from utils.camera import get_camera, DEFAULT_CROP_ROI, compute_hist_distance
from utils.line_detectors.detector import (
    preprocessing_sewer,
    postprocess_lines,
    keep_lines_near_background,
    filter_dominant_angles,
    lsd,
    detect_lines,
    ensemble_detect_lines,
    DetectorMode
)
from utils.helper import (
    _get_control_point,
    _get_control_point_weighted,
    _compute_control_signal,
    tic,
    toc,
    timings,
    counts,
    get_channel_histograms,
    Kalman1DTracker,
    draw_lines
)

rng.seed(42)

def run_cam_return_signal_profiler(
        interval: Tuple[int, int] = (-15, 15),
        img_size=(128, 128),
        enhance_contrast=True,
        smooth_texture=True,
        detector_mode: DetectorMode = DetectorMode.LSD,
        hist_threshold=50000.0,
        bin_num=32,
        dist_method="L2",
        mask_thresh: int = 30,
        use_gray: bool = False,
        max_frames: int = 1000
):
    cam = get_camera()
    if cam is None:
        print("Camera not found.")
        return
    
    print("Waiting 5 seconds for camera auto-exposure to settle...")
    time.sleep(5)

    # Reset profiler
    timings.clear()
    counts.clear()

    print("Warming up JIT compiler and getting initial histogram...")
    # Initialize histogram baseline
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
    tracker1 = Kalman1DTracker(max_missed=2)
    tracker2 = Kalman1DTracker(max_missed=2)

    warmup_frames = 0
    while warmup_frames < 50:
        ret, frame, has_new = cam.read()
        if not ret or frame is None:
            break
        if not has_new:
            time.sleep(0.005)
            continue
            
        gray, img = preprocessing_sewer(
             frame, 
             img_size=img_size, 
             crop_roi=DEFAULT_CROP_ROI,
             enhance_contrast=enhance_contrast, 
             smooth_texture=smooth_texture, 
        )
        current_target = gray if use_gray else img
        current_hist = get_channel_histograms(current_target, bin_num, use_gray=use_gray)
        hist_dist = compute_hist_distance(init_hist, current_hist, method=dist_method)

        if detector_mode == DetectorMode.ENSEMBLE:
            lines, scores, _ = ensemble_detect_lines(gray)
            control_pt1, control_pt2 = _get_control_point_weighted(lines, scores, img.shape)
        else:
            lines = detect_lines(gray, detector_mode, img)
            lines = postprocess_lines(lines)
            lines = keep_lines_near_background(lines, gray, mask_thresh)
            control_pt1, control_pt2 = _get_control_point(lines, img.shape)

        error1, error2 = _compute_control_signal(control_pt1, control_pt2, img.shape)
        error1, _ = tracker1.update(error1)
        error2, _ = tracker2.update(error2)
        
        warmup_frames += 1

    # Reset profiler again to ensure absolutely pure metrics
    timings.clear()
    counts.clear()

    print("Starting profiling...")
    frame_count = 0
    t_start_total = time.time()
    
    peak_frame_times = []
    prev_total_timing_sum = 0.0

    # Re-init trackers for profiling since they could be exhausted or skewed
    tracker1 = Kalman1DTracker(max_missed=2)
    tracker2 = Kalman1DTracker(max_missed=2)

    while frame_count < max_frames:
        t_read = tic()
        ret, frame, has_new = cam.read()
        
        if not ret or frame is None:
            break
            
        if not has_new:
            continue

        toc("cam_read", t_read)

        t = tic()
        gray, img = preprocessing_sewer(
             frame, 
             img_size=img_size, 
             crop_roi=DEFAULT_CROP_ROI,
             enhance_contrast=enhance_contrast, 
             smooth_texture=smooth_texture, 
        )
        toc("preprocessing", t)

        t = tic()
        current_target = gray if use_gray else img
        current_hist = get_channel_histograms(current_target, bin_num, use_gray=use_gray)
        hist_dist = compute_hist_distance(init_hist, current_hist, method=dist_method)
        toc("histogram_dist", t)

        t = tic()
        if detector_mode == DetectorMode.ENSEMBLE:
            lines, scores, _ = ensemble_detect_lines(gray)
            control_pt1, control_pt2 = _get_control_point_weighted(lines, scores, img.shape)
        else:
            lines = detect_lines(gray, detector_mode, img)
            lines = postprocess_lines(lines)
            lines = keep_lines_near_background(lines, gray, mask_thresh)
            control_pt1, control_pt2 = _get_control_point(lines, img.shape)
        toc("detect_and_control_pt", t)

        t = tic()
        error1, error2 = _compute_control_signal(control_pt1, control_pt2, img.shape)
        error1, _ = tracker1.update(error1)
        error2, _ = tracker2.update(error2)
        toc("compute_and_track_error", t)

        t = tic()
        def get_signal(err):
            if err is not None:
                return [err >= min_val, err >= max_val]
            elif hist_dist > hist_threshold:
                return [True, True]
            else:
                return [False, False]

        control_signals = (get_signal(error1), get_signal(error2))
        toc("signal_interval_check", t)
        
        current_sum = sum(timings.values())
        frame_timing_sum = current_sum - prev_total_timing_sum
        prev_total_timing_sum = current_sum
        peak_frame_times.append(frame_timing_sum)

        frame_count += 1

    t_end_total = time.time()

    cam.release()
    cv.destroyAllWindows()

    total_time = t_end_total - t_start_total
    total_time_sum = sum(timings.values())
    fps = frame_count / total_time if total_time > 0 else 0
    print(f"Total Frames Processed: {frame_count}")
    print(f"Total Time: {total_time:.4f} seconds")
    print(f"Total Time (sum of subprocesses): {total_time_sum:.4f} seconds")
    print(f"Average FPS: {fps:.2f}")
    if peak_frame_times:
        max_peak = max(peak_frame_times)
        max_peak_idx = peak_frame_times.index(max_peak)
        min_peak = min(peak_frame_times)
        min_peak_idx = peak_frame_times.index(min_peak)
        print(f"Lowest Subprocess Sum: {min_peak:.6f} seconds (occurred at Frame {min_peak_idx + 1})")
        print(f"Peak Subprocess Sum:   {max_peak:.6f} seconds (occurred at Frame {max_peak_idx + 1})\n")
    else:
        print()

    print("--- Subprocess Average Times (seconds) ---")
    for name, sum_time in timings.items():
        avg_time = sum_time / counts[name] # keep in seconds
        print(f"{name:25s}: {avg_time:.6f} s")


def tune_histogram_threshold(num_bins: int = 32, use_gray: bool = False):
    cam = get_camera()
    if cam is None:
        print("Camera not found.")
        return None

    print("Waiting 5 seconds for camera auto-exposure to settle...")
    time.sleep(5.0)

    # Drain stale frames and grab a fresh stable one
    print("Capturing initial background image...")
    init_frame = None
    for _ in range(30):
        ret, frame, has_new = cam.read()
        if ret and frame is not None and has_new:
            init_frame = frame
            break
        time.sleep(0.05)

    if init_frame is None:
        print("Failed to capture a stable initial frame.")
        cam.release()
        return None

    init_gray, init_img = preprocessing_sewer(init_frame, img_size=(128, 128))
    init_target = init_gray if use_gray else init_img
    
    init_hist = get_channel_histograms(init_target, num_bins, use_gray=use_gray)
    print(f"Background frame captured (shape={init_target.shape}).")

    print("\nWaiting 5 seconds... Please place the object in camera view.")
    time.sleep(5.0)

    # Drain stale frames and grab a fresh stable one
    print("Capturing next scene with object...")
    next_frame = None
    for _ in range(30):
        ret, frame, has_new = cam.read()
        if ret and frame is not None and has_new:
            next_frame = frame
            break
        time.sleep(0.05)

    if next_frame is None:
        print("Failed to capture next frame.")
        cam.release()
        return None

    next_gray, next_img = preprocessing_sewer(next_frame, img_size=(128, 128))
    next_target = next_gray if use_gray else next_img

    next_hist = get_channel_histograms(next_target, num_bins, use_gray=use_gray)
    print(f"Object frame captured (shape={next_target.shape}).\n")

    # --- Metrics on normalized histograms ---
    dist_L2     = float(np.linalg.norm(init_hist - next_hist))
    dist_L1     = float(np.linalg.norm(init_hist - next_hist, ord=1))
    norm1       = np.linalg.norm(init_hist)
    norm2       = np.linalg.norm(next_hist)
    dist_cosine = float(1.0 - np.dot(init_hist, next_hist) / (norm1 * norm2)) if (norm1 > 0 and norm2 > 0) else 1.0

    # --- Chi-squared distance on raw (unnormalized) histograms for extra sensitivity ---
    if use_gray:
        raw_init = cv.calcHist([init_target], [0], None, [num_bins], [0, 256]).flatten().astype(np.float32)
        raw_next = cv.calcHist([next_target], [0], None, [num_bins], [0, 256]).flatten().astype(np.float32)
    else:
        raw_init = np.array([cv.calcHist([init_target], [i], None, [num_bins], [0, 256]).flatten() for i in range(3)]).flatten().astype(np.float32)
        raw_next = np.array([cv.calcHist([next_target], [i], None, [num_bins], [0, 256]).flatten() for i in range(3)]).flatten().astype(np.float32)
        
    denom = raw_init + raw_next
    denom[denom == 0] = 1  # avoid division by zero
    dist_chi2 = float(np.sum((raw_init - raw_next) ** 2 / denom))

    print("--- Histogram Distance Results (normalized histograms) ---")
    print(f"  L2  distance : {dist_L2:.6f}   (typical range: 0 – ~1.0)")
    print(f"  L1  distance : {dist_L1:.6f}   (typical range: 0 – ~2.0)")
    print(f"  Cosine dist  : {dist_cosine:.6f}   (range: 0=identical, 1=orthogonal)")
    print(f"\n--- Raw Chi-Squared Distance (unnormalized) ---")
    print(f"  Chi2 distance: {dist_chi2:.2f}   (higher = more different; scale ~1000s)")

    if init_target.shape == next_target.shape:
        combined_debug = cv.hconcat([init_target, next_target])
        # Zoom for easier viewing
        zoomed_debug = cv.resize(combined_debug, None, fx=2.0, fy=2.0, interpolation=cv.INTER_NEAREST)
        cv.imshow("Background (Left) vs Object Frame (Right)", zoomed_debug)
    else:
        cv.imshow("Background", init_target)
        cv.imshow("Object Frame", next_target)
        
    print("Press any key on the pop-up image window to close and finish...")
    cv.waitKey(0)

    cam.release()
    cv.destroyAllWindows()
    return {"L2": dist_L2, "L1": dist_L1, "cosine": dist_cosine, "chi2": dist_chi2}


def run_detector_missing_frame_profiler(
        execution_time: float = 60.0,
        img_size=(128, 128),
        enhance_contrast=True,
        smooth_texture=True,
        use_canny=False,
        square_contrast=True,
        mask_thresh: int = 30,
):
    cam = get_camera()
    if cam is None:
        print("Camera not found.")
        return

    print("Waiting 2 seconds for camera auto-exposure to settle...")
    time.sleep(2.0)

    print(f"Starting missing frame profiling for {execution_time} seconds...")
    
    total_frames = 0
    missed_frames = 0
    
    start_time = time.time()
    while time.time() - start_time < execution_time:
        ret, frame, has_new = cam.read()
        
        if not ret or frame is None:
            break
            
        if not has_new:
            time.sleep(0.005)
            continue
            
        gray, img = preprocessing_sewer(
             frame, 
             img_size=img_size, 
             crop_roi=DEFAULT_CROP_ROI,
             enhance_contrast=enhance_contrast, 
             smooth_texture=smooth_texture, 
             # use_canny=use_canny,
             square_contrast=square_contrast
        )
        lines = detect_lines(gray, DetectorMode.LSD)
        lines = postprocess_lines(lines)
        lines = keep_lines_near_background(lines, gray, mask_thresh)
        control_pt1, control_pt2 = _get_control_point(lines, img.shape)
        
        total_frames += 1
        if control_pt1 is not None:
            print(f"Frame {total_frames}: Line detected!")
        else:
            print(f"Frame {total_frames}: Frame missed!")
            missed_frames += 1
            
        # Visualization
        drawn_img = img.copy()
        if lines is not None:
            lines = lines.astype(np.float32)
            drawn_img = draw_lines(drawn_img, lines, DetectorMode.LSD)
            
        if control_pt1 is not None:
            cv.circle(drawn_img, control_pt1, 1, (0, 0, 255), -1)
            
        zoomed_vis = cv.resize(drawn_img, None, fx=3.0, fy=3.0, interpolation=cv.INTER_NEAREST)
        cv.imshow("Missing Frame Profiler (ROI)", zoomed_vis)
        
        # Exit on 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            print("Profiling interrupted by user.")
            break

    cam.release()
    cv.destroyAllWindows()
    
    if total_frames > 0:
        missing_percentage = (missed_frames / total_frames) * 100
    else:
        missing_percentage = 0.0
        
    print("\n--- Missing Frame Profiling Results ---")
    print(f"Execution Time:     {execution_time:.2f} seconds")
    print(f"Total Frames:       {total_frames}")
    print(f"Missed Frames:      {missed_frames}")
    print(f"Missing Percentage: {missing_percentage:.2f}%")


def show_raw_camera():
    cam = get_camera()
    if cam is None:
        print("Camera not found.")
        return

    print("Showing raw camera feed. Press 'q' to exit.")
    time.sleep(0.05)

    while True:
        ret, frame, has_new = cam.read()
        if not ret or frame is None:
            break

        if not has_new:
            time.sleep(0.005)
            continue

        cv.imshow("Raw Camera Feed", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    # print("Running profiler for run_cam_return_signal...")
    # run_cam_return_signal_profiler(max_frames=100)
    # print("Running threshold tuner...")
    # tune_histogram_threshold()
    # print("Running missing frame profiler...")
    # run_detector_missing_frame_profiler(execution_time=60.0)
    print("Running raw camera feed...")
    show_raw_camera()