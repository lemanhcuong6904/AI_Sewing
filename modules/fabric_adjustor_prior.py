"""
modules/fabric_adjustor_prior.py
---------------------------------
DexiNed-based prior-mask fabric adjustment pipeline.

All stateless helper functions (apply_morphology, _match_masks, etc.) live in
utils/mask_helpers.py and are re-exported here for backward compatibility.
"""

import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Tuple
import cv2 as cv
import numpy as np

from utils.camera import get_camera
from utils.line_detectors.detector import DetectorMode, preprocessing_adjustor, preprocessing_adjustor_batch
from utils.edge_detectors.tensorrt_inference import infer_dexined, load_dexined, infer_dexined_batch

# ---------------------------------------------------------------------------
# Re-export all stateless helpers so existing callers (evaluate_adjustor.py,
# test_mask_matching.py, etc.) continue to work without any import changes.
# ---------------------------------------------------------------------------
from utils.mask_helpers import (           # noqa: F401  (intentional re-export)
    apply_morphology,
    _get_moments_stats,
    _get_bbox_stats,
    _feature_matching,
    _contour_matching,
    _match_masks,
    _filter_and_mean,
    _visualize_prior_adjustment,
)
from utils.constants import DEFAULT_X_BASIS, DEFAULT_Y_BASIS, AXIS_LENGTH_PX, TRANSLATION_DEADBAND_CM, PX_PER_CM_X, PX_PER_CM_Y
from utils.viz_adjustor import draw_adjustor_overlay


# ===========================================================================
# PriorMaskAdjustor Class
# ===========================================================================

class PriorMaskAdjustor:
    """
    OOP wrapper for the DexiNed-based prior mask fabric adjustment pipeline.

    Eagerly loads the model and generates the prior mask on construction.
    Use run(visualize=False) as a generator to stream control signals.

    Usage:
        adjustor = PriorMaskAdjustor(prior_mask_path="mask_priors/1.png", ...)
        for control_signals in adjustor.run(visualize=True):
            ...  # None during cooldown/accumulation, else ((tx,ty),(r1,r2))
    """

    def __init__(
        self,
        prior_mask_path: str,
        img_size=None,
        x_basis: float = DEFAULT_X_BASIS,
        y_basis: float = DEFAULT_Y_BASIS,
        enhance_contrast: bool = False,
        smooth_texture: bool = True,
        match_method: str = 'moments',
        morphology: str = 'fill',
        velocity: Optional[List] = None,
        speed_state: Optional[List] = None,
        batch_size: int = 5,
        cooldown_time: float = 0.0,
        device: str = 'cuda',
        crop_roi: Optional[tuple] = None,
    ):
        self.prior_mask_path = prior_mask_path
        self.img_size = img_size
        self.x_basis = x_basis
        self.y_basis = y_basis
        self.enhance_contrast = enhance_contrast
        self.smooth_texture = smooth_texture
        self.match_method = match_method
        self.morphology = morphology
        self.velocity = velocity if velocity is not None else [0.0, 0.0, 0.0]
        self.speed_state = speed_state if speed_state is not None else ['1', 1]
        self.batch_size = batch_size
        self.cooldown_time = cooldown_time
        self.device = device
        self.crop_roi = crop_roi

        # Eagerly load model and generate prior mask
        print("Loading DexiNed model...")
        self.model = load_dexined(device=self.device)
        print(f"Generating prior mask from {prior_mask_path}...")
        self.prior_mask, self.prior_img = self._load_prior_mask()

    def _load_prior_mask(self):

        path = self.prior_mask_path
        if not os.path.exists(path) and not os.path.isabs(path):
            alt_path = os.path.join("mask_priors", path)
            if os.path.exists(alt_path):
                path = alt_path

        prior_img_raw = cv.imread(path)
        if prior_img_raw is None:
            raise FileNotFoundError(f"Could not read prior mask image from: {self.prior_mask_path} (also checked {path})")
        _, img, offset = preprocessing_adjustor(
            prior_img_raw, img_size=self.img_size, crop_roi=self.crop_roi,
            enhance_contrast=self.enhance_contrast, smooth_texture=self.smooth_texture,
        )
        prior_mask = infer_dexined(self.model, img, device=self.device)
        prior_mask = apply_morphology(prior_mask, method=self.morphology)
        return prior_mask, img

    def _process_batch(
        self,
        frame_buffer: List[np.ndarray],
        frame_count: int = 0,
        session_dir: Optional[str] = None,
        visualize: bool = False,
    ) -> Tuple[List[Tuple[float, float, float, float]], List[np.ndarray], List[np.ndarray]]:
        """
        Executes parallel preprocessing, batched DexiNed edge inference, and threaded mask matching.

        Parameters:
            frame_buffer: List[np.ndarray]
                List of raw camera BGR frames (each dtype: uint8, shape: [H, W, 3]).
            frame_count: int
                Global frame index used for labeling saved visualization files.
            session_dir: Optional[str]
                Directory path to save debug visual overlays if `visualize` is True.
            visualize: bool
                Whether to render and save visualization images.

        Returns:
            Tuple[List[Tuple[float, float, float, float]], List[np.ndarray], List[np.ndarray]]:
                signals: List of (tx, ty, r1, r2) tuples for each frame in pixels and degrees.
                processed_masks: List of cleaned binary masks (dtype: uint8, shape: [H_roi, W_roi]).
                img_buffer: List of preprocessed BGR ROI images (dtype: uint8, shape: [H_roi, W_roi, 3]).
        """
        # ── Step 1: Batched image cropping, resizing, and contrast enhancement ─
        _, img_buffer, offsets = preprocessing_adjustor_batch(
            frame_buffer, img_size=self.img_size, crop_roi=self.crop_roi,
            enhance_contrast=self.enhance_contrast, smooth_texture=self.smooth_texture,
        )

        # ── Step 2: Batched neural network edge prediction (DexiNed ONNX/TRT) ─
        raw_masks = infer_dexined_batch(self.model, img_buffer, device=self.device)

        # ── Step 3: Threaded morphology cleanup and mask template matching ────
        def process_single(i):
            mask = apply_morphology(raw_masks[i], method=self.morphology)
            (tx, ty), (r1, r2) = _match_masks(
                self.prior_mask, mask,
                method=self.match_method,
                x_basis=self.x_basis, y_basis=self.y_basis,
                prior_img=self.prior_img, current_img=img_buffer[i],
            )
            # Add crop offsets to translation signals to map back to original uncropped coordinates
            offset_x, offset_y = offsets[i]
            tx += offset_x
            ty += offset_y
            
            if visualize and session_dir is not None:
                _visualize_prior_adjustment(
                    frame_buffer[i], mask, self.prior_mask,
                    ((tx, ty), (r1, r2)), self.match_method,
                    x_basis=self.x_basis, y_basis=self.y_basis,
                    save_dir=session_dir, frame_count=frame_count + i, show=False,
                    offset_x=offset_x, offset_y=offset_y,
                )
            return (tx, ty, r1, r2), mask

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_single, range(len(img_buffer))))

        signals = [r[0] for r in results]
        processed_masks = [r[1] for r in results]
        return signals, processed_masks, img_buffer

    def compute_single_signal(self, cam, visualize: bool = False):
        """
        Gathers a burst of camera frames, batches them through DexiNed mask inference,
        and computes the stabilized trimmed-mean control signal for event-driven adjustment.

        Parameters:
            cam: Union[Camera, FastCamera, cv.VideoCapture]
                Active camera stream object providing `.read()` and optionally `.clear_buffer()`.
            visualize: bool
                Whether to render live prior mask matching comparison display via OpenCV.

        Returns:
            Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
                Stabilized control signals `((mean_tx, mean_ty), (mean_r1, mean_r2))` where:
                    mean_tx, mean_ty: float
                        Trimmed-mean translation displacement in image pixels.
                    mean_r1, mean_r2: float
                        Trimmed-mean rotational misalignment in degrees.
                Returns None if camera read fails or frames cannot be acquired.
        """
        if cam is None:
            return None

        # ── Step 1: Flush hardware ring buffer ────────────────────────────
        # Clean queue to discard old pre-trigger frames so only current physical state is analyzed
        cam.clear_buffer() if hasattr(cam, 'clear_buffer') else None

        target_frames = 55
        frames_processed = 0
        frame_buffer = []
        all_signals = []
        all_processed_masks = []
        all_img_buffer = []

        # ── Step 2: Accumulate and process burst frames in batches ────────
        while frames_processed < target_frames:
            ret, frame, has_new = cam.read()
            if not ret or frame is None:
                return None
            if not has_new:
                time.sleep(0.005)
                continue
            frame_buffer.append(frame)

            if len(frame_buffer) >= self.batch_size or (frames_processed + len(frame_buffer) == target_frames):
                signals, processed_masks, img_buffer = self._process_batch(
                    frame_buffer, frame_count=0, session_dir=None, visualize=visualize,
                )
                
                all_signals.extend(signals)
                all_processed_masks.extend(processed_masks)
                all_img_buffer.extend(img_buffer)
                
                frames_processed += len(frame_buffer)
                frame_buffer.clear()

        # ── Step 3: GUI visualization on main thread ──────────────────────
        if visualize and len(all_img_buffer) > 0 and len(frame_buffer) > 0:
            last_idx = -1
            tx, ty, r1, r2 = all_signals[last_idx]  # flat → nested for vis
            _visualize_prior_adjustment(
                frame_buffer[last_idx], all_processed_masks[last_idx], self.prior_mask,
                ((tx, ty), (r1, r2)), self.match_method,
                x_basis=self.x_basis, y_basis=self.y_basis,
                save_dir=None, frame_count=0, show=True,
                offset_x=self.crop_roi[0] if self.crop_roi else 0,
                offset_y=self.crop_roi[2] if self.crop_roi else 0,
            )

        # ── Step 4: Trimmed mean statistical filtering across burst ───────
        mean_tx, mean_ty, mean_r1, mean_r2 = _filter_and_mean(all_signals)
        return ((mean_tx, mean_ty), (mean_r1, mean_r2))

    def run(self, cam=None, visualize: bool = False):
        """
        Main generator. Yields control signals from the camera stream.

        Args:
            visualize: If True, show + save annotated frames.

        Yields:
            None                   — during cooldown or buffer accumulation.
            ((tx, ty), (r1, r2))  — filtered batch-mean control signals (pixels / radians).
        """
        if cam is None:
            cam = get_camera()
        if cam is None:
            return

        time.sleep(4)

        frame_buffer = []
        cooldown_end_time = 0.0
        frame_count = 0

        session_dir = None
        if visualize:
            base_capture_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "captured_frames"
            )
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            session_dir = os.path.join(base_capture_dir, current_time)
            os.makedirs(session_dir, exist_ok=True)
            cv.namedWindow("Prior Mask Adjustment", cv.WINDOW_NORMAL)

        last_yielded_tx = None
        last_yielded_ty = None

        while True:
            ret, frame, has_new = cam.read()
            if not ret or frame is None:
                break

            if not has_new:
                if visualize:
                    cv.waitKey(1)
                else:
                    time.sleep(0.005)
                yield None
                continue

            if time.time() < cooldown_end_time:
                if visualize:
                    display_img = cv.resize(frame, self.img_size) if self.img_size is not None else frame.copy()
                    draw_adjustor_overlay(display_img, (self.x_basis, self.y_basis), AXIS_LENGTH_PX)
                    cv.putText(display_img, "COOLDOWN", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv.imshow("Prior Mask Adjustment", display_img)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break
                yield None
                continue

            frame_buffer.append(frame)

            yielded_signal = None
            if len(frame_buffer) >= self.batch_size:
                signals, processed_masks, img_buffer = self._process_batch(
                    frame_buffer, frame_count=frame_count, session_dir=session_dir, visualize=visualize,
                )

                if visualize:
                    # Show the last frame of the batch in the main thread
                    last_idx = len(img_buffer) - 1
                    tx, ty, r1, r2 = signals[last_idx]
                    _visualize_prior_adjustment(
                        frame_buffer[last_idx], processed_masks[last_idx], self.prior_mask,
                        ((tx, ty), (r1, r2)), self.match_method,
                        x_basis=self.x_basis, y_basis=self.y_basis,
                        save_dir=session_dir, frame_count=frame_count + last_idx, show=True,
                        offset_x=self.crop_roi[0] if self.crop_roi else 0,
                        offset_y=self.crop_roi[2] if self.crop_roi else 0,
                    )

                mean_tx, mean_ty, mean_r1, mean_r2 = _filter_and_mean(signals)
                

                tx_cm = mean_tx / PX_PER_CM_X
                ty_cm = mean_ty / PX_PER_CM_Y
                
                yield_it = True
                if last_yielded_tx is not None and last_yielded_ty is not None:
                    if abs(tx_cm - last_yielded_tx) < TRANSLATION_DEADBAND_CM and abs(ty_cm - last_yielded_ty) < TRANSLATION_DEADBAND_CM:
                        yield_it = False
                        
                if yield_it:
                    yielded_signal = ((mean_tx, mean_ty), (mean_r1, mean_r2))
                    last_yielded_tx = tx_cm
                    last_yielded_ty = ty_cm
                    cooldown_end_time = time.time() + self.cooldown_time
                else:
                    yielded_signal = None

                frame_count += len(frame_buffer)
                frame_buffer.clear()
            else:
                if visualize:
                    display_img = cv.resize(frame, self.img_size) if self.img_size is not None else frame.copy()
                    draw_adjustor_overlay(display_img, (self.x_basis, self.y_basis), AXIS_LENGTH_PX)
                    cv.imshow("Prior Mask Adjustment", display_img)

            yield yielded_signal

            if visualize and (cv.waitKey(1) & 0xFF == ord('q')):
                break

        cam.release()
        cv.destroyAllWindows()


# ===========================================================================
# Backward-compatibility shims for main_adjustor.py
# ===========================================================================

def run_cam_return_signal(
    prior_mask_path: str,
    # img_size=(128, 128),
    x_basis=DEFAULT_X_BASIS,
    y_basis=DEFAULT_Y_BASIS,
    enhance_contrast=False,
    smooth_texture=True,
    match_method='moments',
    morphology='fill',
    velocity: Optional[List] = None,
    speed_state: Optional[List] = None,
    batch_size: int = 5,
    cooldown_time: float = 0.0,
):
    """Headless signal generator (no visualization)."""
    yield from PriorMaskAdjustor(
        prior_mask_path=prior_mask_path, # img_size=img_size,
        x_basis=x_basis, y_basis=y_basis,
        enhance_contrast=enhance_contrast, smooth_texture=smooth_texture,
        match_method=match_method, morphology=morphology,
        velocity=velocity, speed_state=speed_state,
        batch_size=batch_size, cooldown_time=cooldown_time,
    ).run(visualize=False)


def run_cam_return_signal_with_img(
    prior_mask_path: str,
    # img_size=(128, 128),
    x_basis=DEFAULT_X_BASIS,
    y_basis=DEFAULT_Y_BASIS,
    enhance_contrast=False,
    smooth_texture=True,
    match_method='moments',
    morphology='fill',
    velocity: Optional[List] = None,
    speed_state: Optional[List] = None,
    batch_size: int = 5,
    cooldown_time: float = 0.0,
):
    """Signal generator with live visualization and frame capture."""
    yield from PriorMaskAdjustor(
        prior_mask_path=prior_mask_path, # img_size=img_size,
        x_basis=x_basis, y_basis=y_basis,
        enhance_contrast=enhance_contrast, smooth_texture=smooth_texture,
        match_method=match_method, morphology=morphology,
        velocity=velocity, speed_state=speed_state,
        batch_size=batch_size, cooldown_time=cooldown_time,
    ).run(visualize=True)
