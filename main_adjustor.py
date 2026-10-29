#!/usr/bin/env python3
"""
main_adjustor.py
----------------
AI Sewing — Fabric Adjustor Entry Point

Streams control signals from the vision pipeline and sends motor commands
to the STM32 controller over UART using the format "M1,M2,M3\\n" (values in mm/deg).
  M1 = horizontal correction (tx, mm)
  M2 = vertical correction   (ty, mm)
  M3 = rotation correction   (rot, deg)
"""

from typing import Tuple, Optional
import serial
import time
import sys
try:
    import tty
    import termios
except ImportError:
    tty = None
    termios = None
import argparse
import select
import cv2 as cv
import numpy as np

from jetson_uart import get_default_port, send_and_receive
from utils.line_detectors.detector import DetectorMode
from utils.constants import (
    DEFAULT_X_BASIS, 
    DEFAULT_Y_BASIS, 
    DEFAULT_PIVOT,
    DEFAULT_PIVOT_X,
    DEFAULT_PIVOT_Y,
    U1_VECTOR,
    U2_VECTOR,
    PX_PER_CM_X, 
    PX_PER_CM_Y,
    AXIS_LENGTH_PX,
    ROTATION_DEADBAND_DEG, 
    TRANSLATION_DEADBAND_CM
)
from utils.camera import get_camera
from utils.viz_adjustor import draw_adjustor_overlay
from modules.fabric_adjustor_prior import PriorMaskAdjustor
from modules.fabric_adjustor import LineDetectionAdjustor


# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="AI Sewing Fabric Adjustor")
    parser.add_argument("--visualize", action="store_true",
                        help="Show live annotated camera feed and save frames.")
    parser.add_argument("--img-size", type=int, nargs=2, default=None, dest="img_size",
                        help="Image size passed to the detector (W H).")
    parser.add_argument("--x-basis", type=float, default=DEFAULT_X_BASIS, dest="x_basis",
                        help=f"X coordinate of the fabric corner reference point (pixels). Default: {DEFAULT_X_BASIS}")
    parser.add_argument("--y-basis", type=float, default=DEFAULT_Y_BASIS, dest="y_basis",
                        help=f"Y coordinate of the fabric corner reference point (pixels). Default: {DEFAULT_Y_BASIS}")
    parser.add_argument("--detector", type=str, default='dexined',
                        choices=['lsd', 'fld', 'edlines', 'ensemble', 'mlsd', 'dexined'],
                        help="Vision backend (default: dexined).")
    parser.add_argument("--prior-mask", type=str, default=None, dest="prior_mask",
                        help="Path to the prior mask image (required for dexined).")
    parser.add_argument("--match-method", type=str, default='contour',
                        choices=['moments', 'ecc', 'bbox', 'feature', 'contour'],
                        dest="match_method",
                        help="Mask matching algorithm (dexined only).")
    parser.add_argument("--morphology", type=str, default='fill',
                        choices=['none', 'erode', 'open', 'fill'], dest="morphology",
                        help="Morphological post-processing for edge masks.")
    parser.add_argument("--enhance-contrast", action="store_true", dest="enhance_contrast",
                        help="Enable CLAHE contrast enhancement in preprocessing.")
    parser.add_argument("--no-smooth", action="store_false", dest="smooth_texture",
                        help="Disable Gaussian texture smoothing.")
    parser.add_argument("--batch-size", type=int, default=10, dest="batch_size",
                        help="Frames to accumulate before computing the batch-mean signal (dexined).")
    parser.add_argument("--max-moderations", type=int, default=2, dest="max_moderations",
                        help="Maximum number of times to moderate a single axis before moving to the next.")
    parser.add_argument("--max-angle-moderations", type=int, default=2, dest="max_angle_moderations",
                        help="Number of steps (moderations) before angle modification stops.")
    parser.add_argument("--max-retries", type=int, default=20, dest="max_retries",
                        help="Maximum consecutive None signal retries before releasing and reporting error.")
    parser.add_argument("--crop-roi", type=int, nargs=4, default=None, dest="crop_roi",
                        help="Crop the image before processing (sx ex sy ey).")
    parser.add_argument("--continuous", action="store_true", dest="continuous",
                        help="Run the older continuous pipeline using adjustor.run() instead of the UART event-driven pipeline.")
    parser.add_argument("--cooldown-time", type=float, default=0.0, dest="cooldown_time",
                        help="Cooldown time (in seconds) between emitting signals in continuous mode.")
    parser.add_argument("--x-offset-cm", type=float, default=0.0, dest="x_offset_cm",
                        help="Offset in cm for L2/L2b phase (moves x_basis right by x_offset_cm).")
    parser.add_argument("--y-offset-cm", type=float, default=0.4, dest="y_offset_cm",
                        help="Offset in cm for L2/L2b phase (moves y_basis down by y_offset_cm).")
    parser.add_argument("--max-moderations-l2a", "--max-moderation-l2a", type=int, default=1, dest="max_moderations_l2a",
                        help="Number of moderation iterations for L2a phase (default: 1). The rest (max_moderations - max_moderations_l2a) is for L2b.")
    parser.add_argument("--x-offset-cm-l2a", type=float, default=-0.5, dest="x_offset_cm_l2a",
                        help="Additional X offset in cm for L2a phase modified on top of L2b (default: -0.5).")
    parser.add_argument("--y-offset-cm-l2a", type=float, default=0.0, dest="y_offset_cm_l2a",
                        help="Additional Y offset in cm for L2a phase modified on top of L2b (default: 0.0).")
    parser.add_argument("--rot-alignment", type=str, default="horizontal", choices=["horizontal", "vertical"], dest="rot_alignment",
                        help="Which alignment's rotation angle to use from rot_angles: 'horizontal' (rot_angles[0]) or 'vertical' (rot_angles[1]). Default: 'horizontal'")
    parser.add_argument("--target-frames", type=int, default=150, dest="target_frames",
                        help="Number of frames to average for each signal in continuous mode or event-driven mode.")
    parser.add_argument("--pivot", type=float, nargs=2, default=None, dest="pivot",
                        help=f"Initial rotation center of the robot gripper in camera pixels (px py). Default: ({DEFAULT_PIVOT_X}, {DEFAULT_PIVOT_Y})")
    parser.add_argument("--no-compensate-rotation", action="store_false", dest="compensate_rotation",
                        help="Disable rotation compensation in translation vector calculation.")
    # UART
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port to the STM32 (default: auto-detect).")
    parser.add_argument("--baud", type=int, default=115200,
                        help="UART baud rate (default: 115200).")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Print UART commands without opening a serial port.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# FabricAdjustorApp (Modular Subsystem Orchestrator)
# ---------------------------------------------------------------------------

class FabricAdjustorApp:
    """Modular application orchestrator for the AI Sewing Fabric Adjustor subsystem.

    Integrates camera image acquisition, serial UART communication with the
    STM32 motor driver, vision-based geometric line/edge estimation backends,
    and a multi-phase sequential moderation state machine (L1, L2a, L2b).

    Architecture Overview:
        - Vision Subsystem: Interfaces with CSI/USB cameras (via GStreamer or V4L2)
          and runs classical line detection (LSD, FAST_LSD), deep learning line
          estimation (M-LSD), or prior template mask matching (DexiNed).
        - Actuation Subsystem: Communicates over UART with the STM32 stepper motor
          driver using formatted ASCII displacement vectors "M1, M2, M3\\n" (mm/deg)
          and clamp release commands ("k\\n").
        - Phase State Machine: Manages multi-step sequential alignment:
            * L1: Initial rough corner alignment against reference basis (x_basis_l1, y_basis_l1).
            * L2a: Intermediate alignment with fine offset adjustment (x_basis_l2a, y_basis_l2a).
            * L2b: Final sewing line alignment with target offset (x_basis_l2b, y_basis_l2b).
        - Sequential Moderation: Enforces deadband thresholds and moderates translation
          before rotation to prevent fabric buckling and oscillatory overshoot.
    """

    def __init__(
        self,
        img_size: Optional[Tuple[int, int]] = None,
        x_basis: Optional[float] = None,
        y_basis: Optional[float] = None,
        visualize: bool = False,
        detector_mode: DetectorMode = DetectorMode.DEXINED,
        prior_mask_path: Optional[str] = None,
        match_method: str = 'contour',
        morphology: str = 'fill',
        enhance_contrast: bool = False,
        smooth_texture: bool = True,
        batch_size: int = 20,
        max_moderations: int = 2,
        max_angle_moderations: int = 2,
        max_retries: int = 20,
        crop_roi: Optional[Tuple[int, int, int, int]] = None,
        continuous: bool = False,
        cooldown_time: float = 0.0,
        port: Optional[str] = None,
        baud: int = 115200,
        dry_run: bool = False,
        x_offset_cm: float = 0.0,
        y_offset_cm: float = 0.4,
        max_moderations_l2a: int = 1,
        x_offset_cm_l2a: float = -0.5,
        y_offset_cm_l2a: float = 0.0,
        rot_alignment: str = 'horizontal',
        target_frames: int = 150,
        pivot: Optional[Tuple[float, float]] = None,
        compensate_rotation: bool = True,
        **kwargs,
    ):
        """Initialize the FabricAdjustorApp instance with pipeline configuration.

        Args:
            img_size: Optional[Tuple[int, int]]
                Target resolution (width, height) to resize camera frames to before
                detector processing. If None, the camera's native resolution is used.
            x_basis: Optional[float]
                X-coordinate (pixels) of the target fabric corner in camera frame.
                Defaults to DEFAULT_X_BASIS (247.0).
            y_basis: Optional[float]
                Y-coordinate (pixels) of the target fabric corner in camera frame.
                Defaults to DEFAULT_Y_BASIS (203.0).
            visualize: bool
                If True, opens OpenCV live visualization windows with HUD overlays.
            detector_mode: DetectorMode
                Line detector backend algorithm (e.g., DEXINED, MLSD, LSD, FAST_LSD).
            prior_mask_path: Optional[str]
                Filesystem path to pre-captured reference binary mask or image.
                Required when detector_mode is DEXINED or non-orthogonal template matching.
            match_method: str
                Algorithm used for prior mask registration ('contour', 'moments', 'ecc').
            morphology: str
                Morphological post-processing operation applied to edge masks ('fill', 'open').
            enhance_contrast: bool
                If True, applies CLAHE contrast equalization during image preprocessing.
            smooth_texture: bool
                If True, applies Gaussian filtering to attenuate fabric weave texture noise.
            batch_size: int
                Number of consecutive video frames grouped per detection inference batch.
            max_moderations: int
                Maximum number of sequential correction iterations allowed per phase
                before releasing the fabric clamp with 'k'.
            max_angle_moderations: int
                Maximum number of iterations allowed specifically for rotation angle adjustment.
            max_retries: int
                Maximum consecutive None vision signal retries before releasing the fabric
                and dispatching an error code ('e1' or 'e2').
            crop_roi: Optional[Tuple[int, int, int, int]]
                Region of interest bounding box (sx, ex, sy, ey) in pixels to restrict
                edge detection. If None, computed automatically around the active basis.
            continuous: bool
                If True, runs continuous streaming mode via adjustor.run(); if False,
                runs the event-driven mode responding to STM32 UART triggers.
            cooldown_time: float
                Cooldown delay in seconds between emitting consecutive motor commands
                in continuous streaming mode.
            port: Optional[str]
                Serial port device path (e.g. '/dev/ttyUSB0', '/dev/ttyTHS1').
                If None, automatically detected via get_default_port().
            baud: int
                UART serial communication baud rate (default: 115200).
            dry_run: bool
                If True, disables physical UART communication and logs motor commands
                to console for testing.
            x_offset_cm: float
                Physical horizontal shift (cm) added to x_basis for Phase 2 (L2b).
            y_offset_cm: float
                Physical vertical shift (cm) added to y_basis for Phase 2 (L2b).
            max_moderations_l2a: int
                Moderation iteration budget allocated to Phase 2a (L2a). The remainder
                (max_moderations - max_moderations_l2a) is allocated to L2b.
            x_offset_cm_l2a: float
                Additional horizontal physical shift (cm) for L2a relative to L2b.
            y_offset_cm_l2a: float
                Additional vertical physical shift (cm) for L2a relative to L2b.
            rot_alignment: str
                Specifies which alignment rotation angle to use from rot_angles:
                'horizontal' (index 0) or 'vertical' (index 1).
            target_frames: int
                Number of video frames accumulated per detection cycle.
            pivot: Optional[Tuple[float, float]]
                Initial rotation center (px, py) in camera pixels of the robot gripper.
                Defaults to DEFAULT_PIVOT (385.0, 83.0). Reset at the start of L1 and L2.
            **kwargs:
                Additional keyword arguments absorbed for forward compatibility.

        Raises:
            ValueError: If rot_alignment is not 'horizontal' or 'vertical'.
        """
        if rot_alignment not in ('horizontal', 'vertical'):
            raise ValueError("rot_alignment must be 'horizontal' or 'vertical'")

        # Configuration
        self.img_size = img_size
        self.visualize = visualize
        self.detector_mode = detector_mode
        self.prior_mask_path = prior_mask_path
        self.match_method = match_method
        self.morphology = morphology
        self.enhance_contrast = enhance_contrast
        self.smooth_texture = smooth_texture
        self.batch_size = batch_size
        self.max_moderations = max_moderations
        self.max_angle_moderations = max_angle_moderations
        self.max_retries = max_retries
        self.continuous = continuous
        self.cooldown_time = cooldown_time
        self.port = port
        self.baud = baud
        self.dry_run = dry_run
        self.x_offset_cm = x_offset_cm
        self.y_offset_cm = y_offset_cm
        self.x_offset_cm_l2a = x_offset_cm_l2a
        self.y_offset_cm_l2a = y_offset_cm_l2a
        self.rot_alignment = rot_alignment
        self.target_frames = target_frames
        self.rot_index = 0 if rot_alignment == 'horizontal' else 1
        self.initial_pivot = tuple(pivot) if pivot is not None else DEFAULT_PIVOT
        self.compensate_rotation = compensate_rotation

        # Target Coordinates for L1, L2b, and L2a
        self.active_x_basis = x_basis if x_basis is not None else DEFAULT_X_BASIS
        self.active_y_basis = y_basis if y_basis is not None else DEFAULT_Y_BASIS

        # Target Coordinates for L1, L2b, and L2a in vector form
        base_basis = np.array([self.active_x_basis, self.active_y_basis], dtype=np.float64)
        self.x_basis_l1 = float(base_basis[0])
        self.y_basis_l1 = float(base_basis[1])

        offset_l2b = np.array([self.x_offset_cm * PX_PER_CM_X, self.y_offset_cm * PX_PER_CM_Y], dtype=np.float64)
        basis_l2b = base_basis + offset_l2b
        self.x_basis_l2b = float(basis_l2b[0])
        self.y_basis_l2b = float(basis_l2b[1])

        offset_l2a = np.array([self.x_offset_cm_l2a * PX_PER_CM_X, self.y_offset_cm_l2a * PX_PER_CM_Y], dtype=np.float64)
        basis_l2a = basis_l2b + offset_l2a
        self.x_basis_l2a = float(basis_l2a[0])
        self.y_basis_l2a = float(basis_l2a[1])

        # Budget Partitioning between L2a and L2b
        clamped_l2a = max(0, min(max_moderations_l2a, self.max_moderations))
        if clamped_l2a != max_moderations_l2a:
            print(f"[CONFIG] max_moderations_l2a ({max_moderations_l2a}) clamped to {clamped_l2a} (max_moderations={self.max_moderations})")
        self.max_moderations_l2a = clamped_l2a
        self.max_moderations_l2b = self.max_moderations - self.max_moderations_l2a

        # ROI Bounding Box
        if crop_roi is None:
            self.crop_roi = (
                max(0, int(self.active_x_basis - 3.0 * PX_PER_CM_X)),
                int(self.active_x_basis + 7 * PX_PER_CM_X),
                max(0, int(self.active_y_basis - 4 * PX_PER_CM_Y)),
                int(self.active_y_basis + 2 * PX_PER_CM_Y)
            )
        else:
            self.crop_roi = crop_roi

        # Runtime State
        self.current_state = 'l1'
        self.moderation_count = 0
        self.is_m1_m2_modified = False
        self.is_m3_modified = False
        self.trigger_compute = False
        self.retry_count = 0
        self.keyboard_buffer = ""

        # Hardware & Subsystem Instances
        self.ser: Optional[serial.Serial] = None
        self.cam = None
        self.adjustor = None
        self.fd = None
        self.old_settings = None

    # -----------------------------------------------------------------------
    # Hardware & Subsystem Lifecycle
    # -----------------------------------------------------------------------

    def _init_serial(self) -> None:
        """Initialize serial connection to the STM32 motor driver.

        Opens the UART device port with 8 data bits, no parity, 1 stop bit (8N1),
        and 1.0s read timeout. Waits 2 seconds after connection establishment to
        allow the STM32 board to complete reset cycles triggered by serial DTR toggling.

        In dry-run mode, UART initialization is skipped and motor commands are logged.
        """
        if self.dry_run:
            print("[dry-run] UART disabled — commands will be printed only.")
            return

        port = self.port or get_default_port()
        print(f"Connecting to {port} at {self.baud} baud...")
        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=self.baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1.0,
            )
            time.sleep(2)   # allow STM32 to reset after DTR toggle
            print(f"Connected to {port}.")
        except serial.SerialException as e:
            print(f"[UART] Cannot open {port}: {e}")
            print("  Tip: sudo chmod 666 <port>")
            sys.exit(1)

    def _close_serial(self) -> None:
        """Close the serial connection cleanly and release port resources."""
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
            print("Serial port closed.")

    def _init_vision(self) -> None:
        """Instantiate camera device and select the appropriate adjustor backend.

        Initializes the video capture device via get_camera() and configures one of:
          - PriorMaskAdjustor: Used when detector_mode is DEXINED; aligns edges against
            a pre-computed template mask.
          - NonOrthoLineAdjustor: Used when non-orthogonal template guidance is provided.
          - LineDetectionAdjustor: Standard classical or deep line detection adjustor.

        Raises:
            ValueError: If detector_mode is DEXINED but prior_mask_path is None.
        """
        self.cam = get_camera()

        if self.detector_mode == DetectorMode.DEXINED:
            if self.prior_mask_path is None:
                raise ValueError("--prior-mask is required when using the dexined detector.")
            self.adjustor = PriorMaskAdjustor(
                prior_mask_path=self.prior_mask_path,
                img_size=self.img_size,
                x_basis=self.active_x_basis,
                y_basis=self.active_y_basis,
                enhance_contrast=self.enhance_contrast,
                smooth_texture=self.smooth_texture,
                match_method=self.match_method,
                morphology=self.morphology,
                batch_size=self.batch_size,
                crop_roi=self.crop_roi,
                cooldown_time=self.cooldown_time,
            )
        else:
            if self.prior_mask_path is not None:
                from modules.fabric_adjustor import NonOrthoLineAdjustor
                self.adjustor = NonOrthoLineAdjustor(
                    prior_img_path=self.prior_mask_path,
                    img_size=self.img_size,
                    x_basis=self.active_x_basis,
                    y_basis=self.active_y_basis,
                    enhance_contrast=self.enhance_contrast,
                    smooth_texture=self.smooth_texture,
                    detector_mode=self.detector_mode,
                    batch_size=self.batch_size,
                    crop_roi=self.crop_roi,
                    cooldown_time=self.cooldown_time,
                    rot_alignment=self.rot_alignment,
                    pivot=self.initial_pivot,
                    compensate_rotation=self.compensate_rotation,
                )
            else:
                self.adjustor = LineDetectionAdjustor(
                    img_size=self.img_size,
                    x_basis=self.active_x_basis,
                    y_basis=self.active_y_basis,
                    enhance_contrast=self.enhance_contrast,
                    smooth_texture=self.smooth_texture,
                    detector_mode=self.detector_mode,
                    batch_size=self.batch_size,
                    crop_roi=self.crop_roi,
                    cooldown_time=self.cooldown_time,
                    rot_alignment=self.rot_alignment,
                    pivot=self.initial_pivot,
                    compensate_rotation=self.compensate_rotation,
                )

    def _init_terminal(self) -> None:
        """Configure non-blocking cbreak terminal mode for interactive keystrokes.

        Saves current termios attributes and enables cbreak mode on POSIX systems
        so single characters can be read from sys.stdin immediately without waiting
        for a newline. On non-POSIX systems, this operation is gracefully bypassed.
        """
        self.fd = sys.stdin.fileno() if hasattr(sys.stdin, 'fileno') else None
        self.old_settings = termios.tcgetattr(self.fd) if (termios is not None and self.fd is not None) else None
        if tty is not None and self.fd is not None:
            tty.setcbreak(self.fd)

    def _restore_terminal(self) -> None:
        """Restore original terminal termios settings to preserve shell usability."""
        if termios is not None and self.fd is not None and self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    # -----------------------------------------------------------------------
    # Phase & State Machine Transition Logic
    # -----------------------------------------------------------------------

    def switch_to_phase(self, phase: str, reset_count: bool = True) -> None:
        """Switch active adjustment phase, update target basis coordinates, and manage iteration budgets.

        Transitions the state machine between 'l1', 'l2a', and 'l2b', adjusting the
        backend adjustor's reference coordinates (x_basis, y_basis) to match the
        designated target for that phase:
          - 'l1': Uses base reference coordinates (x_basis_l1, y_basis_l1) with full budget.
          - 'l2' / 'l2a': If max_moderations_l2a > 0, switches to L2a coordinates
            (x_basis_l2a, y_basis_l2a); otherwise falls back directly to L2b.
          - 'l2b': Uses offset target coordinates (x_basis_l2b, y_basis_l2b).

        Args:
            phase: Target phase identifier ('l1', 'l2', 'l2a', 'l2b'). Case-insensitive.
            reset_count: If True, resets iteration counter and deadband flags for the new phase.
        """
        p = phase.lower()
        if p == 'l1':
            self.current_state = 'l1'
            self.adjustor.x_basis = self.x_basis_l1
            self.adjustor.y_basis = self.y_basis_l1
            if reset_count:
                self.moderation_count = 0
                self.is_m1_m2_modified = False
                self.is_m3_modified = False
                if hasattr(self.adjustor, 'reset_pivot'):
                    self.adjustor.reset_pivot(self.initial_pivot)
                    print(f"[PIVOT] Reset to starting point: ({self.adjustor.pivot[0]:.1f}, {self.adjustor.pivot[1]:.1f}) px")
            print(f"\n[PHASE] Switched to L1 (basis=({self.adjustor.x_basis:.1f}, {self.adjustor.y_basis:.1f}), budget={self.max_moderations})")
        elif p in ('l2', 'l2a'):
            if self.max_moderations_l2a > 0:
                self.current_state = 'l2a'
                self.adjustor.x_basis = self.x_basis_l2a
                self.adjustor.y_basis = self.y_basis_l2a
                if reset_count:
                    self.moderation_count = 0
                    self.is_m1_m2_modified = False
                    self.is_m3_modified = False
                    if hasattr(self.adjustor, 'reset_pivot'):
                        self.adjustor.reset_pivot(self.initial_pivot)
                        print(f"[PIVOT] Reset to starting point: ({self.adjustor.pivot[0]:.1f}, {self.adjustor.pivot[1]:.1f}) px")
                print(f"\n[PHASE] Switched to L2a (basis=({self.adjustor.x_basis:.1f}, {self.adjustor.y_basis:.1f}), budget={self.max_moderations_l2a}/{self.max_moderations})")
            else:
                self.current_state = 'l2b'
                self.adjustor.x_basis = self.x_basis_l2b
                self.adjustor.y_basis = self.y_basis_l2b
                if reset_count:
                    self.moderation_count = 0
                    self.is_m1_m2_modified = False
                    self.is_m3_modified = False
                    if hasattr(self.adjustor, 'reset_pivot'):
                        self.adjustor.reset_pivot(self.initial_pivot)
                        print(f"[PIVOT] Reset to starting point: ({self.adjustor.pivot[0]:.1f}, {self.adjustor.pivot[1]:.1f}) px")
                print(f"\n[PHASE] Switched to L2b (L2a budget=0, basis=({self.adjustor.x_basis:.1f}, {self.adjustor.y_basis:.1f}), budget={self.max_moderations_l2b}/{self.max_moderations})")
        elif p == 'l2b':
            self.current_state = 'l2b'
            self.adjustor.x_basis = self.x_basis_l2b
            self.adjustor.y_basis = self.y_basis_l2b
            if reset_count:
                # Direct switch to L2b allocates the remaining L2b portion
                self.moderation_count = self.max_moderations_l2a
                self.is_m1_m2_modified = False
                self.is_m3_modified = False
                if hasattr(self.adjustor, 'reset_pivot'):
                    self.adjustor.reset_pivot(self.initial_pivot)
                    print(f"[PIVOT] Reset to starting point: ({self.adjustor.pivot[0]:.1f}, {self.adjustor.pivot[1]:.1f}) px")
            print(f"\n[PHASE] Switched to L2b (basis=({self.adjustor.x_basis:.1f}, {self.adjustor.y_basis:.1f}), budget={self.max_moderations_l2b}/{self.max_moderations})")

    def release_fabric(self) -> None:
        """Release fabric clamp and transition active phase to completed state.

        Sends the 'k' clamp-release command to the STM32 controller over UART,
        resets moderation counters and deadband tracking flags, transitions
        current_state to 'l1_done' or 'l2_done', and resets adjustor basis
        coordinates to L1 defaults and pivot to starting point.
        """
        self.is_m1_m2_modified = False
        self.is_m3_modified = False
        self.moderation_count = 0
        time.sleep(1)
        if self.ser is not None:
            # send_and_receive(self.ser, "50, 0, 0")
            send_and_receive(self.ser, 'k')
        else:
            print("[dry-run] Would send: 'k'")
        print("\nReleased!")

        if hasattr(self.adjustor, 'reset_pivot'):
            self.adjustor.reset_pivot(self.initial_pivot)

        if self.current_state in ('l2', 'l2a', 'l2b'):
            self.current_state = 'l2_done'
            self.adjustor.x_basis = self.x_basis_l1
            self.adjustor.y_basis = self.y_basis_l1
        else:
            self.current_state = 'l1_done'

    # -----------------------------------------------------------------------
    # Signal Processing & Motor Actuation
    # -----------------------------------------------------------------------

    def process_control_signals(
        self,
        control_signals: Optional[Tuple[Tuple[float, float], Tuple[float, float]]]
    ) -> None:
        """Convert vision control signals to physical motor commands and dispatch to STM32.

        Translates raw pixel displacements (tx_px, ty_px) to physical millimeters (m1_mm, m2_mm)
        using calibrated px/cm ratios (PX_PER_CM_X, PX_PER_CM_Y), and extracts the rotation
        correction angle (m3_deg).

        Control Workflow:
          1. Workspace Limits: Clamps translation displacements to [-2.0 cm, +3.0 cm] and
             rotation to [-10 deg, +10 deg] to prevent motor collisions or fabric tearing.
          2. Sequential Moderation: Enforces translation adjustment before rotation:
             - Translates if displacement exceeds TRANSLATION_DEADBAND_CM (0.05 cm).
             - Rotates if angular error exceeds ROTATION_DEADBAND_DEG (0.1 deg) within
               max_angle_moderations budget.
             - Advances iteration counter once both axes are evaluated.
          3. Command Dispatch: Formats command as "M1, M2, M3\\n" and sends to STM32 over UART.
          4. Sub-phase Transition: Automatically transitions from 'l2a' to 'l2b' when
             max_moderations_l2a iterations have been reached.
          5. Budget Exhaustion: Calls release_fabric() once max_moderations is exceeded.

        Args:
            control_signals: Optional[Tuple[Tuple[float, float], Tuple[float, float]]]
                Vision alignment tuple ((tx_px, ty_px), (rot_h_deg, rot_v_deg)).
                If None, the method returns without actuating motors.
        """
        # Guard: If stage is idle or completed, do not actuate
        if self.current_state in ('l1_done', 'l2_done', 'idle'):
            return

        # Check moderation budget exhaustion
        if self.moderation_count >= self.max_moderations:
            self.release_fabric()
            return

        if control_signals is None:
            return

        (tx_px, ty_px), rot_angles = control_signals

        # ── Step 1: Pixel to physical conversion and mechanical limits ──
        tx_cm = tx_px / PX_PER_CM_X
        ty_cm = ty_px / PX_PER_CM_Y

        # Bound physical limits to avoid motor travel collision [-2.0 cm, +3.0 cm]
        tx_cm = max(-2.0, min(3.0, tx_cm))
        ty_cm = max(-2.0, min(3.0, ty_cm))

        m1_mm = tx_cm * 10.0
        m2_mm = ty_cm * 10.0
        m3_deg = max(-10.0, min(10.0, rot_angles[self.rot_index]))

        # ── Step 2: Sequential Moderation State Machine ──
        if self.moderation_count < self.max_moderations:
            # Moderate translation first: zero out tiny sub-threshold jitter
            if not self.is_m1_m2_modified:
                if abs(tx_cm) < TRANSLATION_DEADBAND_CM:
                    m1_mm = 0.0
                if abs(ty_cm) < TRANSLATION_DEADBAND_CM:
                    m2_mm = 0.0
                self.is_m1_m2_modified = True

                # Check rotation deadband if rotation is enabled for this moderation cycle
                if not self.is_m3_modified and self.moderation_count < self.max_angle_moderations:
                    if abs(m3_deg) < ROTATION_DEADBAND_DEG:
                        m3_deg = 0.0
                    self.is_m3_modified = True
                else:
                    m3_deg = 0.0

            # If both translation and angle updates occurred, increment iteration counter
            if self.is_m1_m2_modified and self.is_m3_modified:
                self.moderation_count += 1
                self.is_m1_m2_modified = False
                if self.moderation_count < self.max_angle_moderations:
                    self.is_m3_modified = False
                print("M3: ", self.is_m3_modified)
        else:
            self.release_fabric()
            return

        # ── Step 4: Dispatch command string 'M1, M2, M3' to STM32 controller ──
        active_phase = self.current_state
        if self.ser is not None:
            cmd = f"{m1_mm}, {m2_mm}, {m3_deg}"
            print(f"[CTRL] [{active_phase.upper()}] raw_tx={m1_mm:+.3f} mm, raw_ty={m2_mm:+.3f} mm, raw_rot={m3_deg:+.3f} deg (iteration {self.moderation_count}/{self.max_moderations})")
            send_and_receive(self.ser, cmd)
        else:
            print(f"[dry-run] [{active_phase.upper()}] Would send: raw_tx={m1_mm} mm, raw_ty={m2_mm} mm, raw_rot={m3_deg} deg (iteration {self.moderation_count}/{self.max_moderations})")

        # ── Step 4b: Update tracked pivot in camera pixels based on motor actuation ──
        if hasattr(self.adjustor, 'update_pivot'):
            tx_px = (m1_mm / 10.0) * PX_PER_CM_X
            ty_px = (m2_mm / 10.0) * PX_PER_CM_Y
            delta_p = -(tx_px * U1_VECTOR + ty_px * U2_VECTOR)
            new_p = self.adjustor.update_pivot(float(delta_p[0]), float(delta_p[1]))
            print(f"[PIVOT] Hand center updated: ({new_p[0]:.1f}, {new_p[1]:.1f}) px (delta=({delta_p[0]:+.1f}, {delta_p[1]:+.1f}) px)")

        # ── Step 5: Check sub-phase transition (switch to L2b after finishing L2a) ──
        if self.current_state == 'l2a' and self.moderation_count >= self.max_moderations_l2a:
            if self.moderation_count < self.max_moderations:
                self.current_state = 'l2b'
                self.adjustor.x_basis = self.x_basis_l2b
                self.adjustor.y_basis = self.y_basis_l2b
                print(f"\n[PHASE] L2a finished ({self.moderation_count}/{self.max_moderations_l2a}). Switched to L2b (basis=({self.adjustor.x_basis:.1f}, {self.adjustor.y_basis:.1f})).")

    # -----------------------------------------------------------------------
    # UART & Keyboard Input Handlers
    # -----------------------------------------------------------------------

    def handle_uart_line(self, line: str) -> None:
        """Process incoming UART status line from STM32 and trigger alignment.

        Parses status lines emitted by the STM32 motor driver:
          - "OK": Signals the STM32 has completed mechanical motion and is ready
            for vision re-evaluation (sets trigger_compute = True).
          - "L1": Signals beginning of Phase 1 rough adjustment.
          - "L2A" / "L2a": Signals beginning of Phase 2a coarse alignment.
          - "L2B" / "L2b": Signals beginning of Phase 2b fine offset adjustment.
          - "L2": Generic Phase 2 trigger; defaults to L2a (or L2b if budget is 0).

        Local command echos and intermediate log lines are safely ignored.

        Args:
            line: Stripped text line received from serial UART.
        """
        print("log: ", line)
        if "OK" in line:
            # Only trigger compute if actively in an adjustment phase
            if self.current_state in ('l1', 'l2a', 'l2b'):
                self.trigger_compute = True
                self.retry_count = 0
        if "L1" in line:
            print(f"UART Trigger Received (L1): {line}")
            if self.current_state != 'l1':
                self.switch_to_phase('l1')
                self.trigger_compute = True
                self.retry_count = 0
        elif "L2A" in line or "L2a" in line:
            print(f"UART Trigger Received (L2a): {line}")
            if self.current_state != 'l2a':
                self.switch_to_phase('l2a')
                self.trigger_compute = True
                self.retry_count = 0
        elif "L2B" in line or "L2b" in line:
            print(f"UART Trigger Received (L2b): {line}")
            if self.current_state != 'l2b':
                self.switch_to_phase('l2b')
                self.trigger_compute = True
                self.retry_count = 0
        elif "L2" in line:
            print(f"UART Trigger Received (L2): {line}")
            if self.current_state not in ('l2', 'l2a', 'l2b'):
                self.switch_to_phase('l2')
                self.trigger_compute = True
                self.retry_count = 0

    def handle_keyboard_char(self, raw_char: str, is_continuous: bool = False) -> bool:
        """Process interactive terminal keystroke command.

        Maintains a rolling keystroke buffer to detect multi-character triggers
        and executes corresponding control actions:
          - 'l1': Manually switch to Phase 1 (L1).
          - 'l2a' / '2a': Manually switch to Phase 2a (L2a).
          - 'l2b' / '2b': Manually switch to Phase 2b (L2b).
          - 'l2': Manually switch to Phase 2 (L2).
          - 's': Send status query / start signal to STM32.
          - 'k': Manually issue clamp release command 'k' and reset to idle.
          - 'q' / 'x': Request application termination.
          - '1' / '2': Send test motor command and quit.
          - 'o': In dry-run mode, simulate receiving an 'OK' trigger from STM32.

        Args:
            raw_char: Single character string read from standard input.
            is_continuous: If True, indicates invocation from continuous streaming mode.

        Returns:
            bool: True if application exit was requested, False otherwise.
        """
        cmd_key = raw_char.strip().lower()
        if raw_char:
            self.keyboard_buffer += cmd_key
            self.keyboard_buffer = self.keyboard_buffer[-3:]

        if self.keyboard_buffer.endswith('l1'):
            print("\nUser manually triggered step 1 (L1)")
            self.switch_to_phase('l1')
            if not is_continuous:
                self.trigger_compute = True
                self.retry_count = 0
            self.keyboard_buffer = ""
        elif self.keyboard_buffer.endswith('l2a') or self.keyboard_buffer.endswith('2a'):
            print("\nUser manually triggered step 2a (L2a)")
            self.switch_to_phase('l2a')
            if not is_continuous:
                self.trigger_compute = True
                self.retry_count = 0
            self.keyboard_buffer = ""
        elif self.keyboard_buffer.endswith('l2b') or self.keyboard_buffer.endswith('2b'):
            print("\nUser manually triggered step 2b (L2b)")
            self.switch_to_phase('l2b')
            if not is_continuous:
                self.trigger_compute = True
                self.retry_count = 0
            self.keyboard_buffer = ""
        elif self.keyboard_buffer.endswith('l2'):
            print("\nUser manually triggered step 2 (L2)")
            self.switch_to_phase('l2')
            if not is_continuous:
                self.trigger_compute = True
                self.retry_count = 0
            self.keyboard_buffer = ""
        elif cmd_key == 's' and not is_continuous:
            self.trigger_compute = False
            send_and_receive(self.ser, cmd_key)
        elif cmd_key == 'k':
            self.trigger_compute = False
            self.moderation_count = 0
            self.is_m1_m2_modified = False
            self.is_m3_modified = False
            self.retry_count = 0
            self.adjustor.x_basis = self.x_basis_l1
            self.adjustor.y_basis = self.y_basis_l1
            self.current_state = 'idle'
            send_and_receive(self.ser, cmd_key)
        elif cmd_key in ('q', 'x'):
            print("\nQuit requested.")
            self.moderation_count = 0
            self.is_m1_m2_modified = False
            self.is_m3_modified = False
            self.current_state = 'idle'
            return True
        elif cmd_key in ['1', '2']:
            if self.ser is not None:
                print(f"\n[CTRL] Sent {cmd_key}")
                send_and_receive(self.ser, cmd_key)
            print("\nQuit requested.")
            self.moderation_count = 0
            self.is_m1_m2_modified = False
            self.is_m3_modified = False
            self.current_state = 'idle'
            return True
        elif cmd_key == 'o' and self.dry_run and not is_continuous:
            if self.current_state in ('l1', 'l2a', 'l2b'):
                print(f"\n[dry-run] User chose OK")
                self.trigger_compute = True
                self.retry_count = 0
            else:
                print(f"\n[dry-run] Ignored OK while state is {self.current_state}")
            self.keyboard_buffer = ""

        return False

    def render_hud(self, frame) -> None:
        """Render annotated live camera preview and HUD display overlay.

        Draws target basis reticle with coordinate axes via draw_adjustor_overlay(),
        overlays current state machine status, and updates the OpenCV window.

        Args:
            frame: Raw BGR camera image frame captured from video stream.
        """
        if not self.visualize or frame is None:
            return
        display_img = cv.resize(frame, self.img_size) if self.img_size is not None else frame.copy()
        draw_adjustor_overlay(display_img, (self.adjustor.x_basis, self.adjustor.y_basis), AXIS_LENGTH_PX)
        cv.putText(display_img, f"WAITING FOR ARM (STATE: {self.current_state.upper()})", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        window_name = "Prior Mask Adjustment" if self.detector_mode == DetectorMode.DEXINED else "Fabric Adjustor"
        cv.imshow(window_name, display_img)
        cv.waitKey(1)

    # -----------------------------------------------------------------------
    # Main Loops
    # -----------------------------------------------------------------------

    def run_continuous(self) -> None:
        """Execute continuous visual servoing loop using adjustor.run() generator.

        Continuously streams frames from camera through adjustor backend, calculates
        displacement vectors on every frame, and dispatches motor commands with
        interleaved non-blocking keyboard input and UART line monitoring.
        """
        # pyrefly: ignore [unexpected-keyword]
        for control_signals in self.adjustor.run(cam=self.cam, target_frames=self.target_frames, visualize=self.visualize):
            # 1. Non-blocking key check
            if select.select([sys.stdin], [], [], 0)[0]:
                raw_char = sys.stdin.read(1)
                if self.handle_keyboard_char(raw_char, is_continuous=True):
                    break

            # 2. UART Phase Check in continuous mode
            if self.ser is not None and self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if "L1" in line:
                    if self.current_state != 'l1':
                        self.switch_to_phase('l1')
                elif "L2A" in line or "L2a" in line:
                    if self.current_state != 'l2a':
                        self.switch_to_phase('l2a')
                elif "L2B" in line or "L2b" in line:
                    if self.current_state != 'l2b':
                        self.switch_to_phase('l2b')
                elif "L2" in line:
                    if self.current_state not in ('l2', 'l2a', 'l2b'):
                        self.switch_to_phase('l2')

            if control_signals is None:
                print("Returned None signal, recalculating...")
                continue
            self.process_control_signals(control_signals)

    def run_event_driven(self) -> None:
        """Execute default event-driven visual servoing loop responding to STM32 triggers.

        Production execution loop for Jetson Orin deployment. Remains in an idle
        camera monitoring state until a UART readiness trigger ("OK", "L1", "L2") or
        manual keyboard command arrives. Upon trigger receipt:
          1. Captures fresh camera frames and computes a single-shot alignment vector.
          2. Handles missing vision detections with retry budget (max_retries), releasing
             fabric clamp with 'k' and error codes ('e1'/'e2') if exhausted.
          3. Dispatches computed motor correction signals via process_control_signals().
        """
        self.trigger_compute = False
        self.retry_count = 0

        while True:
            # 1. Non-blocking key check
            if select.select([sys.stdin], [], [], 0)[0]:
                raw_char = sys.stdin.read(1)
                if self.handle_keyboard_char(raw_char, is_continuous=False):
                    break

            # 2. UART Event Trigger Wait
            if not self.dry_run and self.ser is not None and self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                self.handle_uart_line(line)

            # 3. If no trigger, just display live camera feed and wait
            if not self.trigger_compute:
                if self.visualize and self.cam is not None:
                    ret, frame, has_new = self.cam.read()
                    if ret and frame is not None and has_new:
                        self.render_hud(frame)
                else:
                    time.sleep(0.01)
                continue

            # 4. Trigger received: check budget or compute control signal
            if self.moderation_count >= self.max_moderations:
                self.trigger_compute = False
                self.process_control_signals(None)
                continue

            # pyrefly: ignore [unexpected-keyword]
            control_signals = self.adjustor.compute_single_signal(self.cam, target_frames=self.target_frames, visualize=self.visualize)
            if control_signals is None:
                self.retry_count += 1
                if self.retry_count < self.max_retries:
                    print(f"Returned None signal ({self.retry_count}/{self.max_retries}), recalculating...")
                    continue
                else:
                    print(f"Returned None signal {self.retry_count} times (max_retries={self.max_retries} reached). Forcing stop and release...")
                    # send_and_receive(self.ser, "50, 0, 0")
                    send_and_receive(self.ser, 'k')
                    err_cmd = 'e1' if self.current_state == 'l1' else 'e2'
                    send_and_receive(self.ser, err_cmd)
                    self.trigger_compute = False
                    self.is_m1_m2_modified = False
                    self.is_m3_modified = False
                    self.moderation_count = 0
                    self.retry_count = 0
                    self.adjustor.x_basis = self.x_basis_l1
                    self.adjustor.y_basis = self.y_basis_l1
                    self.current_state = 'l1_done' if self.current_state == 'l1' else 'l2_done'
                    continue

            self.retry_count = 0
            self.trigger_compute = False
            self.process_control_signals(control_signals)

    def run(self) -> None:
        """Run the complete fabric adjustor application lifecycle.

        Orchestrates resource initialization (_init_serial, _init_vision, _init_terminal),
        branches into either continuous or event-driven execution mode, traps
        KeyboardInterrupt for graceful termination, and guarantees clean teardown
        of serial ports and terminal settings in a finally block.
        """
        try:
            self._init_serial()
            self._init_vision()
            self._init_terminal()

            print("\n--- ENGINE CONTROLLER INTERFACE ---")
            print(f"[CONFIG] Moderation budgets -> L1: {self.max_moderations}, L2: {self.max_moderations} (L2a: {self.max_moderations_l2a}, L2b: {self.max_moderations_l2b})")

            if self.continuous:
                self.run_continuous()
            else:
                self.run_event_driven()

        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            self._restore_terminal()
            self._close_serial()


# ---------------------------------------------------------------------------
# Procedural Entry Point (Backward-Compatible Wrapper)
# ---------------------------------------------------------------------------

def main(**kwargs) -> None:
    """Instantiate and execute the FabricAdjustorApp pipeline.

    Backward-compatible procedural entry point that forwards keyword arguments
    to FabricAdjustorApp and invokes app.run().

    Args:
        **kwargs: Configuration keyword arguments passed directly to FabricAdjustorApp.__init__.
    """
    app = FabricAdjustorApp(**kwargs)
    app.run()


if __name__ == '__main__':
    args = parse_args()
    mode = DetectorMode(args.detector)
    img_size = tuple(args.img_size) if args.img_size else None
    main(
        img_size=img_size,
        x_basis=args.x_basis,
        y_basis=args.y_basis,
        visualize=args.visualize,
        detector_mode=mode,
        prior_mask_path=args.prior_mask,
        match_method=args.match_method,
        morphology=args.morphology,
        enhance_contrast=args.enhance_contrast,
        smooth_texture=args.smooth_texture,
        batch_size=args.batch_size,
        max_moderations=args.max_moderations,
        max_angle_moderations=args.max_angle_moderations,
        max_retries=args.max_retries,
        port=args.port,
        baud=args.baud,
        dry_run=args.dry_run,
        crop_roi=tuple(args.crop_roi) if args.crop_roi is not None else None,
        continuous=args.continuous,
        cooldown_time=args.cooldown_time,
        x_offset_cm=args.x_offset_cm,
        y_offset_cm=args.y_offset_cm,
        max_moderations_l2a=args.max_moderations_l2a,
        x_offset_cm_l2a=args.x_offset_cm_l2a,
        y_offset_cm_l2a=args.y_offset_cm_l2a,
        rot_alignment=args.rot_alignment,
        target_frames=args.target_frames,
        pivot=tuple(args.pivot) if args.pivot is not None else None,
        compensate_rotation=args.compensate_rotation,
    )
