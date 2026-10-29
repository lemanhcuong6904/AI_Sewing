"""
utils/viz_adjustor.py
---------------------
Shared OpenCV drawing helpers for the fabric adjustor visualisation pipeline.

Used by:
  - data_collector_adjustor.py   (live data-collection overlay)
  - utils/helper.py              (visualize_and_save_frame_adjustor)
"""

import math
import cv2 as cv
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

from utils.constants import PX_PER_CM_X, PX_PER_CM_Y, AXIS_LENGTH_PX, U1_VECTOR, U2_VECTOR
TICK_HALF      = 5         # half-length of each cm tick mark in pixels
PIVOT_LINE_LEN = 60        # length of every pivot reference line

# PIVOT_ANGLES  = [math.pi / 8, math.pi / 6, math.pi / 4, math.pi / 10]   # ±22.5°, ±30°, ±45°, ±18°
PIVOT_ANGLES  = []  

PIVOT_COLOURS = [
    (255, 200,   0),   # cyan-ish  — ±22.5°
    (  0, 165, 255),   # orange    — ±30°
    (180,   0, 255),   # magenta   — ±45°
    (0,   255, 0),     # green     — ±18°
]

AXIS_COLOUR_X = (0,   0, 255)   # Red  (BGR)
AXIS_COLOUR_Y = (0, 255,   0)   # Green (BGR)
GRID_COLOUR   = (80, 80,  80)   # Dark grey
GRID_ALPHA    = 0.25            # Grid transparency


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_grid(img, basis, alpha: float = GRID_ALPHA) -> None:
    """Draw a full-image coordinate grid anchored at *basis* (in-place).

    Grid lines are spaced appropriately apart in both axes.
    A semi-transparent overlay keeps the camera feed visible.
    """
    h, w = img.shape[:2]
    bx, by = int(basis[0]), int(basis[1])
    overlay = img.copy()

    # Vertical lines (X-axis direction)
    x = bx
    while x >= 0:
        cv.line(overlay, (int(x), 0), (int(x), h), GRID_COLOUR, 1)
        x -= PX_PER_CM_X
    x = bx + PX_PER_CM_X
    while x < w:
        cv.line(overlay, (int(x), 0), (int(x), h), GRID_COLOUR, 1)
        x += PX_PER_CM_X

    # Horizontal lines (Y-axis direction)
    y = by
    while y < h:
        cv.line(overlay, (0, int(y)), (w, int(y)), GRID_COLOUR, 1)
        y += PX_PER_CM_Y
    y = by - PX_PER_CM_Y
    while y >= 0:
        cv.line(overlay, (0, int(y)), (w, int(y)), GRID_COLOUR, 1)
        y -= PX_PER_CM_Y

    cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_axes(img, basis, axis_length: int) -> None:
    """Draw the X (red, pointing left) and Y (green, pointing down) axes.

    Args:
        img: BGR image, modified in-place.
        basis: (x, y) origin of the coordinate system.
        axis_length: length of each axis arrow in pixels.
    """
    bx, by = int(basis[0]), int(basis[1])
    tip = max(0.01, 10 / axis_length)   # proportional arrowhead

    # X-axis (U1)
    ex = int(bx + U1_VECTOR[0] * axis_length)
    ey = int(by + U1_VECTOR[1] * axis_length)
    cv.arrowedLine(img, (bx, by), (ex, ey),
                   AXIS_COLOUR_X, 1, tipLength=tip)
    
    text_offset_x = 15 if U1_VECTOR[0] < 0 else -5
    text_offset_y = 5 if U1_VECTOR[1] >= 0 else -15
    cv.putText(img, "X", (ex + text_offset_x, ey + text_offset_y),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, AXIS_COLOUR_X, 1)

    # Y-axis (U2)
    ex_y = int(bx + U2_VECTOR[0] * axis_length)
    ey_y = int(by + U2_VECTOR[1] * axis_length)
    cv.arrowedLine(img, (bx, by), (ex_y, ey_y),
                   AXIS_COLOUR_Y, 1, tipLength=tip)
                   
    text_offset_x_y = 5 if U2_VECTOR[0] >= 0 else -15
    text_offset_y_y = 15 if U2_VECTOR[1] >= 0 else -5
    cv.putText(img, "Y", (ex_y + text_offset_x_y, ey_y + text_offset_y_y),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, AXIS_COLOUR_Y, 1)

    # Origin dot
    cv.circle(img, (bx, by), 1, (0, 0, 255), -1)


def draw_axis_ticks(img, basis, n_ticks: int | None = None) -> None:
    """Draw cm tick marks every PX_PER_CM_X/Y pixels along both axes.

    Args:
        img: BGR image, modified in-place.
        basis: (x, y) origin.
        n_ticks: number of tick marks per axis.  Defaults to filling the image.
    """
    h, w = img.shape[:2]
    bx, by = int(basis[0]), int(basis[1])

    if n_ticks is None:
        n_ticks_x = int(w // PX_PER_CM_X) + 1
        n_ticks_y = int(h // PX_PER_CM_Y) + 1
    else:
        n_ticks_x = n_ticks
        n_ticks_y = n_ticks

    # X-axis ticks (mm level)
    for i in range(1, n_ticks_x * 10 + 1):
        d = i * (PX_PER_CM_X / 10.0)
        
        tx = int(bx + U1_VECTOR[0] * d)
        ty = int(by + U1_VECTOR[1] * d)
        
        if 0 <= tx < w and 0 <= ty < h:
            if i % 10 == 0:
                tick_len = TICK_HALF
            elif i % 5 == 0:
                tick_len = int(TICK_HALF * 0.75)
            else:
                tick_len = TICK_HALF // 2
                
            u1_perp = np.array([-U1_VECTOR[1], U1_VECTOR[0]], dtype=np.float64)
            dx_tick = int(u1_perp[0] * tick_len)
            dy_tick = int(u1_perp[1] * tick_len)
            cv.line(img, (tx - dx_tick, ty - dy_tick), (tx + dx_tick, ty + dy_tick),
                    AXIS_COLOUR_X, 1, cv.LINE_AA)
                    
    # Y-axis ticks (mm level)
    for i in range(1, n_ticks_y * 10 + 1):
        d = i * (PX_PER_CM_Y / 10.0)
        
        tx_y = int(bx + U2_VECTOR[0] * d)
        ty_y = int(by + U2_VECTOR[1] * d)
        
        if 0 <= tx_y < w and 0 <= ty_y < h:
            if i % 10 == 0:
                tick_len = TICK_HALF
            elif i % 5 == 0:
                tick_len = int(TICK_HALF * 0.75)
            else:
                tick_len = TICK_HALF // 2
                
            u2_perp = np.array([-U2_VECTOR[1], U2_VECTOR[0]], dtype=np.float64)
            dx_tick_y = int(u2_perp[0] * tick_len)
            dy_tick_y = int(u2_perp[1] * tick_len)
            cv.line(img, (tx_y - dx_tick_y, ty_y - dy_tick_y), (tx_y + dx_tick_y, ty_y + dy_tick_y),
                    AXIS_COLOUR_Y, 1, cv.LINE_AA)


def draw_pivot_lines(img, basis, line_len: int = PIVOT_LINE_LEN) -> None:
    """Draw pivot reference lines for both axes if PIVOT_ANGLES is configured.

    Args:
        img: BGR image, modified in-place.
        basis: (x, y) origin.
        line_len: length of each pivot line in pixels.
    """
    if not PIVOT_ANGLES:
        return

    bx, by = int(basis[0]), int(basis[1])

    # Dynamic base angles from U1 and U2
    angle_u1 = math.atan2(U1_VECTOR[1], U1_VECTOR[0])
    angle_u2 = math.atan2(U2_VECTOR[1], U2_VECTOR[0])
    axes_base_angles = [angle_u1, angle_u2]

    for base_angle in axes_base_angles:
        for colour, pivot_angle in zip(PIVOT_COLOURS, PIVOT_ANGLES):
            for sign in (+1, -1):
                a = base_angle + sign * pivot_angle
                ex = int(bx + line_len * math.cos(a))
                ey = int(by + line_len * math.sin(a))
                cv.line(img, (bx, by), (ex, ey), colour, 1, cv.LINE_AA)


def draw_adjustor_overlay(
    img: np.ndarray,
    basis: tuple | list,
    axis_length: int,
    best_h_line: list | np.ndarray | None = None,
    best_v_line: list | np.ndarray | None = None,
    control_signals: tuple | None = None,
) -> None:
    """
    Renders the composite visualization overlay in-place onto the given camera frame.

    Draws coordinate background grid, axes arrows, millimeter tick marks, pivot lines,
    detected horizontal (yellow) and vertical (magenta) fabric edges, and numerical
    displacement/rotation readout text.

    Parameters:
        img: np.ndarray
            BGR image frame (dtype: uint8, shape: [H, W, 3]), modified in-place.
        basis: Union[Tuple[float, float], List[float]]
            (x, y) pixel coordinates of the target coordinate system origin.
        axis_length: int
            Length of the coordinate axis indicator arrows in pixels.
        best_h_line: Optional[Union[List[float], np.ndarray]]
            Coordinates [x1, y1, x2, y2] of the selected horizontal fabric edge, or None.
        best_v_line: Optional[Union[List[float], np.ndarray]]
            Coordinates [x1, y1, x2, y2] of the selected vertical fabric edge, or None.
        control_signals: Optional[Tuple[Tuple[float, float], Tuple[float, float]]]
            Computed adjustment signals `((tx, ty), (ax, ay))` in pixels and degrees, or None.

    Returns:
        None: Modifies `img` directly in-place.
    """
    # ── Step 1: Draw coordinate reference scaffolding ─────────────────
    draw_grid(img, basis)
    draw_pivot_lines(img, basis)
    draw_axes(img, basis, axis_length)
    draw_axis_ticks(img, basis)

    # ── Step 2: Overlay tracked horizontal and vertical fabric edges ──
    if best_h_line is not None:
        hx1, hy1, hx2, hy2 = map(int, best_h_line)
        cv.line(img, (hx1, hy1), (hx2, hy2), (0, 255, 255), 2)   # Yellow for horizontal edge

    if best_v_line is not None:
        vx1, vy1, vx2, vy2 = map(int, best_v_line)
        cv.line(img, (vx1, vy1), (vx2, vy2), (255, 0, 255), 2)   # Magenta for vertical edge

    # ── Step 3: Overlay control signals readout text ───────────────────
    if control_signals is not None and best_h_line is not None and best_v_line is not None:
        trans_vec, rot_angles = control_signals
        tx, ty = trans_vec
        ax, ay = rot_angles
        h_img = img.shape[0]
        cv.putText(img, f"Offset: ({tx:.1f}, {ty:.1f})",
                   (5, h_img - 20), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv.putText(img, f"Angles: ({ax:.0f}, {ay:.0f})",
                   (5, h_img - 5),  cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)


def draw_target_axes(
    img: np.ndarray,
    basis: tuple | list,
    tx_cm: float,
    ty_cm: float,
    angle_h_deg: float,
    angle_v_deg: float,
    line_len: int = 200,
) -> None:
    """
    Renders the projected target fabric edge lines that the sewing piece must align to.

    Calculates the physical intersection position shifted by translation (tx_cm, ty_cm)
    relative to the origin, and extends target lines rotated by `angle_h_deg` and `angle_v_deg`.

    Parameters:
        img: np.ndarray
            BGR image frame (dtype: uint8, shape: [H, W, 3]), modified in-place.
        basis: Union[Tuple[float, float], List[float]]
            (x, y) pixel coordinates of the coordinate system origin.
        tx_cm: float
            Target translation along X axis in centimeters (positive translates left in image coordinates).
        ty_cm: float
            Target translation along Y axis in centimeters (positive translates down in image coordinates).
        angle_h_deg: float
            Target rotation angle of the horizontal fabric line in degrees.
        angle_v_deg: float
            Target rotation angle of the vertical fabric line in degrees.
        line_len: int
            Half-length of each drawn target line in pixels (default: 200).

    Returns:
        None: Modifies `img` directly in-place.
    """
    b_vec = np.array(basis, dtype=np.float64)

    # ── Step 1: Target corner position in general vector form ─────────
    v_target = (tx_cm * PX_PER_CM_X) * U1_VECTOR + (ty_cm * PX_PER_CM_Y) * U2_VECTOR
    target_corner = b_vec + v_target
    ix = int(round(target_corner[0]))
    iy = int(round(target_corner[1]))

    # Draw intersection center dot
    cv.circle(img, (ix, iy), 5, (255, 255, 255), -1)

    # ── Step 2: Render horizontal and vertical target lines ───────────
    # Orientation normal determinant converts physical counterclockwise angle to image frame angle
    det_basis = U1_VECTOR[0] * U2_VECTOR[1] - U1_VECTOR[1] * U2_VECTOR[0]
    sign_rot = 1.0 if det_basis >= 0 else -1.0

    # Horizontal target line along U1 rotated by angle_h_deg
    base_angle_u1 = math.atan2(U1_VECTOR[1], U1_VECTOR[0])
    rad_h = base_angle_u1 + sign_rot * math.radians(angle_h_deg)
    dx_h = int(round(line_len * math.cos(rad_h)))
    dy_h = int(round(line_len * math.sin(rad_h)))
    cv.line(img, (ix - dx_h, iy - dy_h), (ix + dx_h, iy + dy_h), (0, 255, 255), 2, cv.LINE_AA)
    cv.putText(img, f"H:{angle_h_deg:+.1f} deg", (ix + abs(dx_h) + 4, iy + dy_h),
               cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Vertical target line along U2 rotated by angle_v_deg
    base_angle_u2 = math.atan2(U2_VECTOR[1], U2_VECTOR[0])
    rad_v = base_angle_u2 + sign_rot * math.radians(angle_v_deg)
    dx_v = int(round(line_len * math.cos(rad_v)))
    dy_v = int(round(line_len * math.sin(rad_v)))
    cv.line(img, (ix - dx_v, iy - dy_v), (ix + dx_v, iy + dy_v), (255, 0, 255), 2, cv.LINE_AA)
    cv.putText(img, f"V:{angle_v_deg:+.1f} deg", (ix - dx_v - 80, iy - dy_v - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
