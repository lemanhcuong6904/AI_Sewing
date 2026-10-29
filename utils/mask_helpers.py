"""
utils/mask_helpers.py
---------------------
Stateless helper functions for DexiNed-based prior-mask fabric adjustment.

These utilities handle:
  - Morphological post-processing of edge masks
  - Mask statistics extraction (moments, bounding-box)
  - Mask matching / alignment methods (moments, bbox, ECC, feature, contour)
  - Noise-robust batch signal filtering
  - Visualization of prior-vs-current mask alignment

All functions are pure / stateless and carry no camera or model state,
making them safe to import independently for offline evaluation,
unit testing, or any other context.
"""

import math
from typing import Optional

import cv2 as cv
import numpy as np

from utils.constants import DEFAULT_X_BASIS, DEFAULT_Y_BASIS, AXIS_LENGTH_PX, U1_VECTOR, U2_VECTOR
from utils.viz_adjustor import draw_adjustor_overlay


# ---------------------------------------------------------------------------
# Morphological post-processing
# ---------------------------------------------------------------------------

def apply_morphology(mask, method: str = 'none'):
    """Apply a morphological operation to strip noise from an edge mask.

    Args:
        mask:   uint8 numpy array (H, W).
        method: One of 'none' | 'erode' | 'open' | 'fill'.

    Returns:
        Processed mask with the same dtype and shape.
    """
    if method == 'none' or mask is None:
        return mask

    kernel = np.ones((7, 7), np.uint8)
    if method == 'erode':
        return cv.erode(mask, kernel, iterations=1)
    elif method == 'open':
        return cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    elif method == 'fill':
        # Small closing kernel to avoid bridging background noise
        close_kernel = np.ones((1, 1), np.uint8)
        closed = cv.morphologyEx(mask, cv.MORPH_CLOSE, close_kernel)
        contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        largest_contour = max(contours, key=cv.contourArea)
        filled = np.zeros_like(mask)
        cv.drawContours(filled, [largest_contour], 0, 255, thickness=cv.FILLED)
        return filled

    return mask


# ---------------------------------------------------------------------------
# Mask statistics helpers
# ---------------------------------------------------------------------------

def _get_moments_stats(mask):
    """Return (centroid, orientation_angle) via image moments."""
    M = cv.moments(mask)
    if M['m00'] == 0:
        return (mask.shape[1] / 2, mask.shape[0] / 2), 0.0
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    theta = 0.5 * math.atan2(2 * M['mu11'], M['mu20'] - M['mu02'])
    return (cx, cy), theta


def _get_bbox_stats(mask):
    """Return (centroid, orientation_angle) via minimum-area bounding rectangle."""
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (mask.shape[1] / 2, mask.shape[0] / 2), 0.0
    largest_contour = max(contours, key=cv.contourArea)
    rect = cv.minAreaRect(largest_contour)
    (cx, cy), (w, h), angle = rect
    if w < h:
        angle = angle + 90
    theta = math.radians(angle)
    theta = (theta + np.pi / 2) % np.pi - np.pi / 2
    return (cx, cy), theta


# ---------------------------------------------------------------------------
# Matching methods
# ---------------------------------------------------------------------------

def _feature_matching(prior_img, current_img, prior_mask, current_mask, x_basis, y_basis):
    """Estimate fabric displacement via ORB feature matching + affine RANSAC."""
    if prior_img is None or current_img is None:
        print("[mask_helpers] Warning: Feature matching requires original images.")
        return (0.0, 0.0), (0.0, 0.0)

    orb = cv.ORB_create(500)
    kp_p, des_p = orb.detectAndCompute(prior_img, prior_mask)
    kp_c, des_c = orb.detectAndCompute(current_img, current_mask)

    if des_p is None or des_c is None or len(des_p) < 3 or len(des_c) < 3:
        return (0.0, 0.0), (0.0, 0.0)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_p, des_c)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:50]

    if len(good_matches) < 3:
        return (0.0, 0.0), (0.0, 0.0)

    pts_p = np.float32([kp_p[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_c = np.float32([kp_c[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    transform_matrix, _ = cv.estimateAffinePartial2D(pts_p, pts_c, method=cv.RANSAC)
    if transform_matrix is None:
        return (0.0, 0.0), (0.0, 0.0)

    pt = np.array([[[x_basis, y_basis]]], dtype=np.float32)
    pt_mapped = cv.transform(pt, transform_matrix)[0][0]
    I_x, I_y = pt_mapped[0], pt_mapped[1]

    v = np.array([I_x - x_basis, I_y - y_basis])
    tx = np.dot(v, U1_VECTOR)
    ty = np.dot(v, U2_VECTOR)
    theta = math.atan2(transform_matrix[1, 0], transform_matrix[0, 0])
    return (tx, ty), (theta, theta)


def _contour_matching(prior_mask, current_mask, x_basis, y_basis):
    """Estimate fabric displacement via centroid tracking + brute-force rotation search."""
    contours_p, _ = cv.findContours(prior_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_c, _ = cv.findContours(current_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours_p or not contours_c:
        return (0.0, 0.0), (0.0, 0.0)

    cnt_p = max(contours_p, key=cv.contourArea)
    cnt_c = max(contours_c, key=cv.contourArea)

    M_p = cv.moments(cnt_p)
    cx_p = M_p['m10'] / (M_p['m00'] + 1e-5)
    cy_p = M_p['m01'] / (M_p['m00'] + 1e-5)

    M_c = cv.moments(cnt_c)
    cx_c = M_c['m10'] / (M_c['m00'] + 1e-5)
    cy_c = M_c['m01'] / (M_c['m00'] + 1e-5)

    best_angle = 0
    max_overlap = -1

    h, w = prior_mask.shape
    M_translate = np.float32([[1, 0, cx_p - cx_c], [0, 1, cy_p - cy_c]])
    current_translated = cv.warpAffine(current_mask, M_translate, (w, h))

    for angle_i in range(-450, 451, 5):
        angle = angle_i / 10.0
        M_rot = cv.getRotationMatrix2D((cx_p, cy_p), angle, 1.0)
        current_rotated = cv.warpAffine(current_translated, M_rot, (w, h))
        overlap_area = cv.countNonZero(cv.bitwise_and(prior_mask, current_rotated))
        if overlap_area > max_overlap:
            max_overlap = overlap_area
            best_angle = angle

    dtheta = math.radians(best_angle)

    dx_p = x_basis - cx_p
    dy_p = y_basis - cy_p
    delta_theta = -dtheta
    dx_c = dx_p * math.cos(delta_theta) - dy_p * math.sin(delta_theta)
    dy_c = dx_p * math.sin(delta_theta) + dy_p * math.cos(delta_theta)

    I_x = cx_c + dx_c
    I_y = cy_c + dy_c

    v = np.array([I_x - x_basis, I_y - y_basis])
    tx = np.dot(v, U1_VECTOR)
    ty = np.dot(v, U2_VECTOR)
    return (tx, ty), (dtheta, dtheta)


def _chamfer_matching(prior_mask, current_mask, x_basis, y_basis, fast=False):
    """Estimate fabric displacement via distance transform and template matching."""
    # 1. Compute distance transform of the current mask
    inv_current = cv.bitwise_not(current_mask)
    dist_map = cv.distanceTransform(inv_current, cv.DIST_L2, 3)

    # 2. Extract bounding box from prior mask to use as template
    x, y, w, h = cv.boundingRect(prior_mask)
    if w == 0 or h == 0:
        return (0.0, 0.0), (0.0, 0.0)
    
    pad = int(math.ceil(math.sqrt(w**2 + h**2) / 2))
    x_start = max(0, x - pad)
    y_start = max(0, y - pad)
    x_end = min(prior_mask.shape[1], x + w + pad)
    y_end = min(prior_mask.shape[0], y + h + pad)
    
    prior_roi = prior_mask[y_start:y_end, x_start:x_end]
    
    M_p = cv.moments(prior_roi)
    if M_p['m00'] == 0:
        return (0.0, 0.0), (0.0, 0.0)
    
    cx_roi = M_p['m10'] / M_p['m00']
    cy_roi = M_p['m01'] / M_p['m00']
    
    cx_p = cx_roi + x_start
    cy_p = cy_roi + y_start
    
    best_angle = 0.0
    min_dist_sum = float('inf')
    best_loc = (0, 0)
    
    def search_angles(angles_to_test):
        nonlocal best_angle, min_dist_sum, best_loc
        for angle in angles_to_test:
            M_rot = cv.getRotationMatrix2D((cx_roi, cy_roi), angle, 1.0)
            rotated_roi = cv.warpAffine(prior_roi, M_rot, (prior_roi.shape[1], prior_roi.shape[0]))
            
            template = (rotated_roi / 255.0).astype(np.float32)
            num_edge_pixels = np.sum(template)
            
            if num_edge_pixels == 0:
                continue
                
            res = cv.matchTemplate(dist_map, template, cv.TM_CCORR)
            min_val, _, min_loc, _ = cv.minMaxLoc(res)
            avg_dist = min_val / num_edge_pixels
            
            if avg_dist < min_dist_sum:
                min_dist_sum = avg_dist
                best_angle = angle
                best_loc = min_loc

    if not fast:
        angles = [a / 10.0 for a in range(-450, 451, 5)]
        search_angles(angles)
    else:
        search_angles(range(-45, 46, 5))
        search_angles([a * 1.0 for a in range(int(best_angle) - 4, int(best_angle) + 5, 1)])
        search_angles([best_angle - 0.5, best_angle + 0.5])
        
    cx_c = best_loc[0] + cx_roi
    cy_c = best_loc[1] + cy_roi
    
    dtheta = math.radians(best_angle)
    
    dx_p = x_basis - cx_p
    dy_p = y_basis - cy_p
    delta_theta = -dtheta
    dx_c = dx_p * math.cos(delta_theta) - dy_p * math.sin(delta_theta)
    dy_c = dx_p * math.sin(delta_theta) + dy_p * math.cos(delta_theta)

    I_x = cx_c + dx_c
    I_y = cy_c + dy_c

    v = np.array([I_x - x_basis, I_y - y_basis])
    tx = np.dot(v, U1_VECTOR)
    ty = np.dot(v, U2_VECTOR)
    return (tx, ty), (dtheta, dtheta)


def _match_masks(
    prior_mask, current_mask, method: str = 'moments',
    x_basis: float = DEFAULT_X_BASIS, y_basis: float = DEFAULT_Y_BASIS,
    prior_img=None, current_img=None,
):
    """
    Match the current mask to the prior mask and return the required adjustment.

    Args:
        prior_mask, current_mask: uint8 edge masks (H, W).
        method: 'moments' | 'bbox' | 'ecc' | 'feature' | 'contour'.
        x_basis, y_basis: Fabric corner reference point (pixels).
        prior_img, current_img: Color images (required for 'feature' method).

    Returns:
        translation_vector: (tx, ty) in pixels.
        rotation_angles:    (r1, r2) in radians.
    """
    if method in ('moments', 'bbox'):
        if method == 'moments':
            (cx_p, cy_p), theta_p = _get_moments_stats(prior_mask)
            (cx_c, cy_c), theta_c = _get_moments_stats(current_mask)
        else:
            (cx_p, cy_p), theta_p = _get_bbox_stats(prior_mask)
            (cx_c, cy_c), theta_c = _get_bbox_stats(current_mask)

        dtheta = theta_p - theta_c
        dtheta = (dtheta + np.pi / 2) % np.pi - np.pi / 2

        dx_p = x_basis - cx_p
        dy_p = y_basis - cy_p
        delta_theta = -dtheta
        dx_c = dx_p * math.cos(delta_theta) - dy_p * math.sin(delta_theta)
        dy_c = dx_p * math.sin(delta_theta) + dy_p * math.cos(delta_theta)

        I_x = cx_c + dx_c
        I_y = cy_c + dy_c
        v = np.array([I_x - x_basis, I_y - y_basis])
        tx = np.dot(v, U1_VECTOR)
        ty = np.dot(v, U2_VECTOR)
        return (tx, ty), (dtheta, dtheta)

    elif method == 'ecc':
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 50, 1e-4)
        try:
            _, warp_matrix = cv.findTransformECC(
                prior_mask, current_mask, warp_matrix, cv.MOTION_EUCLIDEAN, criteria, None, 1
            )
            warp_inv = cv.invertAffineTransform(warp_matrix)
            pt = np.array([[[x_basis, y_basis]]], dtype=np.float32)
            pt_mapped = cv.transform(pt, warp_inv)[0][0]
            I_x, I_y = pt_mapped[0], pt_mapped[1]
            v = np.array([I_x - x_basis, I_y - y_basis])
            tx = np.dot(v, U1_VECTOR)
            ty = np.dot(v, U2_VECTOR)
            theta = math.atan2(warp_inv[1, 0], warp_inv[0, 0])
            return (tx, ty), (theta, theta)
        except cv.error:
            print("[mask_helpers] Warning: ECC failed to converge.")
            return (0.0, 0.0), (0.0, 0.0)

    elif method == 'feature':
        return _feature_matching(prior_img, current_img, prior_mask, current_mask, x_basis, y_basis)

    elif method == 'contour':
        return _contour_matching(prior_mask, current_mask, x_basis, y_basis)

    elif method == 'chamfer':
        return _chamfer_matching(prior_mask, current_mask, x_basis, y_basis, fast=False)
        
    elif method == 'chamfer_fast':
        return _chamfer_matching(prior_mask, current_mask, x_basis, y_basis, fast=True)

    else:
        raise ValueError(f"[mask_helpers] Unknown matching method: '{method}'")


# ---------------------------------------------------------------------------
# Batch signal filtering
# ---------------------------------------------------------------------------

def _filter_and_mean(buffer, tx_thresh: float = 15.0, ty_thresh: float = 15.0, r_thresh: float = 0.15):
    """
    Reject noisy predictions from a batch using greedy Hough-space clustering,
    then return the mean of the largest consensus cluster.

    This mirrors the LineStabilizer strategy from utils/tracking_helper.py.

    Args:
        buffer:    List of (tx, ty, r1, r2) tuples.
        tx_thresh: Max allowed |Δtx| within a cluster (pixels).
        ty_thresh: Max allowed |Δty| within a cluster (pixels).
        r_thresh:  Max allowed |Δr| within a cluster (radians).

    Returns:
        (mean_tx, mean_ty, mean_r1, mean_r2)
    """
    if not buffer:
        return 0.0, 0.0, 0.0, 0.0
    if len(buffer) == 1:
        return buffer[0]

    assigned = [False] * len(buffer)
    clusters = []

    for i in range(len(buffer)):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        tx_i, ty_i, r1_i, r2_i = buffer[i]

        for j in range(i + 1, len(buffer)):
            if assigned[j]:
                continue
            tx_j, ty_j, r1_j, r2_j = buffer[j]
            if (abs(tx_i - tx_j) < tx_thresh and
                abs(ty_i - ty_j) < ty_thresh and
                abs(r1_i - r1_j) < r_thresh and
                abs(r2_i - r2_j) < r_thresh):
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)

    largest = max(clusters, key=len)
    valid = [buffer[idx] for idx in largest]
    n = len(valid)
    return (
        sum(v[0] for v in valid) / n,
        sum(v[1] for v in valid) / n,
        sum(v[2] for v in valid) / n,
        sum(v[3] for v in valid) / n,
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _visualize_prior_adjustment(
    img, current_mask, prior_mask, control_signals, method: str,
    x_basis: float = DEFAULT_X_BASIS, y_basis: float = DEFAULT_Y_BASIS,
    save_dir: Optional[str] = None, frame_count: int = 0, show: bool = True,
    offset_x: int = 0, offset_y: int = 0,
):
    """
    Render a side-by-side overlay of the prior and current masks with
    control signal annotations.

    Args:
        img:             Current preprocessed BGR image.
        current_mask:    uint8 edge mask for the current frame.
        prior_mask:      uint8 edge mask for the reference frame.
        control_signals: ((tx, ty), (r1, r2)).
        method:          Matching method name (used for axis visualisation).
        save_dir:        If not None, save to ``<save_dir>/frame_NNNNN.jpg``.
        frame_count:     Index used for the saved filename.
        show:            If True, display via cv.imshow (main thread only!).

    Returns:
        vis: The annotated BGR image.
    """
    vis = img.copy()

    color_current = np.zeros_like(img)
    ch, cw = current_mask.shape
    color_current[offset_y:offset_y+ch, offset_x:offset_x+cw, 2] = current_mask
    vis = cv.addWeighted(vis, 1.0, color_current, 0.5, 0)

    color_prior = np.zeros_like(img)
    ph, pw = prior_mask.shape
    color_prior[offset_y:offset_y+ph, offset_x:offset_x+pw, 1] = prior_mask
    vis = cv.addWeighted(vis, 1.0, color_prior, 0.5, 0)

    draw_adjustor_overlay(vis, (x_basis, y_basis), AXIS_LENGTH_PX, control_signals=control_signals)

    tx, ty = control_signals[0]
    r1, r2 = control_signals[1]

    if method == 'moments':
        (cx_p, cy_p), theta_p = _get_moments_stats(prior_mask)
        (cx_c, cy_c), theta_c = _get_moments_stats(current_mask)
    elif method == 'bbox':
        (cx_p, cy_p), theta_p = _get_bbox_stats(prior_mask)
        (cx_c, cy_c), theta_c = _get_bbox_stats(current_mask)
    else:
        cx_p, cy_p = x_basis, y_basis
        theta_p = 0.0
        cx_c = x_basis - tx
        cy_c = y_basis - ty
        theta_c = -r1

    axis_len = 100
    cv.line(vis, (int(cx_p), int(cy_p)),
            (int(cx_p + axis_len * math.cos(theta_p)), int(cy_p + axis_len * math.sin(theta_p))),
            (0, 255, 0), 2)
    cv.circle(vis, (int(cx_p), int(cy_p)), 5, (0, 255, 0), -1)

    cv.line(vis, (int(cx_c), int(cy_c)),
            (int(cx_c + axis_len * math.cos(theta_c)), int(cy_c + axis_len * math.sin(theta_c))),
            (0, 0, 255), 2)
    cv.circle(vis, (int(cx_c), int(cy_c)), 5, (0, 0, 255), -1)

    cv.putText(vis, f"Method: {method}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Crop to a 300x300 window around the basis for clearer visualization
    roi_size = 300
    h, w = vis.shape[:2]
    sx = int(max(0, x_basis - roi_size / 2))
    sy = int(max(0, y_basis - roi_size / 2))
    ex = int(min(w, sx + roi_size))
    ey = int(min(h, sy + roi_size))
    vis_cropped = vis[sy:ey, sx:ex]

    if show:
        cv.imshow("Prior Mask Adjustment", vis_cropped)
        cv.waitKey(1)

    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        cv.imwrite(os.path.join(save_dir, f"frame_{frame_count:05d}.jpg"), vis_cropped)

    return vis_cropped
