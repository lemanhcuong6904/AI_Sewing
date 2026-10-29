"""
evaluate_adjustor.py
--------------------
Offline evaluation of the fabric adjustor pipeline on a validation dataset.

Directory layout expected under ./adjustor_data/<dir>/:
    data/    – PNG images named 0.png, 1.png, ...
    label/   – matching text files named 0.txt, 1.txt, ...
               Each label file contains one line: tx_cm ty_cm angle_h_deg angle_v_deg

Usage:
    python evaluate_adjustor.py --dir square-no_light
    python evaluate_adjustor.py --dir square-no_light --detector lsd --x-basis 550 --y-basis 50

Output:
    ./adjustor_data/<dir>/eval.txt      – per-sample predictions + mean errors summary
    ./adjustor_data/<dir>/eval_vis/     – visualization images with detected lines overlaid
"""

from utils.helper import draw_lines
import os
import sys
import math
import argparse

import cv2 as cv
import numpy as np

from utils.line_detectors.detector import DetectorMode
from utils.viz_adjustor import draw_adjustor_overlay
from modules.fabric_adjustor import _perform_line_detection, _calculate_control_signals

from utils.constants import DEFAULT_X_BASIS, DEFAULT_Y_BASIS, PX_PER_CM_X, PX_PER_CM_Y, AXIS_LENGTH_PX


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the fabric adjustor pipeline offline.")
    p.add_argument("--dir", required=True,
                   help="Subfolder name under ./adjustor_data/  (e.g. square-no_light)")
    p.add_argument("--detector", default="mlsd",
                   choices=["lsd", "fld", "edlines", "ensemble", "mlsd", "dexined"],
                   help="Line detector backend (default: mlsd)")
    p.add_argument("--prior-img", type=str, default=None,
                   help="Name of the prior fabric image in ./mask_priors (required for dexined detector)")
    p.add_argument("--match-method", type=str, default="moments", choices=["moments", "ecc", "bbox", "feature", "contour", "chamfer", "chamfer_fast"],
                   help="Matching algorithm for dexined masks")
    p.add_argument("--morphology", type=str, default="fill", choices=["none", "erode", "open", "fill"],
                   help="Morphological operation to strip noise from masks")
    p.add_argument("--x-basis", type=float, default=DEFAULT_X_BASIS, dest="x_basis",
                   help=f"X coordinate of the origin (default: {DEFAULT_X_BASIS})")
    p.add_argument("--y-basis", type=float, default=DEFAULT_Y_BASIS, dest="y_basis",
                   help=f"Y coordinate of the origin (default: {DEFAULT_Y_BASIS})")
    p.add_argument("--enhance-contrast", action="store_true", dest="enhance_contrast",
                   help="Enable contrast enhancement in preprocessing")
    p.add_argument("--no-smooth", action="store_false", dest="smooth_texture",
                   help="Disable texture smoothing in preprocessing")
    p.add_argument("--mask-thresh", type=int, default=30, dest="mask_thresh",
                   help="Background mask threshold (default: 30)")  
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_label(path: str):
    """Read a label file and return (tx_cm, ty_cm, angle_h_deg, angle_v_deg)."""
    with open(path, "r") as f:
        parts = f.read().strip().split()
    if len(parts) != 4:
        raise ValueError(f"Expected 4 values in '{path}', got {len(parts)}")
    return tuple(float(v) for v in parts)


from modules.fabric_adjustor_prior import apply_morphology

def predict(img_path: str, x_basis: float, y_basis: float,
            detector_mode: DetectorMode,
            enhance_contrast: bool, smooth_texture: bool,
            mask_thresh: int,
            model=None, prior_mask=None, prior_img=None, match_method='moments', morphology='none'):
    """Run the adjustor pipeline on a single image.

    Returns:
        ((tx_cm, ty_cm, angle_h_deg, angle_v_deg), vis_img)  on success
        None  when no lines are detected
    """
    frame = cv.imread(img_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    if detector_mode == DetectorMode.DEXINED:
        from modules.fabric_adjustor_prior import _match_masks, _visualize_prior_adjustment
        from utils.edge_detectors.dexined_inference import infer_dexined
        from utils.line_detectors.detector import preprocessing_adjustor

        gray, img, offset = preprocessing_adjustor(
             frame, img_size=None, crop_roi=None,
             enhance_contrast=enhance_contrast, smooth_texture=smooth_texture
        )
        current_mask = infer_dexined(model, img, device="cpu")
        current_mask = apply_morphology(current_mask, method=morphology)
        
        trans_vec, rot_angles = _match_masks(prior_mask, current_mask, method=match_method, x_basis=x_basis, y_basis=y_basis, prior_img=prior_img, current_img=img)
        
        tx_px, ty_px = trans_vec
        angle_rad, _ = rot_angles
        
        tx_cm = tx_px / PX_PER_CM_X
        ty_cm = ty_px / PX_PER_CM_Y
        angle_h = math.degrees(angle_rad)
        angle_v = math.degrees(angle_rad)
        
        vis = _visualize_prior_adjustment(
            frame, current_mask, prior_mask, (trans_vec, rot_angles), match_method,
            x_basis=x_basis, y_basis=y_basis, save_dir=None, frame_count=0, show=False,
            offset_x=offset[0], offset_y=offset[1]
        )
        
        return (tx_cm, ty_cm, angle_h, angle_v), vis

    gray, img, lines, best_h_line, best_v_line, _ = _perform_line_detection(
        frame,
        img_size=None,
        enhance_contrast=enhance_contrast,
        smooth_texture=smooth_texture,
        detector_mode=detector_mode,
        mask_thresh=mask_thresh,
        x_basis=x_basis,
        y_basis=y_basis,
    )

    if best_h_line is None or best_v_line is None:
        # Still produce a visualization showing the raw detection state
        vis = frame.copy()
        draw_adjustor_overlay(vis, (x_basis, y_basis), axis_length=AXIS_LENGTH_PX)
        cv.putText(vis, "NO LINES", (5, vis.shape[0] - 8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        return None, vis

    trans_vec, rot_angles = _calculate_control_signals(
        x_basis, y_basis, best_h_line, best_v_line
    )

    tx_px, ty_px = trans_vec
    angle_h, angle_v = rot_angles

    tx_cm = tx_px / PX_PER_CM_X
    ty_cm = ty_px / PX_PER_CM_Y

    # Build visualization
    vis = frame.copy()
    control_signals = ((tx_px, ty_px), rot_angles)   # pixels/radians for overlay helper
    vis = draw_lines(vis, lines, detector_mode)
    draw_adjustor_overlay(
        vis,
        basis=(x_basis, y_basis),
        axis_length=AXIS_LENGTH_PX,
        best_h_line=best_h_line,
        best_v_line=best_v_line,
        control_signals=control_signals,
    )

    return (tx_cm, ty_cm, angle_h, angle_v), vis


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.dir.startswith("adjustor_data/") or args.dir.startswith("./adjustor_data/"):
        base_dir = args.dir
    else:
        base_dir  = os.path.join("adjustor_data", args.dir)
        
    data_dir  = os.path.join(base_dir, "data")
    label_dir = os.path.join(base_dir, "label")
    vis_dir   = os.path.join(base_dir, "eval_vis")
    out_path  = os.path.join(base_dir, "eval.txt")

    os.makedirs(vis_dir, exist_ok=True)

    for d in (data_dir, label_dir):
        if not os.path.isdir(d):
            print(f"Error: directory not found: {d}")
            sys.exit(1)

    detector_mode = DetectorMode(args.detector)

    model = None
    prior_mask = None
    prior_img = None
    if detector_mode == DetectorMode.DEXINED:
        if not args.prior_img:
            print("Error: --prior-img must be provided when using the dexined detector.")
            sys.exit(1)
            
        from utils.edge_detectors.dexined_inference import load_dexined, infer_dexined
        from utils.line_detectors.detector import preprocessing_adjustor
        
        print("Loading DexiNed model...")
        model = load_dexined()
        
        prior_img_path = os.path.join("mask_priors", args.prior_img)
        base_name, ext = os.path.splitext(prior_img_path)
        saved_mask_path = f"{base_name}_mask.png"
        
        # Load prior_img (needed for feature matching)
        raw_prior = cv.imread(prior_img_path)
        if raw_prior is None:
            print(f"Error: Could not read prior image: {prior_img_path}")
            sys.exit(1)
            
        _, prior_img, _ = preprocessing_adjustor(
             raw_prior, img_size=None, crop_roi=None,
             enhance_contrast=args.enhance_contrast, smooth_texture=args.smooth_texture
        )
        
        if os.path.exists(saved_mask_path):
            print(f"Loading cached prior mask from {saved_mask_path}...")
            prior_mask = cv.imread(saved_mask_path, cv.IMREAD_GRAYSCALE)
        else:
            print(f"Generating prior mask from {prior_img_path}...")
            prior_mask = infer_dexined(model, prior_img, device="cpu")
            cv.imwrite(saved_mask_path, prior_mask)
            print(f"Saved prior mask to {saved_mask_path}")
            
        prior_mask = apply_morphology(prior_mask, method=args.morphology)

    # Collect numeric indices present in the data folder
    indices = sorted(
        int(os.path.splitext(f)[0])
        for f in os.listdir(data_dir)
        if f.endswith(".png") and os.path.splitext(f)[0].lstrip("-").isdigit()
    )

    if not indices:
        print(f"No PNG images found in {data_dir}")
        sys.exit(1)

    print(f"Evaluating {len(indices)} images from '{base_dir}' "
          f"using detector={args.detector}\n")

    output_lines   = []
    errors_trans   = []
    errors_angle_h = []
    errors_angle_v = []
    skipped        = []

    for idx in indices:
        img_path = os.path.join(data_dir,  f"{idx}.png")
        lbl_path = os.path.join(label_dir, f"{idx}.txt")

        if not os.path.exists(lbl_path):
            print(f"  [{idx:4d}] WARN: missing label, skipping.")
            skipped.append(idx)
            continue

        try:
            gt_tx, gt_ty, gt_ah, gt_av = load_label(lbl_path)
        except (ValueError, FileNotFoundError) as e:
            print(f"  [{idx:4d}] WARN: {e}")
            skipped.append(idx)
            continue

        # Skip samples whose ground-truth angles are outside [-30, +30] degrees
        ANGLE_LIMIT = 30
        if abs(gt_ah) > ANGLE_LIMIT or abs(gt_av) > ANGLE_LIMIT:
            print(f"  [{idx:4d}] SKIP: angles ({gt_ah:+.1f}, {gt_av:+.1f}) outside [-{ANGLE_LIMIT}, +{ANGLE_LIMIT}] deg")
            skipped.append(idx)
            continue

        # Skip samples whose ground-truth translation vector magnitude exceeds the radius limit
        TRANS_RADIUS_LIMIT = 5
        gt_trans_radius = math.sqrt(gt_tx ** 2 + gt_ty ** 2)
        if gt_trans_radius > TRANS_RADIUS_LIMIT:
            print(f"  [{idx:4d}] SKIP: translation radius {gt_trans_radius:.2f} cm > {TRANS_RADIUS_LIMIT} cm")
            skipped.append(idx)
            continue

        # Skip samples whose ground-truth translation vector magnitude exceeds the radius limit
        # MAX_LENGTH_LIMIT = 12.0
        # if abs(gt_tx) > MAX_LENGTH_LIMIT or abs(gt_ty) > MAX_LENGTH_LIMIT:
        #     print(f" [{idx:4d}] SKIP: max length {MAX_LENGTH_LIMIT}cm")
        #     skipped.append(idx)
        #     continue

        try:
            pred_result, vis_img = predict(
                img_path, args.x_basis, args.y_basis,
                detector_mode,
                args.enhance_contrast, args.smooth_texture,
                args.mask_thresh,
                model=model, prior_mask=prior_mask, prior_img=prior_img, match_method=args.match_method,
                morphology=args.morphology
            )
        except Exception as e:
            print(f"  [{idx:4d}] ERROR during prediction: {e}")
            output_lines.append(
                f"{idx:4d}: FAILED ({e})  "
                f"gt=({gt_tx:.3f}, {gt_ty:.3f}, {gt_ah:.2f}, {gt_av:.2f})"
            )
            skipped.append(idx)
            continue

        # Always save the visualization image
        vis_path = os.path.join(vis_dir, f"{idx}.png")
        cv.imwrite(vis_path, vis_img)

        if pred_result is None:
            msg = (f"{idx:4d}: NO_LINES  "
                   f"gt=({gt_tx:.3f}, {gt_ty:.3f}, {gt_ah:.2f}, {gt_av:.2f})")
            print(f"  [{idx:4d}] WARN: no lines detected.")
            output_lines.append(msg)
            skipped.append(idx)
            continue

        pred_tx, pred_ty, pred_ah, pred_av = pred_result

        err_trans = math.sqrt((pred_tx - gt_tx) ** 2 + (pred_ty - gt_ty) ** 2)
        err_ah    = abs(pred_ah - gt_ah)
        err_av    = abs(pred_av - gt_av)

        errors_trans.append(err_trans)
        errors_angle_h.append(err_ah)
        errors_angle_v.append(err_av)

        line = (
            f"{idx:4d}: "
            f"pred=({pred_tx:+.3f}, {pred_ty:+.3f}, {pred_ah:+.2f}, {pred_av:+.2f})  "
            f"gt=({gt_tx:+.3f}, {gt_ty:+.3f}, {gt_ah:+.2f}, {gt_av:+.2f})  "
            f"err=(trans:{err_trans:.3f}cm  H:{err_ah:.2f}deg  V:{err_av:.2f}deg)"
        )
        output_lines.append(line)
        print(f"  {line}")

    # ---- Summary ----
    n_valid = len(errors_trans)
    n_total = len(indices)

    if n_valid > 0:
        mean_trans = sum(errors_trans)   / n_valid
        mean_ah    = sum(errors_angle_h) / n_valid
        mean_av    = sum(errors_angle_v) / n_valid
        summary = (
            f"MEAN_ERRORS  valid={n_valid}/{n_total}  "
            f"trans={mean_trans:.4f}cm  "
            f"angle_h={mean_ah:.4f}deg  "
            f"angle_v={mean_av:.4f}deg"
        )
    else:
        summary = f"MEAN_ERRORS  valid=0/{n_total}  (no successful predictions)"

    print(f"\n{summary}")

    # ---- Write output ----
    with open(out_path, "w") as f:
        f.write("\n".join(output_lines))
        f.write("\n")
        f.write(summary + "\n")

    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
