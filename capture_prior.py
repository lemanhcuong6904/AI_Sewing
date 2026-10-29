import cv2 as cv
import os
import numpy as np
from utils.camera import get_camera
from utils.viz_adjustor import draw_adjustor_overlay
from utils.constants import DEFAULT_X_BASIS, DEFAULT_Y_BASIS, AXIS_LENGTH_PX
from utils.helper import _draw_lines_colour
from modules.fabric_adjustor import NonOrthoLineAdjustor
from utils.line_detectors.detector import DetectorMode
from utils.tracking_helper import LineStabilizer
import argparse
import math
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Capture a prior image for the fabric adjustor.")
    parser.add_argument("--mode", type=str, choices=["line", "edge"], default="line",
                        help="Capture mode: 'line' for MLSD arbitrary angle prior, 'edge' for DexiNed mask prior.")
    return parser.parse_args()

def main():
    args = parse_args()
    camera = get_camera()
    if camera is None:
        print("Error: Could not initialize camera.")
        return

    os.makedirs("mask_priors", exist_ok=True)
    
    print("--- Prior Mask Capture ---")
    print("Align the fabric to the corner (origin).")
    print("Press 'Space' to capture an image.")
    print("Press 'q' to quit without saving.")

    window_name = "Prior Capture - Space to Capture, q to Quit"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)

    captured_frame = None
    
    adjustor = None
    stabilizer = None
    stable_alpha = None
    
    if args.mode == 'line':
        adjustor = NonOrthoLineAdjustor(
            x_basis=DEFAULT_X_BASIS, 
            y_basis=DEFAULT_Y_BASIS,
            detector_mode=DetectorMode.MLSD
        )
        stabilizer = LineStabilizer(window=55)

    try:
        while True:
            ret, frame, is_new = camera.read()
            if not ret or frame is None:
                cv.waitKey(10)
                continue
            
            display_frame = frame.copy()
            draw_adjustor_overlay(display_frame, (DEFAULT_X_BASIS, DEFAULT_Y_BASIS), axis_length=AXIS_LENGTH_PX)
            
            if args.mode == 'line' and adjustor is not None and is_new:
                try:
                    adjustor.extract_prior(frame=frame)
                    stab_h, stab_v = stabilizer.update(adjustor.prior_h_line, adjustor.prior_v_line)
                    
                    if stab_v is not None and stab_h is not None:
                        _draw_lines_colour(display_frame, np.array([stab_v]), (0, 0, 255), 1) # Red for Vertical-ish
                        _draw_lines_colour(display_frame, np.array([stab_h]), (255, 0, 0), 1) # Blue for Horizontal-ish
                        
                        dx1 = stab_h[2] - stab_h[0]
                        dy1 = stab_h[3] - stab_h[1]
                        dx2 = stab_v[2] - stab_v[0]
                        dy2 = stab_v[3] - stab_v[1]
                        angle1 = math.atan2(dy1, dx1)
                        angle2 = math.atan2(dy2, dx2)
                        angle_diff = angle1 - angle2
                        diff_deg = abs(math.degrees(angle_diff)) % 180.0
                        if diff_deg > 90.0:
                            diff_deg = 180.0 - diff_deg
                        
                        stable_alpha = diff_deg
                        cv.putText(display_frame, f"Live Stable Angle: {stable_alpha:.1f}", 
                                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                except Exception:
                    # Ignore errors during live view (e.g. no lines found)
                    pass

            cv.imshow(window_name, display_frame)
            key = cv.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Exiting...")
                break
            elif key == ord(' '):
                captured_frame = frame.copy()
                
                if args.mode == 'line':
                    print("\nCaptured! Freezing stable prior lines...")
                    if stable_alpha is None:
                        print("Warning: No stable lines detected yet! Extracting raw fallback...")
                        try:
                            adjustor.extract_prior(frame=captured_frame)
                        except Exception as e:
                            print(f"Failed to extract prior lines: {e}")
                            print("Please readjust the fabric and try again.")
                            captured_frame = None
                            continue
                    else:
                        print("Using stabilized line state!")
                        adjustor.alpha_prior = stable_alpha
                        
                    # Visualize the chosen lines
                    vis_frame = captured_frame.copy()
                    
                    if stable_alpha is not None and stab_v is not None and stab_h is not None:
                        _draw_lines_colour(vis_frame, np.array([stab_v]), (0, 0, 255), 1)
                        _draw_lines_colour(vis_frame, np.array([stab_h]), (255, 0, 0), 1)
                    else:
                        if adjustor.prior_v_line:
                            _draw_lines_colour(vis_frame, np.array([adjustor.prior_v_line]), (0, 0, 255), 1)
                        if adjustor.prior_h_line:
                            _draw_lines_colour(vis_frame, np.array([adjustor.prior_h_line]), (255, 0, 0), 3)
                            
                    draw_adjustor_overlay(vis_frame, (DEFAULT_X_BASIS, DEFAULT_Y_BASIS), axis_length=AXIS_LENGTH_PX)
                    
                    # Display the confirmation window
                    cv.imshow(window_name, vis_frame)
                    print(f"Detected prior angle: {adjustor.alpha_prior:.2f}°")
                    print("Press 's' to Save, 'r' to Retry capture.")
                    
                    while True:
                        confirm_key = cv.waitKey(0) & 0xFF
                        if confirm_key == ord('s'):
                            print("Saving...")
                            break
                        elif confirm_key == ord('r'):
                            print("Retrying capture...")
                            captured_frame = None
                            break
                            
                    if captured_frame is not None:
                        break # User hit 's', break out of capture loop
                else:
                    # mode == 'edge'
                    print("\nCaptured! Freezing frame...")
                    break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cv.destroyAllWindows()
        camera.release()

    if captured_frame is not None:
        while True:
            name = input("Enter a name for this prior mask (e.g. '1', without extension), or 'q' to cancel: ").strip()
            if not name:
                continue
            if name.lower() == 'q':
                print("Cancelled.")
                break
            
            if name.endswith('.png'):
                name = name[:-4]
            elif name.endswith('.json'):
                name = name[:-5]

            img_path = os.path.join("mask_priors", f"{name}.png")
            cv.imwrite(img_path, captured_frame)
            print(f"Success! Saved prior mask to {img_path}")
            
            if args.mode == 'line' and adjustor is not None and adjustor.alpha_prior is not None:
                json_path = os.path.join("mask_priors", f"{name}.json")
                with open(json_path, 'w') as f:
                    json.dump({"alpha_prior": float(adjustor.alpha_prior)}, f, indent=4)
                print(f"Success! Saved stable prior angle to {json_path}")
                
            break

if __name__ == '__main__':
    main()
