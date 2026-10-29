import cv2 as cv
import os
import sys
from datetime import datetime
from utils.camera import get_camera
from utils.viz_adjustor import (

    draw_adjustor_overlay,
    draw_target_axes,
)

from utils.constants import DEFAULT_X_BASIS, DEFAULT_Y_BASIS, PX_PER_CM_X, PX_PER_CM_Y, AXIS_LENGTH_PX

BASIS = (DEFAULT_X_BASIS, DEFAULT_Y_BASIS)                  # fixed origin in the camera frame


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _crop_frame(frame):
    h = frame.shape[0]
    if h > 392:
        frame = frame[0:392, :]
    return frame


def _load_labels(path):
    """Parse label file: each non-empty line is 'tx ty angle_h angle_v'."""
    labels = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                try:
                    labels.append(tuple(float(p) for p in parts))
                except ValueError:
                    pass  # skip malformed lines
    return labels


# ---------------------------------------------------------------------------
# Mode 1: Manual (original flow)
# ---------------------------------------------------------------------------

def main():
    camera = get_camera()
    if camera is None:
        print("Error: Could not initialize camera.")
        return

    # Create session directory
    session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("adjustor_data", session_time)
    data_dir = os.path.join(base_dir, "data")
    label_dir = os.path.join(base_dir, "label")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    
    print(f"Session started. Data will be saved to: {base_dir}")
    print("Press 'Space' to capture an image and enter ground truth.")
    print("Press 'q' to quit.")

    index = 0
    window_name = "Camera Feed - Press Space to Capture, q to Quit"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)

    try:
        while True:
            ret, frame, is_new = camera.read()
            if not ret or frame is None:
                cv.waitKey(10)
                continue
            # frame = _crop_frame(frame)
            display_frame = frame.copy()
            
            draw_adjustor_overlay(display_frame, BASIS, axis_length=AXIS_LENGTH_PX)

            cv.imshow(window_name, display_frame)
            key = cv.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Exiting...")
                break
            elif key == ord(' '):
                print(f"\n--- Capture {index} ---")
                
                img_path = os.path.join(data_dir, f"{index}.png")
                cv.imwrite(img_path, frame)
                print(f"Saved image to {img_path}")
                
                while True:
                    user_input = input("Enter ground truth (trans_vector_x trans_vector_y rotating_angle_horizontal rotating_angle_vertical): ")
                    parts = user_input.strip().split()
                    if len(parts) == 4:
                        try:
                            [float(p) for p in parts]
                            break
                        except ValueError:
                            print("Error: Please enter numeric values only.")
                    else:
                        print("Error: Please enter exactly 4 values separated by space.")
                
                label_path = os.path.join(label_dir, f"{index}.txt")
                with open(label_path, 'w') as f:
                    f.write(user_input.strip())
                print(f"Saved label to {label_path}")
                
                index += 1
                print("\nResuming camera feed... Press 'Space' for next capture or 'q' to quit.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        camera.release()
        cv.destroyAllWindows()


# ---------------------------------------------------------------------------
# Mode 2: Semi-automatic (guided by a pre-loaded label file)
# ---------------------------------------------------------------------------

def main_semi_auto(label_file: str):
    """Guided labeling: read labels from *label_file*, show the target axes,
    user adjusts the fabric, then presses Space to capture + auto-save label.

    Label file format (one label per line):
        tx_cm  ty_cm  angle_h_deg  angle_v_deg
    """
    labels = _load_labels(label_file)
    if not labels:
        print(f"Error: no valid labels found in '{label_file}'.")
        return
    print(f"Loaded {len(labels)} labels from '{label_file}'.")

    camera = get_camera()
    if camera is None:
        print("Error: Could not initialize camera.")
        return

    # Create session directory (named after the label file stem)
    label_stem = os.path.splitext(os.path.basename(label_file))[0]
    session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("adjustor_data", f"{session_time}_{label_stem}")
    data_dir = os.path.join(base_dir, "data")
    label_dir = os.path.join(base_dir, "label")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    print(f"Session data will be saved to: {base_dir}")

    window_name = "Semi-Auto Labeling — Space: capture  |  q: quit  |  s: skip"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)

    idx = 0          # index into labels list
    captured = 0     # how many images have been saved

    print("\nControls:  Space = capture & advance  |  s = skip label  |  q = quit\n")

    try:
        while idx < len(labels):
            tx, ty, angle_h, angle_v = labels[idx]

            # ---- HUD header -----
            progress = f"[{idx + 1}/{len(labels)}]"
            label_str = fr"tx={tx:+.1f}cm  ty={ty:+.1f}cm  H={angle_h:+.1f} deg  V={angle_v:+.1f} deg"

            ret, frame, _ = camera.read()
            if not ret or frame is None:
                cv.waitKey(10)
                continue
            # frame = _crop_frame(frame)
            display_frame = frame.copy()

            # Base coordinate system
            draw_adjustor_overlay(display_frame, BASIS, axis_length=AXIS_LENGTH_PX)

            # Target lines for the current label
            draw_target_axes(display_frame, BASIS, tx, ty, angle_h, angle_v)

            # Progress text
            h_img = display_frame.shape[0]
            cv.putText(display_frame, progress, (5, 18),
                       cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
            cv.putText(display_frame, label_str, (5, 36),
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv.putText(display_frame, "Align fabric to YELLOW+MAGENTA lines, then press SPACE",
                       (5, h_img - 8), cv.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

            cv.imshow(window_name, display_frame)
            key = cv.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Quitting early.")
                break

            elif key == ord('s'):
                print(f"  Skipped label {idx + 1}: {label_str}")
                idx += 1

            elif key == ord(' '):
                # Save image
                img_path = os.path.join(data_dir, f"{captured}.png")
                cv.imwrite(img_path, frame)

                # Save pre-loaded label
                label_path = os.path.join(label_dir, f"{captured}.txt")
                with open(label_path, 'w') as f:
                    f.write(f"{tx} {ty} {angle_h} {angle_v}")

                print(f"  Captured #{captured}: {label_str}  →  {img_path}")
                captured += 1
                idx += 1

        if idx >= len(labels):
            print(f"\nAll {len(labels)} labels processed. {captured} images captured.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        camera.release()
        cv.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--semi-auto" in sys.argv:
        # Usage: python data_collector_adjustor.py --semi-auto <label_file>
        try:
            lf = sys.argv[sys.argv.index("--semi-auto") + 1]
        except IndexError:
            print("Usage: python data_collector_adjustor.py --semi-auto <path/to/labels.txt>")
            sys.exit(1)
        main_semi_auto(lf)
    else:
        main()
