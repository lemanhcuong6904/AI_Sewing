import serial
import time
import sys
import tty
import termios
import select
import os
import cv2 as cv
import numpy as np
from datetime import datetime

from utils.camera import get_camera, DEFAULT_CROP_ROI
from utils.line_detectors.detector import DetectorMode, preprocessing_sewer, detect_lines, postprocess_lines, keep_lines_near_background
from utils.helper import _get_control_point, _compute_control_signal, draw_lines

def main():
    # Cấu hình cổng Serial (thay đổi '/dev/ttyTHS1' hoặc '/dev/ttyUSB0' cho phù hợp với Jetson của bạn)
    serial_port = '/dev/ttyUSB0' 
    baud_rate = 115200
    ser = None

    try:
        # Khởi tạo kết nối Serial
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        print(f"Đã kết nối tới {serial_port} với baud rate {baud_rate}.")
    except Exception as e:
        print(f"Lỗi khi mở cổng serial: {e}")
        print("Vui lòng kiểm tra lại tên cổng Serial hoặc quyền truy cập (sudo chmod a+rw /dev/ttyTHS1)")
        print("Tiếp tục mà không có Serial (chế độ test camera/logic).")

    print("\n--- GIAO DIỆN ĐIỀU KHIỂN TỐC ĐỘ ĐỘNG CƠ ---")
    print("Nhấn các phím sau (trên Terminal hoặc cửa sổ hình ảnh) để điều khiển:")
    print("  '1' - Tốc độ Chậm")
    print("  '2' - Tốc độ Trung bình")
    print("  '3' - Tốc độ Nhanh")
    print("  '0' - Tắt máy")
    print("  'a' - Dừng 2 motor")
    print("  'b' - Chạy nghịch 2 motor")
    print("  'q' - Thoát chương trình")
    print("-------------------------------------------\n")

    cam = get_camera()
    if cam is None:
        print("Không thể mở camera.")
        if ser:
            ser.close()
        return

    time.sleep(2)

    base_capture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captured_frames")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_capture_dir = os.path.join(base_capture_dir, f"{current_time}_uart")
    os.makedirs(session_capture_dir, exist_ok=True)
    frame_count = 0

    img_size = (128, 128)
    enhance_contrast = True
    smooth_texture = True
    detector_mode = DetectorMode.LSD
    mask_thresh = 30

    current_signal = '-'
    
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setraw(fd)

    def write_serial(cmd):
        if ser:
            ser.write(cmd.encode('utf-8'))
            ser.flush()

    try:
        while True:
            key_pressed = None
            if select.select([sys.stdin], [], [], 0)[0]:
                key_pressed = sys.stdin.read(1).strip().lower()
            
            key_cv = cv.waitKey(1) & 0xFF
            if key_cv != 255:
                key_pressed = chr(key_cv).lower()

            if key_pressed == 'q':
                write_serial('0')
                print("\r\nĐang thoát chương trình...\r\n")
                break
            elif key_pressed in ['1', '2', '3', '0', 'a', 'b']:
                current_signal = key_pressed
                write_serial(current_signal)
                
                speed_str = ""
                if current_signal == '1': speed_str = "Chậm"
                elif current_signal == '2': speed_str = "Trung bình"
                elif current_signal == '3': speed_str = "Nhanh"
                elif current_signal == '0': speed_str = "Tắt máy"
                elif current_signal == 'a': speed_str = "Dừng 2 motor"
                elif current_signal == 'b': speed_str = "Nghịch 2 motor"
                
                print(f"\r\n-> Đã gửi: {current_signal} (Mã lệnh: {speed_str})\r\n", end="")

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

            lines = detect_lines(gray, detector_mode, img)
            if detector_mode != DetectorMode.MLSD:
                lines = postprocess_lines(lines)
                lines = keep_lines_near_background(lines, gray, mask_thresh)
            
            if lines is not None:
                lines = lines.astype(np.float32)

            control_pt1, control_pt2 = _get_control_point(lines, img.shape)
            error1, error2 = _compute_control_signal(control_pt1, control_pt2, img.shape)

            drawn_img = img.copy()
            if lines is not None:
                drawn_img = draw_lines(drawn_img, lines, detector_mode)

            y_ofs1 = 60
            y_ofs2 = 20

            text1 = f"{int(error1)} [{current_signal}]" if error1 is not None else f"NaN [{current_signal}]"
            text2 = f"{int(error2)} [{current_signal}]" if error2 is not None else f"NaN [{current_signal}]"
            
            cv.putText(drawn_img, text1, (10, y_ofs1), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv.putText(drawn_img, text2, (10, y_ofs2), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
            combined_vis = cv.hconcat([gray_bgr, drawn_img])
            
            zoomed_vis = cv.resize(combined_vis, None, fx=3.0, fy=3.0, interpolation=cv.INTER_NEAREST)
            
            frame_filename = os.path.join(session_capture_dir, f"{current_time}_uart_{frame_count:05d}.jpg")
            cv.imwrite(frame_filename, zoomed_vis)
            frame_count += 1
            
            cv.imshow("UART Visual Control", zoomed_vis)

    except KeyboardInterrupt:
        print("\r\nChương trình bị dừng bởi người dùng.\r\n")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        if ser and ser.is_open:
            ser.close()
            print("\r\nĐã đóng kết nối Serial.\r\n")
        if cam is not None:
            cam.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()