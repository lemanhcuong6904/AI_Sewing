"""Thu thập ảnh từ camera để test detect góc vải vuông (chạy trên Windows).

Phím tắt (cửa sổ preview):
    SPACE / s : lưu 1 ảnh
    a         : bật/tắt tự động lưu (chỉ lưu khi frame thay đổi so với ảnh đã lưu gần nhất)
    q / ESC   : thoát

Chạy:
    python cuonglm/curve_matcher/collect_frames.py
"""
import time
from pathlib import Path

import cv2 as cv

# ============================ CẤU HÌNH ============================
OUT_DIR = "test_frames/square_01"   # thư mục lưu ảnh
PREFIX = ""                    # tiền tố tên file: frame_0000.png
CAM_INDEX = 0                       # camera index
WIDTH = 640
HEIGHT = 480
AUTO_START = True                  # True: bắt đầu ở chế độ tự động lưu
PIXEL_DIFF = 25                     # 1 pixel coi là "đổi" nếu chênh độ sáng > giá trị này (0-255)
CHANGE_FRAC = 0.01                  # lưu khi > 1% số pixel đổi so với ảnh đã lưu gần nhất
MIN_GAP = 0.5                       # giây tối thiểu giữa 2 lần tự lưu (tránh lưu dồn khi đang di chuyển)
# ==================================================================


def small_gray(frame):
    g = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    g = cv.resize(g, (160, 120), interpolation=cv.INTER_AREA)
    return cv.GaussianBlur(g, (5, 5), 0)          # làm mờ để nhiễu cảm biến không tính là thay đổi


def changed(ref, cur) -> bool:
    if ref is None:
        return True
    return float((cv.absdiff(ref, cur) > PIXEL_DIFF).mean()) > CHANGE_FRAC


def open_camera(index: int, width: int, height: int) -> cv.VideoCapture:
    # DSHOW mở nhanh hơn MSMF trên Windows; thử lần lượt các backend.
    for backend in (cv.CAP_DSHOW, cv.CAP_MSMF, cv.CAP_ANY):
        cap = cv.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()
    raise RuntimeError(f"Không mở được camera index {index}")


def next_index(out_dir: Path, prefix: str) -> int:
    nums = []
    for p in out_dir.glob(f"{prefix}_*.png"):
        tail = p.stem.rsplit("_", 1)[-1]
        if tail.isdigit():
            nums.append(int(tail))
    return max(nums, default=-1) + 1


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = next_index(out_dir, PREFIX)

    cap = open_camera(CAM_INDEX, WIDTH, HEIGHT)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(f"[collect] Camera {CAM_INDEX} mở {w}x{h}, lưu vào {out_dir.resolve()}")
    print("[collect] SPACE/s: lưu | a: auto | q/ESC: thoát")

    auto = AUTO_START
    last_save = 0.0
    flash_until = 0.0
    saved = 0
    ref = None                                      # ảnh (nhỏ, xám) đã lưu gần nhất

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[collect] Đọc frame thất bại")
                break

            now = time.time()
            cur = small_gray(frame)
            do_save = auto and now - last_save >= MIN_GAP and changed(ref, cur)

            key = cv.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord(" "), ord("s")):
                do_save = True
            if key == ord("a"):
                auto = not auto
                ref = None                          # bật auto: lưu ngay frame đầu tiên làm mốc
                print(f"[collect] Auto {'BẬT' if auto else 'TẮT'}")

            if do_save:
                path = out_dir / f"{PREFIX}_{idx:04d}.png"
                cv.imwrite(str(path), frame)  # lưu frame gốc, chưa vẽ overlay
                print(f"[collect] Đã lưu {path.name}")
                idx += 1
                saved += 1
                last_save = now
                ref = cur
                flash_until = now + 0.15

            view = frame.copy()
            status = f"saved: {saved} | auto: {'ON' if auto else 'OFF'}"
            cv.putText(view, status, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if now < flash_until:
                cv.rectangle(view, (0, 0), (w - 1, h - 1), (255, 255, 255), 8)
            cv.imshow("collect_frames", view)
    finally:
        cap.release()
        cv.destroyAllWindows()
        print(f"[collect] Xong, đã lưu {saved} ảnh")


if __name__ == "__main__":
    main()
