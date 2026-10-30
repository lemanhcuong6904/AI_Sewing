"""Record the camera to a video file at a low frame rate.

Run (Windows):
    python cuonglm/curve_matcher/record_video.py

Settings are the constants below. One frame is grabbed every 1/FPS seconds and written at that rate,
so the video plays back in real time.  Stop with q / ESC / Ctrl+C (or after SECONDS).
"""
import os
import time

import cv2

from overlay import AxesOverlay

# ---- config
CAM_INDEX = 0
WIDTH, HEIGHT = 640, 480
FPS = 5.0                # recording frame rate (low)
SECONDS = 0              # stop after this many seconds (0 = until quit)
OUT = None               # output file; None = record_<time>.mp4 next to this script

HERE = os.path.dirname(os.path.abspath(__file__))
WIN = "record_video"


def open_camera(index, width, height):
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()
    raise RuntimeError(f"Cannot open camera {index}")


def main():
    out = OUT or os.path.join(HERE, f"record_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    cap = open_camera(CAM_INDEX, WIDTH, HEIGHT)
    for _ in range(8):                                   # let the exposure settle
        cap.read()
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise SystemExit("The camera returned no frame.")
    h, w = frame.shape[:2]
    axes = AxesOverlay((w, h))

    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Cannot open the video writer for {out}")
    print(f"[rec] {w}x{h} @ {FPS:g} fps -> {os.path.abspath(out)}  (q / ESC / Ctrl+C to stop)")

    period, n, t0 = 1.0 / FPS, 0, time.perf_counter()
    next_t = t0
    try:
        while True:
            now = time.perf_counter()
            if SECONDS and now - t0 >= SECONDS:
                break
            ok, f = cap.read()                           # keep draining so the frame is never stale
            if ok:
                frame = f
            if now >= next_t:
                writer.write(frame)
                n += 1
                next_t += period
                if next_t < now:                         # fell behind: do not burst to catch up
                    next_t = now + period
            view = axes.draw(frame.copy())               # axes only on the preview, not in the video
            cv2.putText(view, f"REC {n} frames  {now - t0:.0f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (27, 27, 224), 2, cv2.LINE_AA)
            cv2.imshow(WIN, view)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
    print(f"[rec] {n} frames, {n / FPS:.1f}s of video -> {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
