import time
import cv2 as cv
from utils.line_detectors.detector import DetectorMode, run_line_detector, detect_lines, preprocessing_sewer
import numpy as np
import os

def main():
    # create a dummy image or read from captured_frames if available
    img_path = None
    frames_dir = "captured_frames"
    if os.path.exists(frames_dir) and os.path.isdir(frames_dir):
        files = [f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))]
        if files:
            img_path = os.path.join(frames_dir, files[0])
            
    if img_path:
        print(f"Testing with image: {img_path}")
        img = cv.imread(img_path)
    else:
        print("No image found, generating a random one.")
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    print("\n--- Testing M-LSD TensorRT Pipeline ---")
    
    # Warmup
    print("Warming up TRT Engine (This may take a minute on the first run)...")
    gray, processed_img = preprocessing_sewer(img)
    t0 = time.time()
    lines = detect_lines(gray, DetectorMode.MLSD, processed_img)
    print(f"Warmup time: {time.time() - t0:.3f}s")
    
    if lines is not None:
        print(f"Detected {len(lines)} lines during warmup.")
    else:
        print("No lines detected.")

    # Benchmark
    print("\nBenchmarking 100 frames...")
    runs = 100
    t_start = time.time()
    for _ in range(runs):
        _ = detect_lines(gray, DetectorMode.MLSD, processed_img)
    t_end = time.time()
    
    fps = runs / (t_end - t_start)
    print(f"Average FPS: {fps:.2f} ({1000/fps:.2f} ms/frame)")
    print("---------------------------------------")

if __name__ == '__main__':
    main()
