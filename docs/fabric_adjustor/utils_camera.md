# Camera Ingestion & Threaded Frame Capture (`utils/camera.py`)

## 1. Overview & Latency Problem

[`utils/camera.py`](file:///d:/ai-sewing/utils/camera.py) provides low-latency, thread-safe video frame ingestion.

### The Driver Buffer Latency Problem
Standard `cv2.VideoCapture.read()` calls block and sequentially drain an internal operating system ring buffer (typically 3–5 frames deep in V4L2 drivers). If image processing takes $50$ ms, subsequent calls to `read()` return stale frames captured $150$–$250$ ms earlier. For high-speed visual servoing, stale frames cause phase lag and unstable oscillations in closed-loop motor control.

`ThreadedCamera` eliminates buffer lag by maintaining an independent daemon thread that drains the camera hardware queue as fast as frames arrive, ensuring the vision pipeline always receives the instantaneous present frame.

```mermaid
flowchart TD
    subgraph DriverHardware["Camera Hardware / OS Buffer"]
        HW[Physical USB Sensor<br/>30 FPS Stream] --> RingBuf[V4L2 Ring Buffer<br/>Depth = 3-5 frames]
    end

    subgraph DaemonThread["ThreadedCamera Background Daemon"]
        RingBuf -->|Continuous cap.read()| UpdateLoop[_update loop]
        UpdateLoop -->|Acquires Lock| Mutex[Atomic Frame Swap]
    end

    subgraph VisionPipeline["Main Vision Pipeline"]
        Consumer[main_adjustor / fabric_adjustor] -->|cam.read()| ReadMethod[read method]
        ReadMethod -->|Acquires Lock| Mutex
        Mutex -->|Returns latest frame & has_new flag| Consumer
    end
```

---

## 2. API Reference

### `class ThreadedCamera`

```python
class ThreadedCamera:
    """Continuously reads from VideoCapture in a background thread,
    always providing the most recent frame."""
```

#### Methods

* **`__init__(cap: cv.VideoCapture)`**:
  Spawns a daemon thread running `self._update()` with an internal `threading.Lock()`.
* **`read() -> Tuple[bool, Optional[np.ndarray], bool]`**:
  Returns:
  1. `ret`: Boolean indicating if camera capture is healthy.
  2. `frame`: The most recently captured frame (shape $[480, 640, 3]$, `uint8`).
  3. `has_new`: `True` if this frame has arrived since the previous call to `read()`, `False` if this is a duplicated read of an already-consumed frame.
* **`release() -> None`**:
  Sets `self._stopped = True`, joins the background capture thread with a 2.0-second timeout, and releases the underlying OpenCV `VideoCapture`.

---

## 3. Hardware Pipeline Fallback Hierarchy (`get_camera()`)

Different platforms (Jetson Linux with JetPack vs developer desktop Ubuntu/Windows) require different driver backends. `get_camera()` attempts initialization in 3 cascading stages:

### Stage 1: Hardware-Accelerated GStreamer (`nvjpegdec`)
Specifically tuned for NVIDIA Jetson Tegra architecture to bypass CPU JPEG decoding:
```python
gst = (
    "v4l2src device=/dev/video0 io-mode=2 ! "
    "image/jpeg, width=640, height=480, framerate=30/1 ! "
    "nvjpegdec ! "
    "video/x-raw ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink drop=true max-buffers=1"
)
cap = cv.VideoCapture(gst, cv.CAP_GSTREAMER)
```
* `appsink drop=true max-buffers=1`: Commands the GStreamer pipeline to drop any frame if the downstream reader is busy, completely eliminating pipeline buffering.

### Stage 2: Explicit V4L2 Backend (`cv.CAP_V4L2`)
Used when GStreamer is not compiled into OpenCV (e.g. standard pip wheels):
```python
cap = cv.VideoCapture(0, cv.CAP_V4L2)
cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
```

### Stage 3: Generic Fallback (`cv.VideoCapture(0)`)
Generic OS driver fallback for local testing on macOS or Windows development workstations.

---

## 4. Histogram Distance Metrics (`compute_hist_distance`)

Used by the pipeline to assess overall scene consistency or detect full-fabric coverage:

$$\begin{aligned}
\text{L1}: \quad & D(H_1, H_2) = \sum_i |H_1(i) - H_2(i)| \\
\text{Cosine}: \quad & D(H_1, H_2) = 1 - \frac{H_1 \cdot H_2}{\|H_1\|_2 \|H_2\|_2} \\
\chi^2 \text{ (Chi-Square)}: \quad & D(H_1, H_2) = \sum_i \frac{(H_1(i) - H_2(i))^2}{H_1(i) + H_2(i) + \epsilon} \\
\text{L2 (Euclidean)}: \quad & D(H_1, H_2) = \sqrt{\sum_i (H_1(i) - H_2(i))^2}
\end{aligned}$$
