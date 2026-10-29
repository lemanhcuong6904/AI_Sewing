# Geometric & Silhouette Mask Helpers (`utils/helper.py`)

## 1. Overview

[`utils/helper.py`](file:///d:/ai-sewing/utils/helper.py) provides shared geometric transformations, background boundary masking, visual debugging routines, and Kalman state estimation used across the vision pipelines.

---

## 2. Silhouette Masking & Proximity Checks

To avoid confusing internal fabric texture lines with actual boundary edges, the pipeline extracts a binary silhouette of the fabric against the background plate.

### 2.1 Mask Generation (`_get_mask`)

```python
def _get_mask(img: np.ndarray, mask_thresh: int = 30) -> np.ndarray:
```
* Generates a binary mask ($0$ = background plate, $255$ = fabric) via thresholding and morphological closing to fill internal pinholes.

### 2.2 Proximity Verification (`_line_is_near_background_adjustor`)

```python
def _line_is_near_background_adjustor(
    line: np.ndarray,
    mask: np.ndarray,
    check_dist: int = 5,
    sample_pts: int = 10
) -> bool:
```
* **Algorithm**:
  1. Samples $N = 10$ equidistant points along the line segment:
     $$\mathbf{p}(t) = (1 - t)\mathbf{p}_1 + t\mathbf{p}_2, \quad t \in [0, 1]$$
  2. For each sample point, inspects a local neighborhood window of radius `check_dist = 5` pixels in the binary silhouette `mask`.
  3. Verifies that the neighborhood contains both fabric pixels ($255$) and background pixels ($0$).
  4. Returns `True` only if at least $70\%$ of the sampled points lie on a genuine fabric-to-background transition boundary. Lines residing entirely inside the fabric silhouette (folds, weave patterns) are rejected.

---

## 3. Geometric Array Normalization (`normalize_lines`)

```python
def normalize_lines(lines: Any) -> Optional[np.ndarray]:
```
* Converts arbitrary OpenCV detector outputs (e.g. OpenCV LSD returning $[N, 1, 4]$, FLD returning $[N, 1, 4]$, or list structures) into a canonical, flat 2D NumPy array:
  $$\text{shape: } (N, 4), \quad \text{dtype: } \text{float32}, \quad \text{format: } [x_1, y_1, x_2, y_2]$$
* Automatically filters out degenerate segments where length $< 1.0$ pixel.

---

## 4. 1D Kalman State Estimation (`Kalman1DTracker`)

```python
class Kalman1DTracker:
```

### State Space Model
Tracks a continuous scalar error $x$ (e.g. translation displacement) and its velocity $\dot{x}$:

$$\mathbf{x}_k = \begin{bmatrix} x_k \\ \dot{x}_k \end{bmatrix}, \quad \mathbf{F} = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix}, \quad \mathbf{H} = \begin{bmatrix} 1 & 0 \end{bmatrix}$$

### Handling Measurement Dropouts (`update`)
```python
def update(self, measurement: Optional[float]) -> Tuple[Optional[float], bool]:
```
* **Measurement Available**: Executes standard Kalman gain computation, state innovation, and covariance update:
  $$\mathbf{K} = \mathbf{P}^- \mathbf{H}^T (\mathbf{H} \mathbf{P}^- \mathbf{H}^T + R)^{-1}$$
  $$\mathbf{x} = \mathbf{x}^- + \mathbf{K}(z - \mathbf{H} \mathbf{x}^-)$$
* **Measurement Missing (`None`)**: Predicts the forward state using velocity coasting ($\mathbf{x} = \mathbf{F} \mathbf{x}^-$) for up to `max_missed = 2` consecutive frames, preventing control discontinuity during transient occlusions.
* **Returns**: `(filtered_value, is_predicted)`.

---

## 5. Visual Frame Logging (`visualize_and_save_frame_adjustor`)

Renders comprehensive debugging frames with side-by-side comparisons:
* **Left Sub-window**: Raw preprocessed grayscale image with detected line segments.
* **Right Sub-window**: Stabilized horizontal (blue) and vertical (red) selected lines, corner intersection pivot, and HUD text metrics.
* Saves frames sequentially to timestamped session folders for offline audit and training.
