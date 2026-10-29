# Core Fabric Adjustor Engine (`modules/fabric_adjustor.py`)

## 1. Overview

[`modules/fabric_adjustor.py`](file:///d:/ai-sewing/modules/fabric_adjustor.py) contains the core algorithmic classes responsible for converting raw image streams into physical alignment control signals:
* **`LineDetectionAdjustor`**: Visual servoing engine for orthogonal (nominal $90^\circ$) fabric panels.
* **`NonOrthoLineAdjustor`**: Extension supporting arbitrary, non-orthogonal fabric corner angles ($\alpha \neq 90^\circ$) using pre-captured prior geometry.

---

## 2. Mathematical Foundations

### 2.1 Multi-Criteria Line Pair Optimization (`_select_best_lines_gradient_priority`)

Rather than relying purely on segment length or position, candidate lines are evaluated pairwise ($L_i, L_j$) using an objective function that prioritizes true physical fabric boundaries over weave textures or internal wrinkles:

1. **Angular Orthogonality Constraint**:
   Two lines forming a corner must not be nearly parallel. Pairs with an angular difference $< 30^\circ$ or $> 150^\circ$ are immediately pruned.
2. **Composite Score Formulation**:
   For each valid pair assigned to horizontal ($L_h$) and vertical ($L_v$) axes:
   $$\text{Score} = w_{\text{grad}} \cdot (S_{\text{grad}, h} + S_{\text{grad}, v}) + w_{\text{len}} \cdot \frac{\text{len}_h + \text{len}_v}{2} - w_{\text{dist}} \cdot d(\mathbf{p}_{\text{mid}}, \mathbf{p}_{\text{origin}}) - w_{\text{angle}} \cdot |\Delta\theta - \alpha_{\text{prior}}|$$
   * $S_{\text{grad}}$: Normal Sobel gradient magnitude sampled along the line segment by [`EdgeGradientScorer`](utils_tracking_helper.md).
   * $w_{\text{grad}} = 23.0$: Heavy scaling prioritizing high-contrast physical edges over internal texture lines.
   * $\alpha_{\text{prior}}$: Expected corner angle ($90.0^\circ$ by default, or loaded from `prior_lines.json`).

### 2.2 Corner Intersection via Cramer's Rule (`_calculate_control_signals`)

Given the horizontal line segment $L_1 = [(x_{1a}, y_{1a}), (x_{2a}, y_{2a})]$ and vertical line segment $L_2 = [(x_{1b}, y_{1b}), (x_{2b}, y_{2b})]$, the 2D corner intersection $\mathbf{I} = (I_x, I_y)$ is computed analytically using Cramer's determinant rule:

$$\text{denom} = (x_{1a} - x_{2a})(y_{1b} - y_{2b}) - (y_{1a} - y_{2a})(x_{1b} - x_{2b})$$

When $|\text{denom}| \ge 10^{-5}$:
$$I_x = \frac{(x_{1a}y_{2a} - y_{1a}x_{2a})(x_{1b} - x_{2b}) - (x_{1a} - x_{2a})(x_{1b}y_{2b} - y_{1b}x_{2b})}{\text{denom}}$$
$$I_y = \frac{(x_{1a}y_{2a} - y_{1a}x_{2a})(y_{1b} - y_{2b}) - (y_{1a} - y_{2a})(x_{1b}y_{2b} - y_{1b}x_{2b})}{\text{denom}}$$

### 2.3 Basis Vector Projection & Rotation Compensation (General Vector Form)

1. **Rotation Compensation Around Pivot**:
   When rotation compensation is enabled with tracked robot gripper pivot $\mathbf{P} = [p_x, p_y]^T$ and effective rotation angle $\theta$, the corner position transforms according to planar rigid rotation:
   $$\mathbf{I}_{\text{rot}} = \mathbf{P} + \mathbf{R}(\theta) (\mathbf{I} - \mathbf{P})$$
   where $\mathbf{R}(\theta)$ is the standard 2D rotation matrix:
   $$\mathbf{R}(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

2. **Translation Vector Projection**:
   The error displacement vector from target origin $\mathbf{B} = [x_{\text{basis}}, y_{\text{basis}}]^T$ to (rotation-compensated) corner $\mathbf{I}_{\text{rot}}$ is:
   $$\mathbf{v} = \mathbf{I}_{\text{rot}} - \mathbf{B}$$
   Projecting $\mathbf{v}$ onto the arbitrary reference unit basis vectors $\mathbf{u}_1$ and $\mathbf{u}_2$:
   $$t_x = \mathbf{v} \cdot \mathbf{u}_1, \quad t_y = \mathbf{v} \cdot \mathbf{u}_2$$
   $$\mathbf{t} = \begin{bmatrix} t_x \\ t_y \end{bmatrix} = \begin{bmatrix} \mathbf{u}_1^T \\ \mathbf{u}_2^T \end{bmatrix} \mathbf{v}$$
   This vector formulation is strictly coordinate-invariant: it holds true regardless of the physical orientation or definition of basis vectors $\mathbf{u}_1$ and $\mathbf{u}_2$.

3. **Signed Rotation Angle**:
   The rotation angle $\theta$ of an edge vector $\mathbf{v}_{\text{edge}}$ relative to reference basis vector $\mathbf{u}$ in the plane spanned by $(\mathbf{u}_1, \mathbf{u}_2)$ is derived from the dot and cross products:
   $$\cos\theta = \frac{\mathbf{v}_{\text{edge}} \cdot \mathbf{u}}{\|\mathbf{v}_{\text{edge}}\| \|\mathbf{u}\|}, \quad \sin\theta = \frac{(\mathbf{u} \times \mathbf{v}_{\text{edge}}) \cdot \hat{\mathbf{z}}}{\|\mathbf{v}_{\text{edge}}\| \|\mathbf{u}\|}$$
   $$\theta = \operatorname{arctan2}(\sin\theta, \cos\theta)$$
   where $\hat{\mathbf{z}} = \frac{\mathbf{u}_1 \times \mathbf{u}_2}{\|\mathbf{u}_1 \times \mathbf{u}_2\|}$ defines the physical normal of the planar basis. The angle is mapped modulo $180^\circ$ and wrapped to $[-90^\circ, +90^\circ]$ to ensure signed consistency for motor actuation.

---

## 3. Core API Reference

### `_perform_line_detection(...)`

```python
def _perform_line_detection(
    frame: np.ndarray,
    img_size: Optional[Tuple[int, int]],
    enhance_contrast: bool,
    smooth_texture: bool,
    detector_mode: DetectorMode,
    crop_roi: Optional[Tuple[int, int, int, int]] = None,
    square_contrast: bool = False,
    noisy_areas: Optional[List] = None,
    length_weighting: bool = False,
    return_scores: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[List[float]], Optional[List[float]], Any]:
```
* **Inputs**:
  * `frame`: Raw input frame (`uint8`, shape $[H, W, 3]$).
  * `img_size`: Processing resolution $(W, H)$.
  * `enhance_contrast`: CLAHE contrast enhancement toggle.
  * `smooth_texture`: Bilateral filtering toggle.
  * `detector_mode`: Line detection algorithm (`DetectorMode.MLSD`, etc.).
  * `crop_roi`: Sensor bounding box coordinates $(s_x, e_x, s_y, e_y)$.
* **Returns**:
  * `(gray, img, lines, best_h_line, best_v_line, per_det)`

### `compute_single_signal(cam, target_frames, visualize)`

Collects a burst of camera frames, purges hardware queue lag via `cam.clear_buffer()`, executes batched line inference, updates the temporal [`LineStabilizer`](utils_tracking_helper.md), and computes stable consensus control signals:
* **Returns**: `Tuple[Tuple[float, float], Tuple[float, float]]` containing $((t_x, t_y), (R_h, R_v))$.

### `run(cam, target_frames, visualize, debug)`

A Python generator yielding real-time control signals continuously:
* Implements a **cooldown timer** between successive adjustments to allow the physical robot stage to settle.
* Filters out movements smaller than `TRANSLATION_DEADBAND_CM` ($0.03$ cm) and `ROTATION_DEADBAND_DEG` ($0.1^\circ$).
* When `debug=True`, provides interactive frame-by-frame inspection without dispatching signals.

---

## 4. Non-Orthogonal Adjustment (`NonOrthoLineAdjustor`)

In standard garment production, pieces with acute or obtuse corners cannot be aligned with standard $90^\circ$ assumptions. `NonOrthoLineAdjustor` resolves this by:

1. Loading `prior_img_path` or JSON reference metadata created by [`capture_prior.py`](capture_prior.md).
2. Extracting the golden reference angle $\alpha_{\text{prior}}$ and reference direction vectors.
3. Calculating rotation errors specifically relative to the non-orthogonal target template rather than a rigid Cartesian right angle.
