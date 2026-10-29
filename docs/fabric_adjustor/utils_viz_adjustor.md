# Visualization & Graphical Overlays (`utils/viz_adjustor.py`)

## 1. Overview & Visual Feedback Role

[`utils/viz_adjustor.py`](file:///d:/ai-sewing/utils/viz_adjustor.py) implements the OpenCV HUD (Heads-Up Display) and diagnostic rendering engine for the fabric adjustor.

During live visual servoing, human machine operators and QA engineers require real-time visual verification of:
* The fixed calibration reference corner `(DEFAULT_X_BASIS, DEFAULT_Y_BASIS)`.
* Metric centimeter grid lines anchored at the origin.
* Detected raw candidate lines vs selected horizontal ($L_h$, blue) and vertical ($L_v$, red) fabric edges.
* The calculated 2D corner intersection point $\mathbf{I}$ and target displacement vectors.
* Real-time metrics HUD: $(t_x, t_y)$ displacement in pixels and cm, rotation angles $\theta_h, \theta_v$, and stabilization status.

---

## 2. Component Rendering Functions

### 2.1 Calibrated Metric Grid (`draw_grid`)

```python
def draw_grid(img: np.ndarray, basis: Tuple[float, float], alpha: float = 0.25) -> None:
```
* **Algorithm**:
  * Draws semi-transparent horizontal and vertical grid lines anchored exactly at `basis` and spaced by `PX_PER_CM_X` ($39$ px) and `PX_PER_CM_Y` ($35$ px).
  * Uses alpha blending via `cv.addWeighted` ($\alpha = 0.25$) so the fabric and sewing plate remain clearly visible behind the coordinate lines.

### 2.2 Coordinate Axes & Direction Arrows (`draw_axes`)

```python
def draw_axes(img: np.ndarray, basis: Tuple[float, float], axis_length: int) -> None:
```
* Renders directional arrows along the physical basis vectors:
  * **X-Axis Arrow**: Points along `U1_VECTOR` ($[1, 0]$), rendered in Red (`AXIS_COLOUR_X`).
  * **Y-Axis Arrow**: Points along `U2_VECTOR` ($[0, -1]$), pointing upward in the physical plane, rendered in Green (`AXIS_COLOUR_Y`).

### 2.3 Centimeter & Millimeter Ticks (`draw_axis_ticks`)

```python
def draw_axis_ticks(img: np.ndarray, basis: Tuple[float, float], n_ticks: Optional[int] = None) -> None:
```
* Generates perpendicular hash marks along both coordinate arrows spaced at $1\text{ mm}$ and $1\text{ cm}$ physical intervals, giving operators an instantaneous visual measurement of displacement without looking at numbers.
* **General Vector Formulation**: Perpendicular tick directions are derived using the 2D orthogonal unit normal vector:
  $$\mathbf{u}_{\perp} = \begin{bmatrix} -u_y \\ u_x \end{bmatrix}, \quad \text{satisfying } \mathbf{u} \cdot \mathbf{u}_{\perp} = 0$$
  Tick offsets are computed as $\pm \mathbf{u}_{\perp} \cdot \ell_{\text{tick}}$, ensuring exact orthogonality for any arbitrary basis vector orientation.

---

## 3. High-Level Composite Overlays

### `draw_adjustor_overlay(...)`

```python
def draw_adjustor_overlay(
    img: np.ndarray,
    basis: Tuple[float, float],
    axis_length: int = AXIS_LENGTH_PX,
    best_h_line: Optional[Union[List[float], np.ndarray]] = None,
    best_v_line: Optional[Union[List[float], np.ndarray]] = None,
    control_signals: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> None:
```
* Combines the grid, axis vectors, centimeter ticks, pivot reference marks, detected fabric boundary lines, and control displacement HUD readout into a unified rendering call that modifies the image array in-place.

### `draw_target_axes(...)`

```python
def draw_target_axes(
    img: np.ndarray,
    basis: Tuple[float, float],
    tx_cm: float,
    ty_cm: float,
    angle_h_deg: float,
    angle_v_deg: float,
    line_len: int = 200,
) -> None:
```
* Renders the **target projected pose** that the fabric piece must align to:
  1. **Corner Projection**: The projected corner position $\mathbf{I}_{\text{target}}$ is computed in general vector form:
     $$\mathbf{I}_{\text{target}} = \mathbf{B} + (tx_{\text{cm}} \cdot \text{PX\_PER\_CM\_X}) \mathbf{u}_1 + (ty_{\text{cm}} \cdot \text{PX\_PER\_CM\_Y}) \mathbf{u}_2$$
  2. **Direction Vectors**: Rotated target line orientations are computed directly from unit vectors $\mathbf{u}_1, \mathbf{u}_2$ and the basis orientation determinant:
     $$\text{rad}_h = \operatorname{atan2}(u_{1y}, u_{1x}) + \operatorname{sgn}(\det[\mathbf{u}_1, \mathbf{u}_2]) \cdot \theta_h$$
     $$\text{rad}_v = \operatorname{atan2}(u_{2y}, u_{2x}) + \operatorname{sgn}(\det[\mathbf{u}_1, \mathbf{u}_2]) \cdot \theta_v$$
