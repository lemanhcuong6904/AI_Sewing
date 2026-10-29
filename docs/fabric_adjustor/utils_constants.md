# Physical Constants & Calibration Parameters (`utils/constants.py`)

## 1. Overview & Physical Setup

[`utils/constants.py`](file:///d:/ai-sewing/utils/constants.py) defines the global physical coordinates, camera-to-world spatial calibration constants, basis vectors, and deadband thresholds used throughout the fabric adjustor subsystem.

These values tie together the vision sensor frame ($640 \times 480$ pixels) and the mechanical XY stepper stage.

---

## 2. Parameter Reference Table

| Constant | Value | Units | Description |
| :--- | :--- | :--- | :--- |
| `DEFAULT_X_BASIS` | `245` | Pixels | Reference X coordinate representing the target corner position of the fabric in the camera image plane. |
| `DEFAULT_Y_BASIS` | `192` | Pixels | Reference Y coordinate representing the target corner position of the fabric in the camera image plane. |
| `DEFAULT_PIVOT_X` | `385.0` | Pixels | Horizontal center of rotation of the robot gripper clamp. |
| `DEFAULT_PIVOT_Y` | `-20.0` | Pixels | Vertical center of rotation of the robot gripper clamp in camera space. |
| `DEFAULT_PIVOT` | `(385.0, -20.0)` | Pixels | Nominal home center of rotation of the robot hand in camera space. |
| `U1_VECTOR` | `np.array([1, 0])` | Dimensionless | Horizontal reference unit vector along the Cartesian X-axis. |
| `U2_VECTOR` | `np.array([0, -1])` | Dimensionless | Vertical reference unit vector pointing upward in the physical plane (opposite to image row index). |
| `PX_PER_CM_X` | `39.0` | Pixels / cm | Horizontal optical magnification scale factor ($1\text{ cm} = 39\text{ px} \implies 1\text{ px} \approx 0.256\text{ mm}$). |
| `PX_PER_CM_Y` | `35.0` | Pixels / cm | Vertical optical magnification scale factor ($1\text{ cm} = 35\text{ px} \implies 1\text{ px} \approx 0.286\text{ mm}$). |
| `AXIS_LENGTH_PX` | `468.0` | Pixels | Graphical axis rendering length for HUD overlays ($12\text{ cm} \times 39.0\text{ px/cm}$). |
| `TRANSLATION_DEADBAND_CM`| `0.03` | Centimeters | Displacement deadband tolerance ($0.3$ mm). Displacements below this magnitude are ignored to prevent motor jitter. |
| `ROTATION_DEADBAND_DEG` | `0.1` | Degrees | Angular rotation deadband tolerance ($0.1^\circ$). Angular errors below this threshold require no motor actuation. |

---

## 3. Geometric Derivations & Unit Conversions

### Coordinate Space Transformations

```
       Image Plane (OpenCV)                   Physical Sewing Table Frame
       
 (0, 0) ──────────────────> +X (Col)           ^ +Y (U2_VECTOR: [0, -1])
   │                                           │
   │           Corner Origin                   │           Corner Origin
   │                 * (245, 192)              │                 * (0, 0)
   │                                           │
   V +Y (Row)                                  └──────────────────> +X (U1_VECTOR: [1, 0])
```

1. **Pixel-to-Centimeter Conversion (Vector Form)**:
   Given a displacement vector $\mathbf{t}_{\text{px}} = [t_x, t_y]^T$ projected along basis vectors $\mathbf{u}_1$ and $\mathbf{u}_2$:
   $$t_{x,\text{cm}} = \frac{t_{x,\text{px}}}{\text{PX\_PER\_CM\_X}} = \frac{t_{x,\text{px}}}{39.0}$$
   $$t_{y,\text{cm}} = \frac{t_{y,\text{px}}}{\text{PX\_PER\_CM\_Y}} = \frac{t_{y,\text{px}}}{35.0}$$

2. **Centimeter-to-Millimeter (Motor Command Vector)**:
   For UART communication with the STM32 controller:
   $$M_1 = t_{x,\text{cm}} \times 10.0 \quad (\text{mm})$$
   $$M_2 = t_{y,\text{cm}} \times 10.0 \quad (\text{mm})$$

3. **General Actuated Motion Vector in Image Plane**:
   Executing motor commands $(M_1, M_2)$ drives the stage along the physical basis:
   $$\mathbf{v}_{\text{actuated}} = \left(\frac{M_1}{10.0} \cdot \text{PX\_PER\_CM\_X}\right) \mathbf{u}_1 + \left(\frac{M_2}{10.0} \cdot \text{PX\_PER\_CM\_Y}\right) \mathbf{u}_2$$

4. **Anisotropic Optical Compensation**:
   Because $\text{PX\_PER\_CM\_X} \neq \text{PX\_PER\_CM\_Y}$ ($39.0$ vs $35.0$), camera perspective tilt and non-square sensor aspect ratio are strictly accounted for independently along both axes, avoiding distortion errors during multi-axis diagonal motions.
