# Main Adjustor Orchestrator (`main_adjustor.py`)

## 1. Overview & System Role

[`main_adjustor.py`](file:///d:/ai-sewing/main_adjustor.py) is the top-level execution script and runtime coordinator for the fabric alignment subsystem. It binds camera acquisition, vision-based geometric line/edge estimation, temporal stabilization, closed-loop moderation logic, and UART serial communication with the physical STM32 motor controller.

Its core objective is to execute **closed-loop visual servoing**:
1. Continuously measure horizontal displacement ($t_x$), vertical displacement ($t_y$), and angular rotation error ($\theta$).
2. Translate vision-space pixel offsets to the actual physical commands in millimeters and degrees.
3. Enforce **sequential moderation steps** (translation deadbanding and step budgeting) to prevent motor over-actuation.
4. Issue real-time actuation strings (`"M1, M2, M3\n"`) to the physical sewing stage, and issue release commands (`"k5\n"`) upon completion.

---

## 2. Configuration & Parameter Reference

The modular [`FabricAdjustorApp`](file:///d:/ai-sewing/main_adjustor.py) orchestrator class and CLI expose 22 configuration parameters:

| Parameter | Type (`dtype`) | Default | CLI Flag | Description |
| :--- | :--- | :--- | :--- | :--- |
| `img_size` | `Optional[Tuple[int, int]]` | `None` | `--img-size` | Image resolution $(W, H)$ fed to the detector. |
| `x_basis` | `Optional[float]` | `DEFAULT_X_BASIS` (`245.0`) | `--x-basis` | Reference X coordinate (pixels) representing target origin on the table. |
| `y_basis` | `Optional[float]` | `DEFAULT_Y_BASIS` (`192.0`) | `--y-basis` | Reference Y coordinate (pixels) representing target origin on the table. |
| `visualize` | `bool` | `False` | `--visualize` | Enables annotated live display overlays and saves debug frames. |
| `detector_mode` | `DetectorMode` | `DetectorMode.DEXINED` | `--detector` | Detection backend: `dexined`, `mlsd`, `lsd`, `fld`, `edlines`, `ensemble`. |
| `prior_mask_path` | `Optional[str]` | `None` | `--prior-mask` | Path to prior reference image/mask (required for `dexined` (for curved fabric) and `NonOrthoLineAdjustor` (for non-orthogonal fabric)). |
| `match_method` | `str` | `'contour'` | `--match-method` | Mask matching metric for DexiNed only: `'contour'`, `'moments'`, `'ecc'`, `'bbox'`, `'feature'`. |
| `morphology` | `str` | `'fill'` | `--morphology` | Morphological cleanup for DexiNed only: `'fill'`, `'open'`, `'erode'`, `'none'`. |
| `enhance_contrast` | `bool` | `False` | `--enhance-contrast` | Applies CLAHE adaptive histogram equalization during preprocessing to enhance the contrast of input image. |
| `smooth_texture` | `bool` | `True` | `--no-smooth` (negated) | Applies bilateral/Gaussian filtering to suppress fabric weave texture. |
| `batch_size` | `int` | `20` | `--batch-size` | Number of frames collected per inference batch. |
| `max_moderations` | `int` | `2` | `--max-moderations` | Maximum adjustment iterations before triggering final release. |
| `max_angle_moderations` | `int` | `2` | `--max-angle-moderations`| Number of moderation steps where rotation angle adjustment is active. |
| `crop_roi` | `Optional[Tuple[int, int, int, int]]` | `None` | `--crop-roi` | Pre-crop bounds $(s_x, e_x, s_y, e_y)$ in sensor pixels. |
| `continuous` | `bool` | `False` | `--continuous` | Toggles continuous generator loop instead of single-shot event loop. |
| `cooldown_time` | `float` | `0.0` | `--cooldown-time` | Sleep cooldown (seconds) between successive adjustments. |
| `port` | `Optional[str]` | `None` | `--port` | Serial device path (e.g. `'/dev/ttyUSB0'`). Auto-detected if `None`. |
| `baud` | `int` | `115200` | `--baud` | UART baud rate. |
| `x_offset_cm` | `float` | `0.0` | `--x-offset-cm` | Base horizontal physical offset in cm for Phase 2 (`L2b`). |
| `y_offset_cm` | `float` | `0.4` | `--y-offset-cm` | Base vertical physical offset in cm for Phase 2 (`L2b`). |
| `max_moderations_l2a` | `int` | `1` | `--max-moderations-l2a` | Number of iterations allocated to Phase 2a (`L2a`). The remaining (`max_moderations - max_moderations_l2a`) are used for `L2b`. |
| `x_offset_cm_l2a` | `float` | `-0.5` | `--x-offset-cm-l2a` | Additional horizontal offset in cm for `L2a` modified on top of `L2b`. |
| `y_offset_cm_l2a` | `float` | `0.0` | `--y-offset-cm-l2a` | Additional vertical offset in cm for `L2a` modified on top of `L2b`. |
| `rot_alignment` | `str` | `'horizontal'` | `--rot-alignment` | Which axis rotation to prioritize: `'horizontal'` ($R_h$) or `'vertical'` ($R_v$). |
| `target_frames` | `int` | `150` | `--target-frames` | Frame count collected for statistical consensus before calculating control signals. |
| `pivot` | `Optional[Tuple[float, float]]` | `DEFAULT_PIVOT` | `--pivot` | Initial gripper rotation center $(p_x, p_y)$ in sensor pixels. |
| `compensate_rotation` | `bool` | `True` | `--no-compensate-rotation` (negated) | Disables translation correction for rotational swing around the pivot. |

---

## 3. Control Signal Processing & Moderation State Machine

The core algorithmic dispatch is encapsulated inside [`FabricAdjustorApp.process_control_signals(...)`](file:///d:/ai-sewing/main_adjustor.py):

```mermaid
stateDiagram-v2
    [*] --> CheckInput
    CheckInput --> DeadbandsAndClamping: control_signals is not None
    CheckInput --> [*]: control_signals is None

    state DeadbandsAndClamping {
        ConvertPixelsToMetric: tx_cm = tx_px / 39.0, ty_cm = ty_px / 35.0
        SafetyClamping: Clamp tx_cm, ty_cm to [-2.0, 3.0] cm
        ScaleToMillimeters: m1_mm = tx_cm * 10, m2_mm = ty_cm * 10
        ClampRotation: Clamp rot to [-10, 10] deg
    }

    DeadbandsAndClamping --> CheckBudget
    
    state CheckBudget {
        state "moderation_count < max_moderations" as InBudget
        state "moderation_count >= max_moderations" as LimitReached
    }

    InBudget --> ApplyTranslationDeadband: abs(tx_cm) < 0.03 -> m1=0; abs(ty_cm) < 0.03 -> m2=0
    ApplyTranslationDeadband --> ApplyRotationDeadband: moderation_count < max_angle_moderations
    ApplyRotationDeadband --> TransmitUART: send_and_receive("M1, M2, M3\n")
    TransmitUART --> IncrementCounter: moderation_count += 1
    IncrementCounter --> [*]

    LimitReached --> TriggerRelease: send_and_receive("k\n")
    TriggerRelease --> ResetCounters: Reset counters & set state to l1_done/l2_done
    ResetCounters --> [*]
```

### Mathematical Formulations

1. **Pixel-to-Physical Scaling**:
   $$t_{x,\text{cm}} = \frac{t_{x,\text{px}}}{\text{PX\_PER\_CM\_X}} = \frac{t_{x,\text{px}}}{39.0}$$
   $$t_{y,\text{cm}} = \frac{t_{y,\text{px}}}{\text{PX\_PER\_CM\_Y}} = \frac{t_{y,\text{px}}}{35.0}$$

2. **Safety Bounding Box**:
   Prevents mechanical hand over-travel by binding the offset ranges:
   $$t_{x,\text{cm}} \leftarrow \operatorname{clip}(t_{x,\text{cm}}, -2.0, +3.0)$$
   $$t_{y,\text{cm}} \leftarrow \operatorname{clip}(t_{y,\text{cm}}, -2.0, +3.0)$$
   $$m_1 = 10 \cdot t_{x,\text{cm}} \quad (\text{mm}), \quad m_2 = 10 \cdot t_{y,\text{cm}} \quad (\text{mm})$$
   $$\theta \leftarrow \operatorname{clip}(\theta, -10.0^\circ, +10.0^\circ)$$

3. **Deadband Filter**:
   Sub-millimeter jitter is zeroed out to prevent overadjusting:
   $$m_1 = 0 \quad \text{if } |t_{x,\text{cm}}| < 0.03\text{ cm}$$
   $$m_2 = 0 \quad \text{if } |t_{y,\text{cm}}| < 0.03\text{ cm}$$
   $$\theta = 0 \quad \text{if } |\theta| < 0.1^\circ$$

4. **Dynamic Pivot Odometry Tracking (General Vector Form)**:
   Let $\mathbf{I} = [I_x, I_y]^T$ be the fabric corner position and $\mathbf{B} = [x_{\text{basis}}, y_{\text{basis}}]^T$ be the reference origin in camera space. The error displacement vector is:
   $$\mathbf{v} = \mathbf{I} - \mathbf{B}$$
   Projected onto unit basis vectors $\mathbf{u}_1$ and $\mathbf{u}_2$:
   $$t_{x,\text{px}} = \mathbf{v} \cdot \mathbf{u}_1, \quad t_{y,\text{px}} = \mathbf{v} \cdot \mathbf{u}_2$$
   The motor actuation $\mathbf{v}_{\text{actuated}} = t_{x,\text{px}} \mathbf{u}_1 + t_{y,\text{px}} \mathbf{u}_2$ drives the stage to cancel the error vector. Because the robotic clamp (center of rotation $\mathbf{P} = [p_x, p_y]^T$) physically translates rigidly with the stage, its displacement in camera space is governed by the general vector form:
   $$\Delta \mathbf{P} = -\mathbf{v}_{\text{actuated}} = -(t_{x,\text{px}} \mathbf{u}_1 + t_{y,\text{px}} \mathbf{u}_2)$$
   $$\mathbf{P}_{\text{new}} = \mathbf{P}_{\text{old}} + \Delta \mathbf{P}$$
   In terms of executed motor displacements $m_1, m_2$ (in mm):
   $$\Delta \mathbf{P} = -\left( \frac{m_1}{10.0} \cdot \text{PX\_PER\_CM\_X} \cdot \mathbf{u}_1 + \frac{m_2}{10.0} \cdot \text{PX\_PER\_CM\_Y} \cdot \mathbf{u}_2 \right)$$
   This formulation is coordinate-invariant: if basis vectors $\mathbf{u}_1, \mathbf{u}_2$ are altered in the future, the physical error-cancellation law $\Delta \mathbf{P} = -\mathbf{v}_{\text{actuated}}$ remains strictly preserved.
   - **Phase L1**: Pivot starts at $\mathbf{P}_{\text{home}} = (385.0, -20.0)\,\text{px}$ and translates on each dispatched $(m_1, m_2, m_3)$ step.
   - **Phase L1 End (`k`)**: Clamp releases and arm returns home; pivot resets to $\mathbf{P}_{\text{home}}$.
   - **Phase L2 Start**: Pivot resets to $\mathbf{P}_{\text{home}}$.
   - **Sub-phase L2a $\to$ L2b**: The clamp continues holding the fabric (no release/homing); the pivot **retains** its accumulated position across the transition.
   - **Phase L2 End (`k`)**: Clamp releases and arm returns home; pivot resets to $\mathbf{P}_{\text{home}}$.

5. **Emergency / Completion Release (`k`)**:
   When `moderation_count >= max_moderations`, the pipeline terminates the current alignment cycle and sends the UART command `'k'`, commanding the STM32 to unclamp the fabric clamp and transition to `l1_done` or `l2_done`. While in these completed states, subsequent `"OK"` unclamp acknowledgments from the STM32 do not re-trigger computation, preventing infinite adjustment loops. Furthermore, UART phase change guards ensure incoming STM32 status echoes (e.g. `"L1 OK"`, `"L2 OK"`) do not reset iteration counters or abort active sub-phases.

---

## 4. User Interaction & Runtime Keybindings

In terminal cbreak mode (`tty.setcbreak`), single-character and buffered command inputs are read non-blocking via `select.select`:

* **`l1`**: Manually locks/resets table reference origin to baseline `(x_basis, y_basis)`.
* **`l2`**: Enters Phase 2 alignment starting at `L2a` with coordinates defined in vector form:
  - `L2a`: $\mathbf{B}_{L2a} = \mathbf{B}_{L2b} + \Delta\mathbf{B}_{L2a}$ where $\Delta\mathbf{B}_{L2a} = [x_{\text{offset\_cm\_l2a}} \cdot \text{PX\_PER\_CM\_X}, y_{\text{offset\_cm\_l2a}} \cdot \text{PX\_PER\_CM\_Y}]^T$ for `max_moderations_l2a` iterations.
  - Automatically transitions to `L2b`: $\mathbf{B}_{L2b} = \mathbf{B}_{\text{base}} + \Delta\mathbf{B}_{L2b}$ where $\Delta\mathbf{B}_{L2b} = [x_{\text{offset\_cm}} \cdot \text{PX\_PER\_CM\_X}, y_{\text{offset\_cm}} \cdot \text{PX\_PER\_CM\_Y}]^T$ for the remaining `(max_moderations - max_moderations_l2a)` iterations.
* **`l2a`** / **`2a`**: Directly triggers Phase 2a (`L2a`).
* **`l2b`** / **`2b`**: Directly triggers Phase 2b (`L2b`).
* **`k`**: Emergency clamp release and return to baseline.
* **`q`** / **`x`**: Terminates program, closes UART serial port, and restores original terminal attributes (`termios.tcsetattr`).
