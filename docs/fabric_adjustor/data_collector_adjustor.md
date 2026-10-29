# Dataset Collection & Benchmarking (`data_collector_adjustor.py`)

## 1. Overview & Dataset Collection Role

[`data_collector_adjustor.py`](file:///d:/ai-sewing/data_collector_adjustor.py) is a data gathering and ground-truth annotation utility. It allows engineers to systematically capture real-world camera frames paired with metric ground-truth pose annotations $(t_x, t_y, \theta_h, \theta_v)$ under realistic factory conditions (varying fabrics, wrinkles, lighting fluctuations, and mechanical backgrounds).

The resulting datasets are used to:
1. Benchmark line detection backends (LSD, FLD, EDLines, MLSD, DexiNed).
2. Measure metric accuracy and repeatability against physical millimeter rulers.
3. Train and fine-tune edge detection and line estimation models.

---

## 2. Dataset Storage Hierarchy

Every recording session creates an isolated, timestamped folder under `adjustor_data/`:

```
d:\ai-sewing\adjustor_data\
└── 20260919_153000/
    ├── data/
    │   ├── 0.png
    │   ├── 1.png
    │   └── ...
    └── label/
        ├── 0.txt
        ├── 1.txt
        └── ...
```

### Label File Format
Each `.txt` label file contains exactly 4 space-separated floating-point values:
```
<tx_cm> <ty_cm> <angle_h_deg> <angle_v_deg>
```
* `tx_cm`: Ground truth horizontal displacement from the table reference corner in centimeters.
* `ty_cm`: Ground truth vertical displacement from the table reference corner in centimeters.
* `angle_h_deg`: Signed rotation angle of the horizontal fabric edge relative to `U1_VECTOR`.
* `angle_v_deg`: Signed rotation angle of the vertical fabric edge relative to `U2_VECTOR`.

---

## 3. Operational Workflows

### 3.1 Mode 1: Manual Collection (Default)

```bash
python data_collector_adjustor.py
```
1. Operator manually positions a fabric swatch at an arbitrary displacement or angle on the sewing table.
2. The live HUD displays the reference origin `(DEFAULT_X_BASIS, DEFAULT_Y_BASIS)` and metric centimeter grid.
3. Operator presses **`Space`**:
   * The frame is saved to `data/<index>.png`.
   * The terminal pauses with a prompt:
     ```
     Enter ground truth (trans_vector_x trans_vector_y rotating_angle_horizontal rotating_angle_vertical):
     ```
4. Operator inputs the physically measured values (e.g. `1.2 -0.5 2.1 2.3`), which is saved to `label/<index>.txt`.
5. Press **`q`** to safely finish the recording session.

---

### 3.2 Mode 2: Semi-Automatic Guided Collection (`--label-file`)

```bash
python data_collector_adjustor.py --label-file target_poses.txt
```
Used for systematic grid sweep testing across a pre-defined test plan:
1. Loads pre-generated target poses from a test script file.
2. For each target pose, [`draw_target_axes`](utils_viz_adjustor.md) renders colored target reference guidelines (yellow and magenta) directly onto the live camera stream.
3. Operator physically nudges the fabric piece until its edges visually align with the target guidelines.
4. Operator presses **`Space`**: the frame is saved with the active label automatically, and the UI advances to the next test pose in the file.
* **`s`**: Skips the current target pose if difficult to position.
* **`q`**: Quits early while preserving all captured data.
