# Fabric Adjustor Pipeline: System Architecture & Technical Documentation

This directory contains technical documentation for the **Fabric Adjustor Pipeline** in the `ai-sewing` robotic system. The pipeline is an autonomous visual servoing system designed to detect, track, and physically align fabric pieces to target coordinates and orientations prior to sewing operations.

---

## 1. High-Level System Architecture

The pipeline closes the loop between high-speed camera acquisition, computer vision line/edge estimation, temporal stabilization, coordinate frame transformation, and multi-axis motor actuation via UART.

```mermaid
flowchart TD
    subgraph VisionAcquisition["1. Vision Acquisition"]
        A[ThreadedCamera<br/>GStreamer / V4L2] -->|640x480 Raw Frame| B[preprocessing_adjustor<br/>Crop ROI, CLAHE, Morphology]
    end

    subgraph DetectionAndScoring["2. Detection & Saliency Scoring"]
        B -->|Grayscale ROI| C[detect_lines / MLSD TRT]
        C -->|Line Candidates| D[filter_noisy_lines_by_area<br/>Exclude Fixtures & Plate]
        D --> E[EdgeGradientScorer<br/>Sobel Gradient Magnitude]
        E --> F[filter_dominant_angles<br/>Cyclic Clustering]
    end

    subgraph TemporalStabilization["3. Temporal Stabilization"]
        F --> G[LineStabilizer<br/>Hough Polar Clustering & Consensus]
    end

    subgraph ErrorDerivation["4. Coordinate Transformation"]
        G --> H[Cramer's Rule Intersection<br/>Fabric Corner Calculation]
        H --> I[Project onto U1, U2 Basis<br/>Rotation relative to U2]
        I --> J[Deadband Filtering<br/>0.03 cm, 0.1 deg]
    end

    subgraph Actuation["5. Control & Motor Actuation"]
        J --> K[Sequential Moderation State Machine<br/>Translation -> Rotation]
        K -->|Clamped [-2.0, 3.0] cm| L[jetson_uart: send_and_receive<br/>M1,M2,M3 Protocol]
        L --> M[STM32 Motor Controller<br/>Stepper Actuators]
    end
```

---

## 2. Coordinate System & Physical References

The system establishes a deterministic geometric mapping between camera sensor space, cropped ROI space, and physical real-world millimeters:

1. **Camera Sensor Frame**: Full resolution is $640 \times 480$ pixels.
2. **Default Physical Origin (Basis)**:
   * `DEFAULT_X_BASIS = 245` pixels.
   * `DEFAULT_Y_BASIS = 192` pixels.
   * Represents the physical reference corner where the fabric origin should be aligned. Can be modified in **`utils_constants.py`**.
3. **Basis Vectors**:
   * Horizontal unit vector: $\mathbf{u}_1 = [1, 0]^T$ (along the horizontal axis).
   * Vertical unit vector: $\mathbf{u}_2 = [0, -1]^T$ (pointing upward in the image plane, opposite to image row indexing).
4. **Physical Scale Conversions**:
   * Horizontal scale: `PX_PER_CM_X = 39.0` pixels per centimeter ($\approx 3.9$ px/mm).
   * Vertical scale: `PX_PER_CM_Y = 35.0` pixels per centimeter ($\approx 3.5$ px/mm).
5. **Deadband Tolerances**:
   * Translation: `TRANSLATION_DEADBAND_CM = 0.03` cm ($0.3$ mm).
   * Rotation: `ROTATION_DEADBAND_DEG = 0.1` degrees.
   * Errors within these thresholds are treated as stable (no motor actuation required).These numbers also can be modified in  **`utils_constants.py`**.

---

## 3. Documentation Index & Navigation

The documentation is organized into modular documents corresponding to each core component of the fabric adjustor subsystem:

| Document | Source File | Description |
| :--- | :--- | :--- |
| [**`main_adjustor.md`**](main_adjustor.md) | [`main_adjustor.py`](file:///d:/ai-sewing/main_adjustor.py) | Main pipeline orchestrator, including CLI parameters, 2-phase moderation state machine and motor safety clamping. |
| [**`jetson_uart.md`**](jetson_uart.md) | [`jetson_uart.py`](file:///d:/ai-sewing/jetson_uart.py) | Asynchronous serial communication interface to the STM32 motor driver board. |
| [**`modules_fabric_adjustor.md`**](modules_fabric_adjustor.md) | [`modules/fabric_adjustor.py`](file:///d:/ai-sewing/modules/fabric_adjustor.py) | Core adjustor engine based on line segments: candidate scoring for line selection, Cramer's rule corner derivation, control signal calulation loops, and `NonOrthoLineAdjustor` for fabric edge with arbitary corner values. |
| [**`utils_detector.md`**](utils_detector.md) | [`utils/line_detectors/detector.py`](file:///d:/ai-sewing/utils/line_detectors/detector.py) | Contains helper functions for multi-model line detection preprocessing/postprocessing, ROI transformations, angle clustering, and noisy area filtering. |
| [**`utils_mlsd_postprocess.md`**](utils_mlsd_postprocess.md) | [`utils/line_detectors/mlsd_postprocess.py`](file:///d:/ai-sewing/utils/line_detectors/mlsd_postprocess.py) | Postprocessing procedure for Mobile-LSD model. Including Mobile-LSD junction heatmap decoding, max-pool non-maximum suppression, and displacement map parsing. |
| [**`utils_trt_infer.md`**](utils_trt_infer.md) | [`utils/line_detectors/trt_infer.py`](file:///d:/ai-sewing/utils/line_detectors/trt_infer.py) | TensorRT GPU execution engine for Mobile-LSD model. Including PyCUDA host/device memory transfers, and dual TRT API support. |
| [**`utils_camera.md`**](utils_camera.md) | [`utils/camera.py`](file:///d:/ai-sewing/utils/camera.py) | Thread-safe frame acquisition daemon (`ThreadedCamera`) and GStreamer hardware pipeline fallback hierarchy for video input. |
| [**`utils_constants.md`**](utils_constants.md) | [`utils/constants.py`](file:///d:/ai-sewing/utils/constants.py) | Global physical constants, including coordinate system, basis vectors, scaling factors, and deadband thresholds. |
| [**`utils_helper.md`**](utils_helper.md) | [`utils/helper.py`](file:///d:/ai-sewing/utils/helper.py) | Contains helper functions, including visualization helpers, Kalman tracking utilities and others. |
| [**`utils_tracking_helper.md`**](utils_tracking_helper.md) | [`utils/tracking_helper.py`](file:///d:/ai-sewing/utils/tracking_helper.py) | **Why LineStabilizer & EdgeGradientScorer matter**: Temporal consensus filtering with LineStabilizer, polarity alignment, and Sobel gradient edge scorer with EdgeGradientScorer. |
| [**`utils_viz_adjustor.md`**](utils_viz_adjustor.md) | [`utils/viz_adjustor.py`](file:///d:/ai-sewing/utils/viz_adjustor.py) | HUD graphical overlay rendering: the visualization includes millimeter grid ticks, coordinate axes, detected line visualizers and state indicators. |
| [**`capture_prior.md`**](capture_prior.md) | [`capture_prior.py`](file:///d:/ai-sewing/capture_prior.py) | Capturing golden reference angles alpha and template geometry for non-orthogonal fabric pieces. |
| [**`data_collector_adjustor.md`**](data_collector_adjustor.md) | [`data_collector_adjustor.py`](file:///d:/ai-sewing/data_collector_adjustor.py) | Dataset recording utility for capturing camera frames and ground-truth pose annotations for benchmarking. |
