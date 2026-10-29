# AI Sewing — Tổng quan dự án (Tiếng Việt)

> Tài liệu này được viết sau khi đọc mã nguồn trong thư mục dự án. Mục tiêu: giúp người mới hiểu dự án làm gì, chạy thế nào, mỗi file có vai trò gì và các thuật toán chính hoạt động ra sao.
> Các con số (hằng số, ngưỡng, mặc định) lấy từ mã tại thời điểm đọc; vài chỗ giữa mã và thư mục `docs/` đang lệch nhau (xem mục 12).

---

## 1. Dự án là gì?

**AI Sewing** là hệ thống thị giác máy tính chạy trên **NVIDIA Jetson Orin**, điều khiển một máy may/cơ cấu kẹp vải tự động. Camera nhìn xuống bàn may, hệ thống xử lý ảnh và gửi lệnh qua **UART** tới vi điều khiển **STM32** (điều khiển động cơ bước/động cơ).

Dự án gồm **hai hệ thống con** độc lập:

| Hệ thống | Entry point | Việc làm |
| :-- | :-- | :-- |
| **Sewer** (dẫn vải khi may) | `main_sewer.py` | Phát hiện mép vải chạy dưới kim, điều khiển 2 động cơ con lăn (roller) để vải đi thẳng theo đường may. |
| **Fabric Adjustor** (căn chỉnh vải trước khi may) | `main_adjustor.py`, `gui_main.py` | Đo vị trí/góc của miếng vải so với mốc chuẩn, rồi gửi lệnh dịch chuyển (mm) và xoay (độ) để đưa vải về đúng vị trí. |

Phần lớn mã và toàn bộ thư mục `docs/` tập trung vào **Fabric Adjustor** — đây là phần phát triển mới nhất (dữ liệu và file cấu hình ghi nhận đến khoảng tháng 5–6/2026).

---

## 2. Cấu trúc thư mục

```
ai-sewing-master/
├── main_sewer.py              # Điều khiển con lăn (hệ Sewer), chạy trên Jetson
├── main_adjustor.py           # Căn chỉnh vải qua dòng lệnh (hệ Adjustor) – lớp FabricAdjustorApp
├── gui_main.py                # Giao diện Tkinter cho Adjustor (bản "native")
├── jetson_uart.py             # Hàm gửi/nhận UART với STM32 (+ CLI thử)
├── capture_prior.py           # Chụp ảnh/góc "chuẩn" (prior) cho vải
├── data_collector_adjustor.py # Thu thập dữ liệu + nhãn thật (ground truth)
├── evaluate_adjustor.py       # Đánh giá offline (vải vuông góc)
├── evaluate_adjustor_experiment.py  # Bản thử nghiệm của đánh giá
├── evaluate_arbitrary.py      # Đánh giá cho vải góc bất kỳ (non-orthogonal)
├── export_dexined.py          # Xuất DexiNed (.pth) sang ONNX
├── build_trt_engine.sh        # Build TensorRT engine FP16 từ ONNX bằng trtexec
├── setup_autostart.sh         # Tạo autostart để gui_main.py chạy khi Jetson khởi động
├── test_angle.py, test_trt.py # Script thử nhỏ (tính góc có dấu; kiểm tra engine TRT)
├── trt_output.txt             # Log lỗi CUDA/TensorRT khi thử trên máy không có GPU
├── test _cases_for_labeling*.txt  # Danh sách ca thử cần chụp/gán nhãn (xem mục 8)
├── requirements.txt
│
├── modules/
│   ├── fabric_adjustor.py        # LineDetectionAdjustor, NonOrthoLineAdjustor (dựa trên đường thẳng)
│   ├── fabric_adjustor_prior.py  # PriorMaskAdjustor (DexiNed + so khớp mask)
│   └── fabric_sewer.py           # Vòng lặp camera → tín hiệu điều khiển cho Sewer
│
├── utils/
│   ├── camera.py               # ThreadedCamera, get_camera(), khoảng cách histogram
│   ├── constants.py            # Hằng số vật lý: gốc toạ độ, px/cm, deadband, pivot
│   ├── helper.py               # Tiện ích: mask nền, điểm điều khiển, Kalman 1D, vẽ/lưu ảnh
│   ├── mask_helpers.py         # Hàm so khớp mask: moments, bbox, ecc, feature, contour, chamfer
│   ├── tracking_helper.py      # LineStabilizer, EdgeGradientScorer
│   ├── viz_adjustor.py         # Vẽ lưới, trục, HUD cho Adjustor
│   ├── line_detectors/
│   │   ├── detector.py         # DetectorMode, tiền xử lý, LSD/FLD/EDLines/ensemble/MLSD, lọc đường
│   │   ├── mlsd_postprocess.py # Hậu xử lý đầu ra Mobile-LSD
│   │   └── trt_infer.py        # Chạy M-LSD bằng TensorRT (PyCUDA)
│   └── edge_detectors/
│       ├── dexined_inference.py    # DexiNed bằng PyTorch (CPU/GPU)
│       └── tensorrt_inference.py   # DexiNed bằng TensorRT
│
├── models/
│   ├── dexined/   # dexined.onnx, dexined_10_model.pth, dexined_fp16.engine, dexined_model.py
│   └── m-lsd/     # mlsd_tiny_512_fp32 (.pth/.onnx), mbv2_mlsd_tiny.py
│
├── mask_priors/   # Ảnh vải mẫu (0–3.png) và các mask tiền tính (0_mask.png, ...)
├── adjustor_data/ # Bộ dữ liệu ảnh + nhãn (xem mục 8)
├── docs/fabric_adjustor/  # 14 file tài liệu kỹ thuật (tiếng Anh) cho Adjustor
└── testing/       # Script thử/profiling: testing.py, test_uart.py, test_mlsd.py, test_mask_matching.py
```

---

## 3. Kiến trúc phần cứng

```
Camera USB (640×480, 30 fps)
        │
        ▼
 NVIDIA Jetson Orin  ──UART 115200──►  STM32 (driver động cơ)
 (Python, OpenCV, TensorRT)  ◄──────  "OK", "L1", "L2", ...
                                          │
                                          ▼
                              Động cơ M1 (X), M2 (Y), M3 (xoay),
                              kẹp vải (clamp), con lăn (Sewer)
```

- **Camera**: `utils/camera.py` thử lần lượt 3 cách mở camera: GStreamer (`v4l2src` + `nvjpegdec`, tối ưu cho Jetson) → V4L2 → `VideoCapture(0)` mặc định. Lớp `ThreadedCamera` đọc liên tục ở luồng nền để luôn có khung hình mới nhất (bỏ khung cũ, tránh trễ).
- **Cổng serial**: mặc định `/dev/ttyUSB0`, 115200 baud (có thể là `/dev/ttyTHS1` nếu dùng chân UART của Jetson). `jetson_uart.get_default_port()` tự dò thiết bị USB/ACM.
- **Phần mềm STM32 không nằm trong repo này** — chỉ thấy giao thức qua mã phía Jetson.

---

## 4. Hệ thống Sewer (dẫn vải khi may)

**Files**: `main_sewer.py`, `modules/fabric_sewer.py`.

Ý tưởng: camera nhìn vùng nhỏ quanh mũi kim. Nếu mép vải lệch khỏi vị trí mong muốn, một trong hai con lăn được dừng hoặc quay ngược để chỉnh hướng vải.

Luồng xử lý (`run_cam_return_signal`):
1. Mở camera, chờ 4 giây cho ổn định, chụp khung đầu làm **histogram tham chiếu** (nền trống).
2. Mỗi khung mới: cắt ROI (`DEFAULT_CROP_ROI = (192, 288, 236, 403)`), làm mờ, tiền xử lý.
3. Tìm đường thẳng (mặc định LSD; có thể FLD, EDLines, ensemble, MLSD), lọc những đường gần nền (`keep_lines_near_background`).
4. Từ các đường, lấy **2 điểm điều khiển** (giao với một đường ngang) → độ lệch `error1`, `error2` của hai bên.
5. Cộng **bù độ trễ theo vận tốc**: `error += step_overhead × hướng × velocity[tốc_độ]` (velocity mặc định `[2.5, 3.5, 7.0]` cho 3 mức tốc độ).
6. **Kalman 1D** (`Kalman1DTracker`, cho phép mất tối đa 2 khung) làm mượt/dự đoán khi mất đường.
7. Tạo tín hiệu `[err >= min, err >= max]` với `interval` (mặc định `(-14, 15)` ở CLI). Nếu không thấy đường mà khoảng cách histogram so với nền trống vượt `hist_threshold` thì coi như vải phủ kín → `[True, True]`.

Bảng ánh xạ tín hiệu → ký tự gửi STM32 (trong `main_sewer.py`):

| Tín hiệu | Ý nghĩa | Ký tự gửi |
| :-- | :-- | :-- |
| `[False, False]` | Giữ nguyên, quay tiến | `'1'/'2'/'3'` (tốc độ hiện tại) |
| `[True, False]` | Dừng động cơ | `'a'` |
| `[True, True]` | Quay ngược | `'b'` |

Phím điều khiển (terminal raw mode): `1` chậm, `2` vừa, `3` nhanh, `q` thoát (gửi `'0'` tắt máy). Chỉ gửi lại UART khi lệnh thay đổi.

> Lưu ý: `main_sewer.py` dùng `tty`/`termios`, chỉ chạy trên Linux/Jetson. Chương trình chỉ điều khiển một cổng serial (cổng 1).

---

## 5. Hệ thống Fabric Adjustor (căn chỉnh vải)

### 5.1. Bài toán

Một cánh tay/kẹp đưa miếng vải vào vùng nhìn của camera. Hệ thống phải tính **vải đang lệch bao nhiêu** so với **mốc chuẩn** (góc vải phải trùng điểm `(x_basis, y_basis)` và các cạnh phải thẳng hàng với trục), rồi ra lệnh dịch/xoay để bù. Sau mỗi lần chỉnh thì đo lại (vòng kín – *visual servoing*), tối đa vài vòng, cuối cùng thả vải (`k`).

### 5.2. Hệ toạ độ & hằng số (`utils/constants.py`)

| Hằng số | Giá trị trong code | Ý nghĩa |
| :-- | :-- | :-- |
| `DEFAULT_X_BASIS / Y_BASIS` | 251 / 200 px | Điểm gốc mong muốn của góc vải trong ảnh camera |
| `U1_VECTOR` | `[1, 0]` | Trục ngang |
| `U2_VECTOR` | `[0, -1]` | Trục dọc, hướng lên (ngược chiều chỉ số hàng ảnh) |
| `PX_PER_CM_X / Y` | 39 / 35 | Hệ số đổi pixel → cm |
| `AXIS_LENGTH_PX` | 39 × 12 = 468 | Độ dài trục vẽ trên HUD |
| `TRANSLATION_DEADBAND_CM` | 0.03 cm | Dưới ngưỡng này coi như không lệch |
| `ROTATION_DEADBAND_DEG` | 0.1° | Tương tự cho góc |
| `DEFAULT_PIVOT` | (545, −370) px | Tâm xoay của tay kẹp trong toạ độ camera |

### 5.3. Ba lựa chọn "bộ phát hiện" (backend)

`DetectorMode` có 6 giá trị: `lsd`, `fld`, `edlines`, `ensemble`, `mlsd`, `dexined`. Chia thành hai họ:

**(a) Họ đường thẳng — `modules/fabric_adjustor.py`**

- `LineDetectionAdjustor`: dành cho vải có góc **vuông (90°)**.
- `NonOrthoLineAdjustor` (kế thừa): dành cho vải có góc **bất kỳ** (không vuông), dùng một góc chuẩn α đã chụp bằng `capture_prior.py`.

Quy trình:
1. **Tiền xử lý** (`preprocessing_adjustor`): cắt ROI, làm mờ Gaussian (`smooth_texture`), tuỳ chọn tăng tương phản.
2. **Tìm đường thẳng** (M-LSD chạy TensorRT hoặc LSD/FLD/EDLines của OpenCV), rồi lọc: loại đường ở vùng nhiễu (đồ gá, tấm đế), đường nằm xa nền, gộp các đường gần như cùng đường (`merge_collinear_lines`), chỉ giữ các hướng chủ đạo (`filter_dominant_angles`).
3. **Chấm điểm đường** bằng gradient Sobel dọc đường (`EdgeGradientScorer`) — mép vải thật có gradient mạnh hơn vân vải.
4. **Chọn cặp đường tốt nhất** (`_select_best_lines_gradient_priority`): loại cặp gần song song (chênh góc < 30° hoặc > 150°), rồi chấm điểm kết hợp *gradient + độ dài − khoảng cách tới gốc − sai lệch góc so với α*. Trọng số gradient lớn (≈ 23) để ưu tiên mép thật.
5. **Ổn định theo thời gian** (`LineStabilizer`): gom nhiều khung (mặc định `target_frames=150` ở chế độ tức thời), gom cụm theo toạ độ Hough (ρ, θ) và lấy đồng thuận để giảm nhiễu.
6. **Tính góc vải** bằng quy tắc Cramer (giao điểm hai đường), chiếu vector sai lệch lên `U1`, `U2` → `(tx, ty)` px, và góc xoay có dấu (về khoảng [−90°, 90°]).
7. **Bù xoay quanh pivot** (`compensate_rotation`): vì kẹp xoay quanh một tâm (pivot) chứ không quay quanh góc vải, tịnh tiến cần được hiệu chỉnh theo phép quay `P + R(θ)(I − P)`.

**(b) Họ DexiNed + mask — `modules/fabric_adjustor_prior.py`**

- Lớp `PriorMaskAdjustor`: dùng cho vải **cong/hình dạng bất kỳ**, cần một **ảnh mẫu** (`--prior-mask`, thường nằm trong `mask_priors/`).
- Quy trình: DexiNed (mạng phát hiện biên, chạy TensorRT FP16, batch tối đa 5, đầu vào 400×640) tạo mask biên → `apply_morphology` (`none/erode/open/fill`) → **so khớp mask hiện tại với mask mẫu** để suy ra `(tx, ty)` và góc.
- Các phương pháp so khớp (`utils/mask_helpers._match_masks`): `moments`, `bbox`, `ecc` (OpenCV ECC), `feature`, `contour` (mặc định), `chamfer`, `chamfer_fast`.
- Có luồng xử lý batch song song (ThreadPoolExecutor) và `_filter_and_mean` để loại giá trị ngoại lai trong batch rồi lấy trung bình.

### 5.4. Từ tín hiệu thị giác tới lệnh động cơ

Hàm trung tâm: `FabricAdjustorApp.process_control_signals` (trong `main_adjustor.py`), logic tương tự ở `gui_main.py`.

1. Đổi đơn vị: `tx_cm = tx_px / 39`, `ty_cm = ty_px / 35`.
2. **Kẹp an toàn** (tránh va cơ khí): `tx_cm, ty_cm ∈ [−2.0, +3.0] cm`; góc xoay ∈ `[−10°, +10°]`.
3. Đổi sang mm: `M1 = tx_cm × 10`, `M2 = ty_cm × 10`, `M3 = góc (độ)`.
4. **Máy trạng thái "điều tiết" (moderation)** — mỗi vòng chia hai nửa: *tịnh tiến trước, xoay sau*:
   - Nếu dịch chuyển nhỏ hơn deadband thì đặt về 0.
   - `max_moderations` (mặc định 2): tổng số vòng chỉnh.
   - `max_angle_moderations` (mặc định 2): số vòng đầu cho phép xoay.
5. Gửi chuỗi `"M1, M2, M3"` (mm, mm, độ) qua UART.
6. Cập nhật **pivot** theo chuyển động đã thực hiện (`ΔP = −(tx·u1 + ty·u2)`).
7. Hết ngân sách vòng → `release_fabric()` gửi `'k'` (thả kẹp, tay quay về vị trí home), đặt trạng thái `l1_done`/`l2_done`.

### 5.5. Các pha (phase) và giao thức UART

Quy trình may có hai "lần kẹp" tương ứng hai pha:

| Pha | Mô tả | Gốc toạ độ dùng |
| :-- | :-- | :-- |
| `L1` | Pha 1: căn chỉnh thô/đầu tiên | `(x_basis, y_basis)` |
| `L2a` | Pha 2a: lệch thêm để căn chỉnh sơ bộ | gốc L2b + offset L2a (mặc định x −0.5 cm, y 0) |
| `L2b` | Pha 2b: căn chỉnh tinh | gốc + offset L2b (mặc định x 0.0, y +0.4 cm ở CLI; GUI mặc định 0.175 / 0.3 cm) |

Tin nhắn **STM32 → Jetson**:
- `OK`: STM32 đã chạy xong chuyển động, Jetson được phép đo lại.
- `L1`, `L2`, `L2a`, `L2b`: bắt đầu pha tương ứng (Jetson chuyển gốc toạ độ và bắt đầu đo).

Tin nhắn **Jetson → STM32**:
- `"M1, M2, M3"` — lệnh dịch/xoay.
- `k` — thả kẹp, quay về home.
- `e1` / `e2` — báo lỗi (pha L1 / L2) khi không phát hiện được vải sau `max_retries` (mặc định 20) lần liên tiếp.
- `s`, `v` — lệnh trạng thái/đọc toạ độ (dùng khi thử), `k5` được GUI dùng để thả.

Chạy kiểu sự kiện (`run_event_driven`, mặc định, dùng trên máy thật):
> chờ trigger (UART hoặc bàn phím) → thu `target_frames` khung → tính tín hiệu → gửi lệnh → chờ `OK` → lặp, cho đến hết ngân sách → `k`.

Chạy liên tục (`--continuous`, cũ hơn): dùng generator `adjustor.run()`, tính tín hiệu liên tục.

### 5.6. Phím tắt khi chạy `main_adjustor.py` (terminal)

| Phím | Tác dụng |
| :-- | :-- |
| `l1` | Chuyển sang pha L1 và đo ngay |
| `l2`, `l2a`/`2a`, `l2b`/`2b` | Chuyển sang pha tương ứng |
| `s` | Gửi trạng thái tới STM32 |
| `k` | Thả kẹp thủ công, về `idle` |
| `o` | Chỉ ở `--dry-run`: giả lập STM32 gửi `OK` |
| `q`/`x` | Thoát |
| `1`/`2` | Gửi lệnh thử rồi thoát |

### 5.7. Tham số dòng lệnh chính của `main_adjustor.py`

`--detector` (mặc định `dexined`), `--prior-mask`, `--match-method`, `--morphology`, `--x-basis/--y-basis`, `--batch-size` (10), `--max-moderations` (2), `--max-angle-moderations` (2), `--max-retries` (20), `--crop-roi`, `--target-frames` (150), `--rot-alignment {horizontal,vertical}`, `--x-offset-cm/--y-offset-cm`, `--max-moderations-l2a`, `--x-offset-cm-l2a/--y-offset-cm-l2a`, `--pivot`, `--no-compensate-rotation`, `--continuous`, `--visualize`, `--port`, `--baud`, `--dry-run`.

Ví dụ:
```bash
# Chạy thật với DexiNed + ảnh mẫu
python main_adjustor.py --detector dexined --prior-mask mask_priors/1.png

# Thử không cần phần cứng STM32 (in lệnh ra màn hình)
python main_adjustor.py --detector mlsd --dry-run --visualize
```

### 5.8. Giao diện đồ hoạ `gui_main.py`

Ứng dụng **Tkinter** ("AI Sewing - Native Adjustor"), cửa sổ 1200×800:
- Bên trái: các ô cấu hình (detector, match method, morphology, gốc X/Y, offset L2, prior mask, batch size, số vòng điều tiết, cổng/baud, các hộp tick *Enhance Contrast*, *Smooth Texture*, *Continuous*, *Dry Run*) và các nút **START ENGINE**, **Trigger L1/L2**, **Dry-Run OK**.
- Bên phải: hình camera trực tiếp kèm lưới/trục, thanh trạng thái.
- Cấu hình được lưu vào `gui_config.json` (file này bị `.gitignore` bỏ qua vì `*.json`).
- Mặc định `dry_run = True` để an toàn.
- Khi dùng DexiNed bắt buộc phải có prior mask; nếu có prior mask với detector đường thẳng thì dùng `NonOrthoLineAdjustor`.
- `setup_autostart.sh` tạo file `~/.config/autostart/ai-sewing.desktop` để GUI tự chạy khi Jetson khởi động (cần chạy script từ thư mục dự án).

---

## 6. Các mô hình AI

| Mô hình | Vai trò | Vị trí | Định dạng |
| :-- | :-- | :-- | :-- |
| **DexiNed** | Phát hiện biên dày đặc cho vải cong/phức tạp | `models/dexined/` | `.pth` (~141 MB), `.onnx` + `.onnx.data`, `dexined_fp16.engine` (~72 MB) |
| **M-LSD (Mobile-LSD tiny, 512)** | Phát hiện đoạn thẳng nhẹ, nhanh | `models/m-lsd/` | `.pth`, `.onnx` |

Quy trình triển khai TensorRT:
1. `export_dexined.py` xuất ONNX từ trọng số `.pth`.
2. `build_trt_engine.sh` chạy `trtexec` tạo engine FP16 với `min/opt` batch 1, `max` batch 5, kích thước `1x3x400x640`. Script cảnh báo sẽ mất vài phút trên Jetson Orin.
3. `utils/edge_detectors/tensorrt_inference.py` (`TrtDexiNed`) và `utils/line_detectors/trt_infer.py` (`TRTInfer`) nạp engine và chạy suy luận bằng PyCUDA.
4. `test_trt.py` kiểm tra nhanh engine nạp/chạy được. File `trt_output.txt` là log thử trên máy không có CUDA (lỗi `CUDA initialization failure`), không phải kết quả đúng.

> Engine TensorRT gắn với phiên bản TensorRT/GPU cụ thể — phải build lại trên thiết bị đích.

---

## 7. Công cụ phụ trợ

### 7.1. `capture_prior.py` — chụp "prior"
Dùng khi cần căn chỉnh vải góc không vuông hoặc dùng DexiNed:
- `--mode line`: dùng `NonOrthoLineAdjustor` + M-LSD + `LineStabilizer(window=55)` để đo **góc α chuẩn** (có thể lưu thành file prior).
- `--mode edge`: chụp ảnh vải đã đặt đúng chỗ, lưu vào `mask_priors/` làm mẫu cho DexiNed.
- Phím: `Space` chụp, `q` thoát không lưu.

### 7.2. `data_collector_adjustor.py` — thu thập dữ liệu
- Chế độ thủ công: nhấn `Space` để chụp ảnh, rồi nhập nhãn thật; lưu vào `adjustor_data/<thời gian>/data/N.png` và `.../label/N.txt`.
- Chế độ bán tự động (`main_semi_auto`): đọc danh sách nhãn từ file `test _cases_for_labeling*.txt`, hướng dẫn người dùng đặt vải theo từng ca rồi chụp.

### 7.3. `evaluate_*.py` — đánh giá offline
- `evaluate_adjustor.py --dir <thư_mục_con_trong_adjustor_data> [--detector ...] [--x-basis ...]`: chạy pipeline trên ảnh `data/`, so sánh với `label/`, ghi `eval.txt` (từng mẫu + sai số trung bình) và ảnh trực quan `eval_vis/`.
- `evaluate_arbitrary.py`: tương tự cho vải góc bất kỳ; `evaluate_adjustor_experiment.py`: bản thử nghiệm.
- Hỗ trợ `--match-method` gồm cả `chamfer`, `chamfer_fast`, và `--prior-img` cho DexiNed.

### 7.4. `testing/`
- `testing.py`: profiler tốc độ pipeline Sewer, dò ngưỡng histogram (`tune_histogram_threshold`), thống kê khung bị mất đường, xem camera thô.
- `test_uart.py`: thử điều khiển động cơ qua UART kèm xem camera (đang dùng hàm `_compute_control_signal` — hàm này **không còn trong** `utils/helper.py`, nên file có thể đã cũ).
- `test_mlsd.py`, `test_mask_matching.py`: thử M-LSD và các thuật toán so khớp với mask giả.

---

## 8. Dữ liệu (`adjustor_data/`)

Mỗi thư mục con chứa `data/N.png` (ảnh camera) và `label/N.txt` (nhãn thật), nhãn gồm **4 số**: `tx_cm ty_cm góc_ngang góc_dọc` (ví dụ `1.0 0.0 0.0 0.0`). Số lượng file hiện có:

| Thư mục | Số file | Ý nghĩa (suy từ tên) |
| :-- | --: | :-- |
| `square-has_light` | 203 | Vải vuông, có đèn |
| `square-no_light` | 241 | Vải vuông, không đèn |
| `non_square-has_light` | 224 | Vải không vuông, có đèn |
| `non_square-has_light-5cm` | 490 | Vải không vuông, cạnh 5 cm, có đèn (có `eval.txt` kết quả) |
| `20260527_..._non_square_newbg` | 24 | Vải không vuông, nền mới (đợt chụp 27/05/2026) |

Các file `test _cases_for_labeling*.txt` liệt kê các ca cần chụp, mỗi dòng dạng `<tx> <ty> <góc_H> <góc_V>`, ví dụ `1 0 -15 -15` — tức dịch X 1 cm, Y 0, xoay −15°. Có ba bản: vuông, không vuông, không vuông 5 cm.

Kết quả mẫu trong `adjustor_data/non_square-has_light-5cm/eval.txt` cho thấy một số mẫu sai lớn (vài cm, hàng chục độ), tức pipeline cho vải không vuông chưa ổn định ở mọi ca; cần xem lại phiên bản/tham số đã dùng để tạo file này.

`mask_priors/` chứa ảnh mẫu `0–3.png` (~360 KB mỗi ảnh) và vài mask tiền tính `0_mask.png`, `0_full_mask_{none,erode,fill}.png`.

---

## 9. Cài đặt & môi trường

`requirements.txt`: `numpy`, `opencv-contrib-python==4.5.5.64`, `pyserial`, `matplotlib`, `llvmlite==0.31.0`, `numba==0.47.0`, `onnxruntime-gpu`, `pycuda`.
Ngoài ra mã còn dùng `Pillow` (GUI), `torch` (DexiNed `.pth`), `tensorrt` và `trtexec` (cài cùng JetPack) — các thư viện này **không có trong `requirements.txt`**.

Gợi ý chạy trên Jetson:
```bash
# Cấp quyền cổng serial
sudo chmod 666 /dev/ttyUSB0

# Build engine (một lần, vài phút)
./build_trt_engine.sh

# Chạy căn chỉnh
python3 main_adjustor.py --detector dexined --prior-mask mask_priors/1.png
# hoặc GUI
python3 gui_main.py
```

Lưu ý khi dùng Windows: `main_adjustor.py` có `try/except` cho `tty/termios` nên import được, nhưng `main_sewer.py` và `testing/test_uart.py` thì không. Thực tế chạy thật là trên Linux (Jetson).

`.gitignore` bỏ qua: `__pycache__`, `captured_frames/`, `mlsd_pytorch/`, `TP-LSD/`, `*.ipynb`, `*.pyc`, `*.json`.

---

## 10. Sơ đồ luồng tổng thể (Adjustor)

```
STM32 gửi "L1"/"OK" ──► Jetson đặt trigger_compute
                              │
        ┌─────────────────────▼─────────────────────┐
        │ Thu target_frames khung (xoá bộ đệm camera) │
        └─────────────────────┬─────────────────────┘
                              ▼
   Tiền xử lý (ROI, blur) ──► Detector (M-LSD/LSD/... hoặc DexiNed)
                              ▼
   Chọn cặp đường / so khớp mask ──► ổn định đa khung (đồng thuận)
                              ▼
   (tx, ty) px + góc ──► đổi cm/mm ──► kẹp an toàn ──► deadband
                              ▼
   Máy trạng thái: dịch trước, xoay sau
                              ▼
        gửi "M1, M2, M3" ──► STM32 chạy ──► gửi "OK" ──► lặp
                              ▼ (hết ngân sách vòng)
                        gửi "k" (thả vải)
```

---

## 11. Bảng tra nhanh: khái niệm hay gặp

| Thuật ngữ | Ý nghĩa |
| :-- | :-- |
| **Basis (gốc)** | Điểm pixel nơi góc vải cần nằm |
| **Pivot** | Tâm xoay của tay kẹp trong toạ độ camera |
| **Prior** | Mẫu tham chiếu "chuẩn" (ảnh/mask hoặc góc α) |
| **Deadband** | Vùng chết: sai lệch nhỏ hơn ngưỡng thì bỏ qua |
| **Moderation** | Một vòng điều chỉnh (dịch rồi xoay) |
| **Roller** | Con lăn dẫn vải (hệ Sewer) |
| **TRT** | TensorRT — chạy mô hình nhanh trên GPU Jetson |
| **M1/M2/M3** | Động cơ dịch X (mm), dịch Y (mm), xoay (độ) |

---

## 12. Điểm cần lưu ý / không nhất quán phát hiện khi đọc code

1. **Giá trị mặc định trong `docs/` đã cũ so với code**: tài liệu ghi `DEFAULT_X_BASIS = 245`, `Y = 192`, trong khi `utils/constants.py` hiện là `251` và `200`. Đường dẫn trong docs trỏ tới `d:/ai-sewing/...` (máy cũ). Pivot home trong docs là `(385, -20)` còn `DEFAULT_PIVOT` là `(545, -370)`.
2. **Deadband dịch chuyển**: docstring `process_control_signals` ghi 0.05 cm, hằng số thực tế là 0.03 cm.
3. **`main_sewer.py`** gọi `run_cam_return_signal` với đối số `velocity`, `step_overhead`, `speed_state`; hàm `main()` truyền `step_overhead=1.2` nhưng `parse_args` mặc định `interval=(-14, 15)` còn hàm `main` mặc định `(-3, 3)` — giá trị thực phụ thuộc đường gọi.
4. **`gui_main.py` import `get_default_port, send_and_receive` từ `main_adjustor`** (nhờ `main_adjustor` đã import chúng từ `jetson_uart`) — chạy được nhưng nên import thẳng từ `jetson_uart`.
5. **`jetson_uart.py`**: hàm `main()` gọi `interactive_mode(ser)` nhưng hàm này đã bị comment → chạy không tham số sẽ lỗi `NameError`.
6. **`testing/test_uart.py`** import `_compute_control_signal` không còn tồn tại trong `utils/helper.py`; chuỗi tiếng Việt trong file cũng bị lỗi mã hoá (mojibake).
7. **`requirements.txt`** thiếu `torch`, `Pillow`, `tensorrt` và có `llvmlite==0.31.0 / numba==0.47.0` rất cũ (chọn để tương thích Python/Jetson cũ).
8. Do các dòng bị comment, tuỳ chọn `enhance_contrast` hoạt động khác tên gọi: trong `preprocessing_sewer` nó chạy CLAHE rồi bình phương độ sáng; trong `preprocessing_adjustor` nó chỉ còn biến đổi lũy thừa bậc 3 (CLAHE đã bị comment).
9. Dữ liệu thử `eval.txt` cho thấy sai số lớn ở nhiều ca vải không vuông (xem mục 8).

---

## 13. Nên đọc theo thứ tự nào?

1. `utils/constants.py` → hiểu hệ toạ độ.
2. `main_adjustor.py` (`FabricAdjustorApp`) → luồng chạy tổng thể, giao thức UART.
3. `modules/fabric_adjustor.py` và `modules/fabric_adjustor_prior.py` → hai thuật toán đo.
4. `utils/line_detectors/detector.py`, `utils/tracking_helper.py`, `utils/mask_helpers.py` → bước tiền/hậu xử lý.
5. `docs/fabric_adjustor/*.md` → mô tả toán học chi tiết từng module (lưu ý mục 12).
6. `evaluate_adjustor.py` + `adjustor_data/` → cách đo chất lượng.
7. `main_sewer.py` + `modules/fabric_sewer.py` → hệ dẫn vải (nếu cần).
