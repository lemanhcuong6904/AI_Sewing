"""Promptable SAM-assisted fabric-mask annotation app."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter, QPen
from PyQt6.QtWidgets import (QApplication, QButtonGroup, QFileDialog, QFormLayout,
                             QHBoxLayout, QLabel, QLineEdit, QListWidget,
                             QMainWindow, QMessageBox, QPushButton, QRadioButton,
                             QSlider, QSplitter, QVBoxLayout, QWidget)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "sam_vit_b_01ec64.pth"


class MaskCanvas(QWidget):
    """Image canvas supporting SAM point/box prompts and direct mask editing."""
    changed = pyqtSignal()
    prompt_changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(400, 340)
        self.image: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.mode = "add"
        self.brush_size, self.opacity = 24, 110
        self.show_mask = True
        self.points: list[tuple[int, int, int]] = []  # x, y, SAM label (1/0)
        self.box: Optional[tuple[int, int, int, int]] = None
        self._painting = self._erase = False
        self._last: Optional[QPoint] = None
        self._prompt_label: Optional[int] = None
        self._prompt_last: Optional[tuple[int, int]] = None
        self._box_start: Optional[QPoint] = None
        self._box_preview: Optional[QPoint] = None
        self._undo: list[np.ndarray] = []
        self._redo: list[np.ndarray] = []

    def set_data(self, image: np.ndarray, mask: Optional[np.ndarray]) -> None:
        self.image = image
        self.mask = (mask > 0).astype(np.uint8) * 255 if mask is not None else np.zeros(image.shape[:2], np.uint8)
        self.points.clear(); self.box = None; self._undo.clear(); self._redo.clear(); self.update()

    def set_mask(self, mask: np.ndarray, remember: bool = True) -> None:
        if self.image is None or mask.shape != self.image.shape[:2]:
            raise ValueError("Mask dimensions must match the source image.")
        if remember: self._snapshot()
        self.mask = (mask > 0).astype(np.uint8) * 255
        self.changed.emit(); self.update()

    def _snapshot(self) -> None:
        if self.mask is not None:
            self._undo.append(self.mask.copy()); self._undo = self._undo[-30:]; self._redo.clear()

    def undo(self) -> None:
        if self._undo and self.mask is not None:
            self._redo.append(self.mask.copy()); self.mask = self._undo.pop(); self.changed.emit(); self.update()

    def redo(self) -> None:
        if self._redo and self.mask is not None:
            self._undo.append(self.mask.copy()); self.mask = self._redo.pop(); self.changed.emit(); self.update()

    def clear_mask(self) -> None:
        if self.mask is not None: self._snapshot(); self.mask.fill(0); self.changed.emit(); self.update()

    def clear_prompts(self) -> None:
        self.points.clear(); self.box = None; self.prompt_changed.emit(); self.update()

    def _rect(self) -> QRect:
        if self.image is None: return QRect()
        h, w = self.image.shape[:2]; scale = min(self.width() / w, self.height() / h)
        dw, dh = round(w * scale), round(h * scale)
        return QRect((self.width() - dw) // 2, (self.height() - dh) // 2, dw, dh)

    def image_point(self, point: QPoint) -> Optional[tuple[int, int]]:
        if self.image is None: return None
        rect = self._rect()
        if not rect.contains(point): return None
        h, w = self.image.shape[:2]
        return (min(w - 1, max(0, int((point.x() - rect.x()) * w / rect.width()))),
                min(h - 1, max(0, int((point.y() - rect.y()) * h / rect.height()))))

    def display_point(self, x: int, y: int) -> QPoint:
        rect = self._rect(); assert self.image is not None
        h, w = self.image.shape[:2]
        return QPoint(rect.x() + round(x * rect.width() / w), rect.y() + round(y * rect.height() / h))

    def _paint(self, a: QPoint, b: QPoint) -> None:
        if self.mask is None: return
        start, end = self.image_point(a), self.image_point(b)
        if start is None or end is None: return
        cv2.line(self.mask, start, end, 0 if self._erase else 255, self.brush_size, cv2.LINE_AA)
        self.changed.emit(); self.update()

    def _append_prompt(self, point: tuple[int, int], label: int) -> None:
        if self._prompt_last is None or np.hypot(point[0] - self._prompt_last[0], point[1] - self._prompt_last[1]) >= 12:
            self.points.append((*point, label)); self._prompt_last = point
            self.prompt_changed.emit(); self.update()

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        point = event.position().toPoint(); image_point = self.image_point(point)
        if image_point is None: return
        if self.mode in ("positive", "negative") and event.button() == Qt.MouseButton.LeftButton:
            self._prompt_label = 1 if self.mode == "positive" else 0; self._prompt_last = None
            self._append_prompt(image_point, self._prompt_label); return
        if self.mode == "box" and event.button() == Qt.MouseButton.LeftButton:
            self._box_start = point; self._box_preview = point; return
        if self.mode in ("add", "erase") and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self._snapshot(); self._painting = True; self._erase = self.mode == "erase" or event.button() == Qt.MouseButton.RightButton
            self._last = point; self._paint(point, point)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        point = event.position().toPoint()
        if self._painting and self._last is not None: self._paint(self._last, point); self._last = point
        if self._prompt_label is not None:
            image_point = self.image_point(point)
            if image_point is not None: self._append_prompt(image_point, self._prompt_label)
        if self._box_start is not None: self._box_preview = point; self.update()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._painting: self._painting = False; self._last = None
        self._prompt_label = None; self._prompt_last = None
        if self._box_start is not None:
            a, b = self.image_point(self._box_start), self.image_point(event.position().toPoint())
            if a is not None and b is not None and abs(a[0] - b[0]) > 3 and abs(a[1] - b[1]) > 3:
                self.box = (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])); self.prompt_changed.emit()
            self._box_start = self._box_preview = None; self.update()

    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.brush_size = max(1, min(200, self.brush_size + (2 if event.angleDelta().y() > 0 else -2)))

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self); painter.fillRect(self.rect(), QColor("#242424"))
        if self.image is None:
            painter.setPen(QColor("#eeeeee")); painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Choose a dataset and image"); return
        rect = self._rect(); rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        painter.drawImage(rect, QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888))
        if self.show_mask and self.mask is not None and np.any(self.mask):
            overlay = np.zeros((*self.mask.shape, 4), np.uint8); overlay[self.mask > 0] = (0, 220, 120, self.opacity)
            painter.drawImage(rect, QImage(overlay.data, overlay.shape[1], overlay.shape[0], overlay.strides[0], QImage.Format.Format_RGBA8888))
        painter.setPen(QPen(QColor("#eeeeee"), 1)); painter.drawRect(rect)
        for x, y, label in self.points:
            p = self.display_point(x, y); color = QColor("#24d66b") if label else QColor("#f04444")
            painter.setBrush(color); painter.setPen(QPen(QColor("white"), 2)); painter.drawEllipse(p, 7, 7)
        prompt_box = self.box
        if self._box_start is not None and self._box_preview is not None:
            a, b = self.image_point(self._box_start), self.image_point(self._box_preview)
            if a and b: prompt_box = (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
        if prompt_box:
            x1, y1, x2, y2 = prompt_box; a, b = self.display_point(x1, y1), self.display_point(x2, y2)
            painter.setPen(QPen(QColor("#f6d365"), 2, Qt.PenStyle.DashLine)); painter.setBrush(Qt.BrushStyle.NoBrush); painter.drawRect(QRect(a, b).normalized())


class MaskPreview(QWidget):
    """Read-only black-and-white representation of the current binary mask."""
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(280, 340)
        self.mask: Optional[np.ndarray] = None

    def set_mask(self, mask: Optional[np.ndarray]) -> None:
        self.mask = None if mask is None else mask.copy()
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("black"))
        if self.mask is None:
            painter.setPen(QColor("#cccccc"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No mask")
            return
        h, w = self.mask.shape
        scale = min(self.width() / w, self.height() / h)
        dw, dh = round(w * scale), round(h * scale)
        rect = QRect((self.width() - dw) // 2, (self.height() - dh) // 2, dw, dh)
        image = QImage(self.mask.data, w, h, self.mask.strides[0], QImage.Format.Format_Grayscale8)
        painter.drawImage(rect, image)
        painter.setPen(QPen(QColor("#888888"), 1)); painter.drawRect(rect)


class SamModel:
    def __init__(self) -> None:
        self.generator = self.predictor = None
        self.prepared_image: Optional[np.ndarray] = None

    def load(self, checkpoint: Path, model_type: str) -> None:
        try:
            import torch
            from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
        except ImportError as exc: raise RuntimeError("Install SAM first (see README.md).") from exc
        if not checkpoint.is_file(): raise RuntimeError(f"Checkpoint not found: {checkpoint}")
        model = sam_model_registry[model_type](checkpoint=str(checkpoint)); model.to(device="cuda" if torch.cuda.is_available() else "cpu")
        self.generator = SamAutomaticMaskGenerator(model, points_per_side=24, pred_iou_thresh=0.80, stability_score_thresh=0.90)
        self.predictor = SamPredictor(model); self.prepared_image = None

    def prepare(self, image_bgr: np.ndarray) -> None:
        if self.predictor is None: raise RuntimeError("Load SAM and a checkpoint first.")
        if self.prepared_image is not image_bgr:
            self.predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)); self.prepared_image = image_bgr

    def predict_prompts(self, image_bgr: np.ndarray, points: list[tuple[int, int, int]], box: Optional[tuple[int, int, int, int]]) -> tuple[list[np.ndarray], np.ndarray]:
        self.prepare(image_bgr)
        coords = np.array([[x, y] for x, y, _ in points]) if points else None
        labels = np.array([label for _, _, label in points]) if points else None
        bbox = np.array(box) if box else None
        masks, scores, _ = self.predictor.predict(point_coords=coords, point_labels=labels, box=bbox, multimask_output=True)
        return [mask.astype(np.uint8) * 255 for mask in masks], scores

    def automatic(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.generator is None: raise RuntimeError("Load SAM and a checkpoint first.")
        candidates = self.generator.generate(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if not candidates: return np.zeros(image_bgr.shape[:2], np.uint8)
        h, w = image_bgr.shape[:2]
        def rank(item: dict) -> float:
            x, y, bw, bh = item["bbox"]; dist = np.hypot(x + bw / 2 - w / 2, y + bh / 2 - h / 2) / np.hypot(w / 2, h / 2)
            return item["predicted_iou"] + item["stability_score"] + min(item["area"] / (w * h), .85) - .25 * dist
        return max(candidates, key=rank)["segmentation"].astype(np.uint8) * 255


class MainWindow(QMainWindow):
    def __init__(self, checkpoint: str, model_type: str) -> None:
        super().__init__(); self.dataset: Optional[Path] = None; self.paths: list[Path] = []; self.current: Optional[Path] = None; self.sam = SamModel(); self.candidates: list[np.ndarray] = []
        self.has_saved_mask = False; self.mask_dirty = False
        self.setWindowTitle("Fabric Mask Labeler (SAM)"); self.resize(1380, 880)
        self.dataset_edit, self.checkpoint_edit, self.model_edit = QLineEdit(), QLineEdit(checkpoint), QLineEdit(model_type)
        self.dataset_edit.setPlaceholderText("Select data/square_01 or data/square_02"); self.checkpoint_edit.setPlaceholderText("SAM checkpoint (.pth)")
        self.images = QListWidget(); self.images.currentRowChanged.connect(self.open_image); self.canvas = MaskCanvas(); self.canvas.prompt_changed.connect(self.prompt_status); self.canvas.changed.connect(self.mask_changed)
        self.preview = MaskPreview(); self.preview_label = QLabel("Mask preview")
        left = QWidget(); form = QFormLayout(left)
        for edit, handler, title in ((self.dataset_edit, self.choose_dataset, "Dataset"), (self.checkpoint_edit, self.choose_checkpoint, "Checkpoint")):
            row = QHBoxLayout(); row.addWidget(edit); button = QPushButton("Browse..."); button.clicked.connect(handler); row.addWidget(button); form.addRow(title, row)
        form.addRow("Model type", self.model_edit); load = QPushButton("Load SAM"); load.clicked.connect(self.load_sam); form.addRow(load)
        form.addRow(QLabel("Images (✓ = saved label)")); form.addRow(self.images)
        next_unlabeled = QPushButton("Next unlabeled image"); next_unlabeled.clicked.connect(self.next_unlabeled); form.addRow(next_unlabeled)
        controls = QWidget(); controls_layout = QHBoxLayout(controls); controls_layout.setContentsMargins(0, 0, 0, 0); controls_layout.addWidget(QLabel("Tool:"))
        group = QButtonGroup(self); modes = (("Draw mask", "add"), ("Erase mask", "erase"), ("Positive scribble (+)", "positive"), ("Negative scribble (-)", "negative"), ("Box prompt", "box"))
        for index, (label, mode) in enumerate(modes):
            radio = QRadioButton(label); radio.setChecked(index == 0); radio.toggled.connect(lambda checked, value=mode: checked and setattr(self.canvas, "mode", value)); group.addButton(radio); controls_layout.addWidget(radio)
        preview_panel = QWidget(); preview_layout = QVBoxLayout(preview_panel); preview_layout.setContentsMargins(0, 0, 0, 0); preview_layout.addWidget(self.preview_label); preview_layout.addWidget(self.preview, 1)
        views = QSplitter(); views.addWidget(self.canvas); views.addWidget(preview_panel); views.setSizes([760, 340])
        right = QWidget(); layout = QVBoxLayout(right); layout.setContentsMargins(6, 6, 6, 6); layout.addWidget(views, 1); layout.addWidget(controls)
        prompt_buttons = QHBoxLayout()
        for label, handler in (("Run prompts", self.run_prompts), ("Automatic mask", self.automatic), ("Clear prompts", self.canvas.clear_prompts)):
            b = QPushButton(label); b.clicked.connect(handler); prompt_buttons.addWidget(b)
        layout.addLayout(prompt_buttons)
        candidate_buttons = QHBoxLayout(); candidate_buttons.addWidget(QLabel("SAM candidate:"))
        for index in range(3):
            button = QPushButton(str(index + 1)); button.setToolTip(f"Use SAM prompt candidate {index + 1}")
            button.clicked.connect(lambda checked=False, value=index: self.use_candidate(value)); candidate_buttons.addWidget(button)
        candidate_buttons.addStretch(); layout.addLayout(candidate_buttons)
        edit_buttons = QHBoxLayout()
        for label, handler in (("Clear mask", self.canvas.clear_mask), ("Undo", self.canvas.undo), ("Redo", self.canvas.redo), ("Keep largest", self.keep_largest), ("Fill holes", self.fill_holes), ("Save mask", self.save)):
            b = QPushButton(label); b.clicked.connect(handler); edit_buttons.addWidget(b)
        layout.addLayout(edit_buttons); layout.addWidget(QLabel("Mask opacity / brush size (mouse wheel also changes brush size)"))
        slider_row = QHBoxLayout(); opacity = QSlider(Qt.Orientation.Horizontal); opacity.setRange(0, 255); opacity.setValue(110); opacity.valueChanged.connect(lambda value: self.set_opacity(value)); brush = QSlider(Qt.Orientation.Horizontal); brush.setRange(1, 200); brush.setValue(24); brush.valueChanged.connect(lambda value: setattr(self.canvas, "brush_size", value)); slider_row.addWidget(QLabel("Opacity")); slider_row.addWidget(opacity); slider_row.addWidget(QLabel("Brush")); slider_row.addWidget(brush); layout.addLayout(slider_row)
        split = QSplitter(); split.addWidget(left); split.addWidget(right); split.setSizes([340, 1040]); self.setCentralWidget(split)
        self._shortcut(QKeySequence.StandardKey.Save, self.save); self._shortcut(QKeySequence.StandardKey.Undo, self.canvas.undo); self._shortcut(QKeySequence.StandardKey.Redo, self.canvas.redo)
        self.statusBar().showMessage("Load SAM. Green points = fabric; red points = background; yellow box = target region.")

    def _shortcut(self, key, handler) -> None:
        action = QAction(self); action.setShortcut(key); action.triggered.connect(handler); self.addAction(action)

    def set_opacity(self, value: int) -> None: self.canvas.opacity = value; self.canvas.update()
    def prompt_status(self) -> None: self.statusBar().showMessage(f"Prompts: {len(self.canvas.points)} point(s), {'one box' if self.canvas.box else 'no box'}. Run prompts to segment.")
    def label_path(self, image_path: Path) -> Path:
        assert self.dataset is not None
        return self.dataset / "labels" / f"{image_path.stem}.png"

    def image_list_text(self, image_path: Path) -> str:
        return f"✓  {image_path.name}" if self.label_path(image_path).is_file() else f"    {image_path.name}"

    def update_preview_status(self) -> None:
        if self.has_saved_mask and not self.mask_dirty: state = "Saved label"
        elif self.has_saved_mask: state = "Saved label — unsaved edits"
        else: state = "No saved label — unsaved mask"
        self.preview_label.setText(f"Mask preview (white = fabric) — {state}")

    def mask_changed(self) -> None:
        self.preview.set_mask(self.canvas.mask)
        self.mask_dirty = True
        self.update_preview_status()

    def choose_dataset(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select dataset folder")
        if folder: self.load_dataset(Path(folder))
    def choose_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select SAM checkpoint", filter="PyTorch checkpoint (*.pth *.pt)")
        if path: self.checkpoint_edit.setText(path)

    def load_dataset(self, folder: Path) -> None:
        image_dir, label_dir = folder / "images", folder / "labels"
        if not image_dir.is_dir(): QMessageBox.warning(self, "Invalid dataset", "Selected folder must contain images/."); return
        label_dir.mkdir(exist_ok=True); self.dataset = folder; self.dataset_edit.setText(str(folder)); self.paths = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        self.images.clear(); self.images.addItems([self.image_list_text(p) for p in self.paths])
        if self.paths: self.images.setCurrentRow(0)
        self.statusBar().showMessage(f"Loaded {len(self.paths)} images from {folder.name}")

    def open_image(self, index: int) -> None:
        if self.dataset is None or not 0 <= index < len(self.paths): return
        self.current = self.paths[index]; image = cv2.imread(str(self.current), cv2.IMREAD_COLOR)
        if image is None: QMessageBox.warning(self, "Read error", f"Cannot read {self.current.name}"); return
        label = self.label_path(self.current); mask = cv2.imread(str(label), cv2.IMREAD_GRAYSCALE) if label.exists() else None
        if mask is not None and mask.shape != image.shape[:2]: QMessageBox.warning(self, "Invalid mask", f"{label.name} has wrong dimensions."); mask = None
        self.canvas.set_data(image, mask); self.has_saved_mask = mask is not None; self.mask_dirty = False; self.preview.set_mask(self.canvas.mask); self.update_preview_status()
        self.sam.prepared_image = None; self.setWindowTitle(f"Fabric Mask Labeler (SAM) - {self.current.name}")

    def next_unlabeled(self) -> None:
        if not self.paths: return
        start = self.images.currentRow()
        for offset in range(1, len(self.paths) + 1):
            index = (start + offset) % len(self.paths)
            if not self.label_path(self.paths[index]).is_file():
                self.images.setCurrentRow(index); self.statusBar().showMessage(f"Opened next unlabeled image: {self.paths[index].name}"); return
        self.statusBar().showMessage("Every image in this dataset already has a saved label.")

    def load_sam(self) -> None:
        try: self.statusBar().showMessage("Loading SAM..."); QApplication.processEvents(); self.sam.load(Path(self.checkpoint_edit.text()), self.model_edit.text().strip()); self.statusBar().showMessage("SAM loaded. Add prompts or use Automatic mask.")
        except Exception as exc: QMessageBox.critical(self, "Could not load SAM", str(exc))

    def run_prompts(self) -> None:
        if self.canvas.image is None: return
        if not self.canvas.points and self.canvas.box is None: QMessageBox.information(self, "No prompt", "Add a positive/negative point or draw a box first."); return
        try:
            self.statusBar().showMessage("SAM is applying prompts..."); QApplication.processEvents(); self.candidates, scores = self.sam.predict_prompts(self.canvas.image, self.canvas.points, self.canvas.box)
            self.canvas.set_mask(self.candidates[int(np.argmax(scores))]); self.statusBar().showMessage("Prompt mask ready. The highest-confidence candidate was selected; refine it with the brush.")
        except Exception as exc: QMessageBox.warning(self, "Prompt prediction unavailable", str(exc))

    def use_candidate(self, index: int) -> None:
        if index >= len(self.candidates):
            self.statusBar().showMessage("Run prompts first to obtain SAM candidates."); return
        self.canvas.set_mask(self.candidates[index])
        self.statusBar().showMessage(f"Selected SAM candidate {index + 1}; refine it with brush tools if needed.")

    def automatic(self) -> None:
        if self.canvas.image is None: return
        try: self.statusBar().showMessage("Generating automatic masks..."); QApplication.processEvents(); self.canvas.set_mask(self.sam.automatic(self.canvas.image)); self.statusBar().showMessage("Automatic mask ready; refine with prompts or brush.")
        except Exception as exc: QMessageBox.warning(self, "Prediction unavailable", str(exc))

    def keep_largest(self) -> None:
        if self.canvas.mask is None: return
        count, labels, stats, _ = cv2.connectedComponentsWithStats((self.canvas.mask > 0).astype(np.uint8), 8)
        if count > 1: self.canvas.set_mask((labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8) * 255)
    def fill_holes(self) -> None:
        if self.canvas.mask is None: return
        inverse = cv2.bitwise_not(self.canvas.mask); flood = inverse.copy(); h, w = flood.shape; cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 0); self.canvas.set_mask(cv2.bitwise_or(self.canvas.mask, flood))
    def save(self) -> None:
        if self.dataset is None or self.current is None or self.canvas.mask is None: return
        output = self.label_path(self.current)
        if cv2.imwrite(str(output), self.canvas.mask):
            self.has_saved_mask = True; self.mask_dirty = False; self.preview.set_mask(self.canvas.mask); self.update_preview_status()
            row = self.images.currentRow()
            if row >= 0: self.images.item(row).setText(self.image_list_text(self.current))
            self.statusBar().showMessage(f"Saved {output.name}. Use Next unlabeled image to continue.")
        else: QMessageBox.critical(self, "Save error", f"Cannot write {output}")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT)); parser.add_argument("--model-type", default="vit_b", choices=("vit_b", "vit_l", "vit_h")); args = parser.parse_args()
    app = QApplication(sys.argv); window = MainWindow(args.checkpoint, args.model_type); default_dataset = PROJECT_ROOT / "data" / "square_01"
    if default_dataset.is_dir(): window.load_dataset(default_dataset)
    window.show(); return app.exec()


if __name__ == "__main__": raise SystemExit(main())
