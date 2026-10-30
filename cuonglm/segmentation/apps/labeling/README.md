# Fabric Mask Labeler

PyQt6 desktop application for creating binary fabric masks. It loads images from
`data/<dataset>/images`, predicts an initial mask with the pretrained Segment
Anything Model (SAM), lets you refine it with a brush, then saves a PNG mask to
`data/<dataset>/labels` using the same basename (for example `0004.png`).

## Install

```powershell
python -m pip install -r requirements.txt
python -m pip install git+https://github.com/facebookresearch/segment-anything.git
```

Download an official SAM checkpoint (for example `sam_vit_b_01ec64.pth`) from
the [Segment Anything repository](https://github.com/facebookresearch/segment-anything#model-checkpoints).

## Run

```powershell
python app.py --checkpoint ..\\..\\models\\sam\\sam_vit_b_01ec64.pth --model-type vit_b
```

Choose `data/square_01` or `data/square_02` as the dataset folder, then load
SAM. Use the automatic-mask button for a first result, or use prompts for more
control:

- **Positive scribble (+)**: draw over fabric to tell SAM what to keep.
- **Negative scribble (-)**: draw over background/other objects to exclude.
- **Box prompt**: drag a box around the garment/fabric.
- **Run prompts**: creates three SAM candidates; choose candidate 1, 2, or 3.
- **Draw mask** / **Erase mask**: manually add or remove pixels. Right mouse
  also erases while in Draw-mask mode.
- **Keep largest** and **Fill holes**: fast binary-mask cleanup operations.
- `Ctrl+Z` / `Ctrl+Shift+Z`: undo / redo mask edits.
- Mouse wheel: change brush size.
- `Ctrl+S`: save the mask.

Saved masks are single-channel PNG files: fabric is `255`, background is `0`,
and dimensions exactly match the source image. The app also supports manual
annotation without SAM installed.
