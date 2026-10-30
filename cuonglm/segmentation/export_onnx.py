"""Export the trained fabric UNet (best.pt) to ONNX, to run with onnxruntime on Ubuntu.

Run (needs torch; the deployment machine only needs onnxruntime):
    python cuonglm/segmentation/export_onnx.py

The per-image / per-channel z-score normalisation of the dataset and the final sigmoid are baked into
the graph, so the ONNX model takes the raw image and returns the fabric probability:
    input  "image": float32 (1, 3, H, W)  RGB, values 0..255
    output "prob" : float32 (1, 1, H, W)  fabric probability 0..1   (mask = prob >= 0.5)
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from models.unet import UNet  # noqa: E402

# ---- config
CHECKPOINT = THIS_DIR / "experiments" / "unet_fabric_segmentation" / "best.pt"
OUT = THIS_DIR / "experiments" / "unet_fabric_segmentation" / "best.onnx"
IMAGE_SIZE = None        # (H, W) of the ONNX input; None = the size the model was trained on (480, 640)
OPSET = 17
CHECK_IMAGE = THIS_DIR / "data" / "square_01" / "images" / "0000.png"   # parity check torch vs onnxruntime


class Deploy(nn.Module):
    """raw RGB 0..255 -> z-score per image/channel (as in the dataset) -> UNet -> sigmoid."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        d = x - x.mean(dim=(2, 3), keepdim=True)
        n = x.shape[2] * x.shape[3]
        std = (d.pow(2).sum(dim=(2, 3), keepdim=True) / (n - 1)).sqrt().clamp_min(1e-6)   # unbiased, like torch.std
        return torch.sigmoid(self.net(d / std))


def main():
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = state.get("config", {})
    h, w = IMAGE_SIZE or tuple(cfg.get("IMAGE_SIZE", (480, 640)))
    separable = bool(cfg.get("SEPARABLE", True))
    print(f"[export] epoch {state.get('epoch')}, best train dice {state.get('best_train_dice')}, "
          f"separable={separable}, input {h}x{w}")

    net = UNet(in_channels=3, separable=separable)
    net.load_state_dict(state["model_state"])
    model = Deploy(net).eval()

    dummy = torch.rand(1, 3, h, w) * 255
    args = dict(input_names=["image"], output_names=["prob"], opset_version=OPSET)
    try:
        torch.onnx.export(model, dummy, str(OUT), dynamo=False, **args)   # classic exporter, no onnxscript needed
    except TypeError:                                                      # torch < 2.5 has no `dynamo` argument
        torch.onnx.export(model, dummy, str(OUT), **args)
    print(f"[export] {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")

    try:
        import onnxruntime as ort
        from PIL import Image
    except ImportError:
        print("[check] onnxruntime / pillow not installed: parity check skipped")
        return
    img = np.asarray(Image.open(CHECK_IMAGE).convert("RGB").resize((w, h)), np.float32)
    x = img.transpose(2, 0, 1)[None]
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"image": x})[0]
    print(f"[check] {CHECK_IMAGE.name}: max |torch - onnx| = {np.abs(ref - got).max():.2e}, "
          f"mask agreement = {((ref >= 0.5) == (got >= 0.5)).mean():.4%}, fabric = {(got >= 0.5).mean():.1%}")


if __name__ == "__main__":
    main()
