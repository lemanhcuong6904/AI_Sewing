import torch
import os
from models.dexined.dexined_model import DexiNed

def export_to_onnx():
    device = "cpu"
    weight_path = os.path.join("models", "dexined", "dexined_10_model.pth")
    onnx_path = os.path.join("models", "dexined", "dexined.onnx")

    if not os.path.exists(weight_path):
        print(f"Error: Weight file not found at {weight_path}")
        return

    print("Loading DexiNed model...")
    model = DexiNed().to(device)
    try:
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return

    model.eval()

    # The original shape for adjustor is 128x128 but we can allow dynamic axes or just fix it.
    # We will fix it to 128x128 for the adjustor pipeline speed.
    dummy_input = torch.randn(1, 3, 128, 128, device=device)

    print(f"Exporting model to {onnx_path}...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['out1', 'out2', 'out3', 'out4', 'out5', 'out6', 'fused_edge_map'],
            dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                          'fused_edge_map': {0: 'batch_size', 2: 'height', 3: 'width'}}
        )

    print(f"Successfully exported DexiNed to {onnx_path}!")

if __name__ == "__main__":
    export_to_onnx()
