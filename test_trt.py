import sys
import numpy as np

print("Loading TRT module...")
from utils.edge_detectors.tensorrt_inference import TrtDexiNed

print("Instantiating engine...")
engine_path = "models/dexined/dexined_fp16.engine"
try:
    model = TrtDexiNed(engine_path)
    print("Engine instantiated successfully!")
except Exception as e:
    print(f"Failed to instantiate engine: {e}")
    sys.exit(1)

print("Testing inference...")
dummy_input = np.random.randn(1, 3, 400, 640).astype(np.float32)
try:
    out = model.run(dummy_input)
    print(f"Inference successful! Output shape: {out.shape}")
except Exception as e:
    print(f"Failed during inference: {e}")

print("Done. Exiting.")
