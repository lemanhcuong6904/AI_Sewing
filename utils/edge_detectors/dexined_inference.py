import numpy as np
import cv2 as cv
import os

_ort_import_error = None
try:
    import onnxruntime as ort
except ImportError as e:
    ort = None
    _ort_import_error = e

_dexined_session = None

def load_dexined(weight_path=None, device="cpu"):
    """
    Load the DexiNed ONNX model using TensorRT execution provider.
    """
    global _dexined_session
    if _dexined_session is None:
        if ort is None:
            err_msg = "[dexined_inference] Error: onnxruntime is not installed or failed to import."
            if _ort_import_error is not None:
                err_msg += f" Details: {_ort_import_error}"
            raise ImportError(err_msg)

        if weight_path is None:
            weight_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "dexined", "dexined.onnx")
            
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"[dexined_inference] Error: DexiNed ONNX model not found at {weight_path}.")

        try:
            # Try loading with TensorRT and CUDA providers
            _dexined_session = ort.InferenceSession(weight_path, providers=[
                ('TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': './models/dexined',
                    'trt_fp16_enable': True
                }),
                'CUDAExecutionProvider',
                'CPUExecutionProvider'
            ])
            print(f"[dexined_inference] ONNX model loaded successfully from {weight_path}.")
        except Exception as e:
            print(f"[dexined_inference] Warning: Failed to load ONNX model with GPU/TensorRT providers. Error: {e}")
            print("[dexined_inference] Falling back to CPUExecutionProvider...")
            try:
                _dexined_session = ort.InferenceSession(weight_path, providers=['CPUExecutionProvider'])
                print(f"[dexined_inference] ONNX model loaded successfully on CPU from {weight_path}.")
            except Exception as cpu_e:
                raise RuntimeError(f"Failed to load ONNX model on CPU fallback. Error: {cpu_e}") from e

    return _dexined_session

def infer_dexined(model, img, device="cpu"):
    """
    Perform inference using DexiNed to get the edge mask via ONNXRuntime.
    img: numpy array, BGR or Grayscale, shape (H, W, C) or (H, W)
    Returns: numpy array of shape (H, W), dtype uint8 (0-255)
    """
    if model is None:
        raise ValueError("[dexined_inference] Error: Model is not initialized.")

    if len(img.shape) == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
        
    orig_h, orig_w = img.shape[:2]
    # Ensure dimensions are divisible by 16 for DexiNed's ONNX graph
    new_h = ((orig_h + 15) // 16) * 16
    new_w = ((orig_w + 15) // 16) * 16
    
    if new_h != orig_h or new_w != orig_w:
        img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
        
    # Preprocess
    img_float = img.astype(np.float32)
    # The ONNX model exported earlier takes BCHW input
    tensor = np.transpose(img_float, (2, 0, 1)) # CHW
    tensor = np.expand_dims(tensor, axis=0) # NCHW
    
    # Inference
    outputs = model.run(None, {model.get_inputs()[0].name: tensor})
    
    # The final fused edge map is typically the last element (index -1)
    edge_map = outputs[-1]
    
    # Postprocess (Apply sigmoid and convert to uint8)
    edge_map = np.squeeze(edge_map)
    edge_map_sigmoid = 1.0 / (1.0 + np.exp(-edge_map))
    
    # Scale to 0-255
    edge_map_uint8 = np.clip(edge_map_sigmoid * 255.0, 0, 255).astype(np.uint8)
    
    if new_h != orig_h or new_w != orig_w:
        edge_map_uint8 = cv.resize(edge_map_uint8, (orig_w, orig_h), interpolation=cv.INTER_LINEAR)
        
    return edge_map_uint8


from concurrent.futures import ThreadPoolExecutor

def infer_dexined_batch(model, images, device="cpu"):
    """
    Perform inference using DexiNed on a batch of images.
    images: list of numpy arrays
    Returns: list of numpy arrays (edge masks)
    """
    if model is None:
        raise ValueError("[dexined_inference] Error: Model is not initialized.")
    if not images:
        return []

    def preprocess_single(img):
        if len(img.shape) == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
            
        orig_h, orig_w = img.shape[:2]
        
        # Ensure dimensions are divisible by 16
        new_h = ((orig_h + 15) // 16) * 16
        new_w = ((orig_w + 15) // 16) * 16
        
        if new_h != orig_h or new_w != orig_w:
            img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
            
        img_float = img.astype(np.float32)
        tensor = np.transpose(img_float, (2, 0, 1)) # CHW
        return tensor, (orig_h, orig_w)
        
    with ThreadPoolExecutor() as executor:
        prep_results = list(executor.map(preprocess_single, images))
        
    processed_tensors = [r[0] for r in prep_results]
    orig_dims = [r[1] for r in prep_results]
        
    # Stack tensors to create batch (B, C, H, W)
    batch_tensor = np.stack(processed_tensors, axis=0)
    
    # Inference
    outputs = model.run(None, {model.get_inputs()[0].name: batch_tensor})
    
    # The final fused edge map is typically the last element (index -1)
    edge_maps = outputs[-1] # Shape: (B, 1, H, W)
    edge_maps = np.squeeze(edge_maps, axis=1) # Shape: (B, H, W)
    
    # Vectorized Sigmoid and uint8 conversion for the entire batch
    edge_maps_sigmoid = 1.0 / (1.0 + np.exp(-edge_maps))
    edge_maps_uint8 = np.clip(edge_maps_sigmoid * 255.0, 0, 255).astype(np.uint8)
    
    def postprocess_single(i):
        edge_map = edge_maps_uint8[i]
        orig_h, orig_w = orig_dims[i]
        new_h, new_w = edge_map.shape[:2]
        if new_h != orig_h or new_w != orig_w:
            edge_map = cv.resize(edge_map, (orig_w, orig_h), interpolation=cv.INTER_LINEAR)
        return edge_map
        
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(postprocess_single, range(len(images))))
        
    return results
