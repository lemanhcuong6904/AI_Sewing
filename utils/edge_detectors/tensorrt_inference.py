import numpy as np
import cv2 as cv
import os
from concurrent.futures import ThreadPoolExecutor

try:
    import tensorrt as trt
    import pycuda.driver as cuda
except ImportError as e:
    trt = None
    cuda = None
    _import_error = e

_dexined_session = None

class TrtDexiNed:
    def __init__(self, engine_path):
        if trt is None or cuda is None:
            raise ImportError(f"tensorrt or pycuda is not installed. Error: {_import_error}")
            
        cuda.init()
        self.device = cuda.Device(0)
        self.cuda_ctx = self.device.make_context()
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")
            
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # Max dimensions from trtexec command
        max_batch = 20
        max_c = 3
        max_h = 400
        max_w = 640
        
        max_input_size = max_batch * max_c * max_h * max_w * np.dtype(np.float32).itemsize
        self.d_input = cuda.mem_alloc(max_input_size)
        
        self.d_outputs = []
        self.output_names = []
        
        # Support TRT 10+ API where num_bindings is removed
        if hasattr(self.engine, 'num_io_tensors'):
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.input_name = name
                else:
                    self.output_names.append(name)
                    max_output_size = max_batch * 1 * max_h * max_w * np.dtype(np.float32).itemsize
                    d_out = cuda.mem_alloc(max_output_size)
                    self.d_outputs.append(d_out)
        else:
            # Fallback for older TRT versions
            self.bindings = []
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                if self.engine.binding_is_input(i):
                    self.bindings.append(int(self.d_input))
                    self.input_name = name
                else:
                    self.output_names.append(name)
                    max_output_size = max_batch * 1 * max_h * max_w * np.dtype(np.float32).itemsize
                    d_out = cuda.mem_alloc(max_output_size)
                    self.d_outputs.append(d_out)
                    self.bindings.append(int(d_out))
                    
        # DexiNed typically has 7 outputs; the final fused map is the last one.
        self.final_output_idx = len(self.d_outputs) - 1
        self.cuda_ctx.pop()



    def run(self, batch_tensor):
        self.cuda_ctx.push()
        try:
            B, C, H, W = batch_tensor.shape
            
            if hasattr(self.context, 'set_input_shape'):
                self.context.set_input_shape(self.input_name, (B, C, H, W))
                self.context.set_tensor_address(self.input_name, int(self.d_input))
                for name, d_out in zip(self.output_names, self.d_outputs):
                    self.context.set_tensor_address(name, int(d_out))
            else:
                self.context.set_binding_shape(0, (B, C, H, W))
            
            # Copy input to device
            cuda.memcpy_htod_async(self.d_input, np.ascontiguousarray(batch_tensor), self.stream)
            
            # Execute
            if hasattr(self.context, 'execute_async_v3'):
                self.context.execute_async_v3(stream_handle=self.stream.handle)
            else:
                self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            
            # Retrieve final output
            h_output = np.empty((B, 1, H, W), dtype=np.float32)
            d_final_out = self.d_outputs[self.final_output_idx]
            cuda.memcpy_dtoh_async(h_output, d_final_out, self.stream)
            
            self.stream.synchronize()
            return h_output
        finally:
            self.cuda_ctx.pop()


def load_dexined(weight_path=None, device="cuda"):
    """
    Load the DexiNed TensorRT engine.
    """
    global _dexined_session
    if _dexined_session is None:
        if weight_path is None:
            weight_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "dexined", "dexined_fp16.engine")
            
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"[tensorrt_inference] Error: DexiNed Engine not found at {weight_path}. Please build it first.")

        try:
            _dexined_session = TrtDexiNed(weight_path)
            print(f"[tensorrt_inference] TensorRT Engine loaded successfully from {weight_path}.")
        except Exception as e:
            raise RuntimeError(f"[tensorrt_inference] Failed to load TensorRT Engine. Error: {e}")

    return _dexined_session


def infer_dexined(model, img, device="cuda"):
    """
    Perform inference using DexiNed via TensorRT.
    img: numpy array, BGR or Grayscale, shape (H, W, C) or (H, W)
    Returns: numpy array of shape (H, W), dtype uint8 (0-255)
    """
    if model is None:
        raise ValueError("[tensorrt_inference] Error: Model is not initialized.")

    if len(img.shape) == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
        
    orig_h, orig_w = img.shape[:2]
    # Use fixed resolution matching the compiled TensorRT Engine
    new_h = 400
    new_w = 640
    
    if new_h != orig_h or new_w != orig_w:
        img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
        
    # Preprocess
    img_float = img.astype(np.float32)
    tensor = np.transpose(img_float, (2, 0, 1)) # CHW
    tensor = np.expand_dims(tensor, axis=0) # NCHW
    
    # Inference
    out_tensor = model.run(tensor) # Shape: (1, 1, H, W)
    
    # Postprocess
    edge_map = np.squeeze(out_tensor) # Shape: (H, W)
    edge_map_sigmoid = 1.0 / (1.0 + np.exp(-edge_map))
    
    # Scale to 0-255
    edge_map_uint8 = np.clip(edge_map_sigmoid * 255.0, 0, 255).astype(np.uint8)
    
    if new_h != orig_h or new_w != orig_w:
        edge_map_uint8 = cv.resize(edge_map_uint8, (orig_w, orig_h), interpolation=cv.INTER_LINEAR)
        
    return edge_map_uint8


def infer_dexined_batch(model, images, device="cuda"):
    """
    Perform inference using DexiNed on a batch of images via TensorRT.
    images: list of numpy arrays
    Returns: list of numpy arrays (edge masks)
    """
    if model is None:
        raise ValueError("[tensorrt_inference] Error: Model is not initialized.")
    if not images:
        return []

    def preprocess_single(img):
        if len(img.shape) == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
            
        orig_h, orig_w = img.shape[:2]
        # Use fixed resolution matching the compiled TensorRT Engine
        new_h = 400
        new_w = 640
        
        if new_h != orig_h or new_w != orig_w:
            img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
            
        img_float = img.astype(np.float32)
        tensor = np.transpose(img_float, (2, 0, 1)) # CHW
        return tensor, (orig_h, orig_w)
        
    with ThreadPoolExecutor() as executor:
        prep_results = list(executor.map(preprocess_single, images))
        
    processed_tensors = [r[0] for r in prep_results]
    orig_dims = [r[1] for r in prep_results]
        
    batch_tensor = np.stack(processed_tensors, axis=0)
    
    # Inference
    out_tensor = model.run(batch_tensor) # Shape: (B, 1, H, W)
    edge_maps = np.squeeze(out_tensor, axis=1) # Shape: (B, H, W)
    
    # Vectorized Sigmoid
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
