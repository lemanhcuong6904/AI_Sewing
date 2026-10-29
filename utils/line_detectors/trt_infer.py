import numpy as np
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    trt = None
    cuda = None

class TRTInfer:
    def __init__(self, engine_path, max_batch_size=5):
        if trt is None or cuda is None:
            raise ImportError("tensorrt or pycuda is not installed")
            
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        self.max_batch_size = max_batch_size
        self.d_input = None
        self.d_outputs = []
        self.output_shapes = []
        
        # Support TRT 10+ API and older
        self.bindings = []
        self.input_name = None
        self.output_names = []
        
        # Support TRT 10+ API where num_bindings is removed
        if hasattr(self.engine, 'num_io_tensors'):
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(name)
                
                volume = 1
                for dim in shape:
                    if dim > 0:
                        volume *= dim
                
                max_size = self.max_batch_size * volume * np.dtype(np.float32).itemsize
                device_mem = cuda.mem_alloc(max_size)
                self.bindings.append(int(device_mem))
                
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.d_input = device_mem
                    self.input_name = name
                else:
                    self.d_outputs.append(device_mem)
                    self.output_names.append(name)
                    self.output_shapes.append(tuple(list(shape)[1:]))
        else:
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                shape = self.engine.get_binding_shape(i)
                
                volume = 1
                for dim in shape:
                    if dim > 0:
                        volume *= dim
                
                max_size = self.max_batch_size * volume * np.dtype(np.float32).itemsize
                device_mem = cuda.mem_alloc(max_size)
                self.bindings.append(int(device_mem))
                
                if self.engine.binding_is_input(i):
                    self.d_input = device_mem
                    self.input_name = name
                else:
                    self.d_outputs.append(device_mem)
                    self.output_names.append(name)
                    # Keep track of original shape without batch
                    self.output_shapes.append(tuple(list(shape)[1:]))
                
    def __call__(self, input_data):
        """
        input_data: numpy array of shape (B, C, H, W)
        """
        B = input_data.shape[0]
        if B > self.max_batch_size:
            raise ValueError(f"Input batch size {B} exceeds max_batch_size {self.max_batch_size}")
            
        # Set dynamic shape if supported
        if hasattr(self.context, 'set_input_shape') and self.input_name is not None:
            self.context.set_input_shape(self.input_name, input_data.shape)
            self.context.set_tensor_address(self.input_name, int(self.d_input))
            for name, d_out in zip(self.output_names, self.d_outputs):
                self.context.set_tensor_address(name, int(d_out))
        else:
            self.context.set_binding_shape(0, input_data.shape)
            
        # Transfer input data to device
        cuda.memcpy_htod_async(self.d_input, np.ascontiguousarray(input_data), self.stream)
        
        # Run inference
        if hasattr(self.context, 'execute_async_v3'):
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Transfer predictions back from device
        outputs = []
        for i, d_out in enumerate(self.d_outputs):
            out_shape = (B,) + self.output_shapes[i]
            h_out = np.empty(out_shape, dtype=np.float32)
            cuda.memcpy_dtoh_async(h_out, d_out, self.stream)
            outputs.append(h_out)
            
        # Synchronize the stream
        self.stream.synchronize()
        
        return outputs

