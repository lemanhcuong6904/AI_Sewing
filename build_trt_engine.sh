#!/bin/bash

# Navigate to the script directory
cd "$(dirname "$0")"

echo "Building TensorRT engine for DexiNed..."
echo "This will take several minutes on the Jetson Orin."

trtexec --onnx=models/dexined/dexined.onnx \
        --saveEngine=models/dexined/dexined_fp16.engine \
        --minShapes=input:1x3x400x640 \
        --optShapes=input:1x3x400x640 \
        --maxShapes=input:5x3x400x640 \
        --fp16

echo "Engine built successfully at models/dexined/dexined_fp16.engine"
