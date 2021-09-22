# TensorRT ONNX Parser Sample

This sample illustrates how to use the TensorRT C++ ONNX Parser API to populate an INetworkDefinition from an input ONNX model file.
Afterwards, an IBuilder instance is used to generate an optimized TensorRT inference engine, ready to be serialized and written to disk.

## Setup

### Prerequisites

This sample requires the following packages.

**TensorRT GA build**
* [TensorRT](https://developer.nvidia.com/nvidia-tensorrt-download) Tested with v7.2.3.4

**System Packages**
* [CUDA](https://developer.nvidia.com/cuda-toolkit)
  * Recommended versions:
  * cuda-11.2 + cuDNN-8.1
* [cmake](https://github.com/Kitware/CMake/releases) >= v3.17
* [python](<https://www.python.org/downloads/>) >= v3.6.5
* [pip](https://pypi.org/project/pip/#history) >= v19.0
* Essential utilities
  * [git](https://git-scm.com/downloads)

**Python Packages**

```
pip install tensorflow tensorflow-hub tf2onnx
```

**ONNX Model File**

For this sample, we use a TensorFlow ResNet-50 model that we fetch from Tensorhub and convert it to ONNX using the tf2onnx python utility.
To download the Tensorflow model from https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5 run:

**Please review the licensing terms of the [downloaded model (Apache-2.0)](https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5) before
running the script!**

```console
cd ../Models
python fetch_model.py
```

[tf2onnx](https://github.com/onnx/tensorflow-onnx) can be used to convert the model to ONNX:
```console
python -m tf2onnx.convert --saved-model resnet50 --output ONNX/resnet50.onnx
```

The converted ONNX model will be saved at TensorRT/Models/Onnx/resnet50.onnx

### Building and Running the Sample on Windows or Linux

Configure the project using CMake:
```console
cd ../TRTOnnxParserSample/

mkdir build && cd build
cmake ..
```

Either open and build the generated project in your preferred IDE, e.g. Visual Studio, or use cmake:
```console
cmake --build . --config Release
```

To run the sample cd to the directory containing the built executable and call

```console
TRTOnnxParserSample.exe path/to/input/onnx_file.onnx path/to/output/engine.plan [--fp16]
# or on Linux
./TRTOnnxParserSample path/to/input/onnx_file.onnx path/to/output/engine.plan [--fp16]
```
The `--fp16` option will enable FP16 mixed precision inference.



This will create a TensorRT engine optimized for the GPU installed and write the serialized engine to `path/to/output/engine.plan`.

