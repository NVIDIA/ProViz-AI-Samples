# TensorRT Plugin Sample

This sample illustrates how to use the ONNX GraphSurgeon python utility to import and manipulate an ONNX model by replacing all "BatchNormalization" layers by custom "MyPlugin" layers.
To parse the manipulated ONNX file we need to implement a custom "MyPlugin" layer in TensorRT, which we wrap in a plugin library.
After linking against that plugin library and registering the libraries custom layers to the TensorRT plugin registry, we use the TensorRT ONNX Parser to import the manipulated ONNX file into TensorRT.
Furthermore, we show how to load and deserialize the generated optimized engine and run inference using the TensorRT C++ runtime.

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
# This will install Nvidia's PyPI for the onnx-graphsurgeon library
pip install nvidia-pyindex
pip install onnx-graphsurgeon onnx
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

### Building and Running the Sample on Windows or Linux

Configure the project using CMake:
```console
mkdir build && cd build
cmake ..
```

Either open and build the generated project in your preferred IDE, e.g. Visual Studio, or use cmake:
```console
cmake --build . --config Release
```

We replace all "BatchNormalization" nodes in an input ONNX model by calling the `replace_bn.py` python script which uses the ONNX GraphSurgeon utility:
```console
cd ..
python replace_bn.py ..\Models\ONNX\resnet50.onnx ..\Models\ONNX\resnet_plugin.onnx
```

To generate a TensorRT engine cd to the directory containing the built executables and call

```console
TRTPluginImportOnnx.exe path/to/plugin/onnx_file.onnx path/to/output/engine.plan [--fp16]
# or on Linux: ./TRTPluginImportOnnx path/to/plugin/onnx_file.onnx path/to/output/engine.plan [--fp16]
```
The `--fp16` option will enable FP16 mixed precision inference.
This will create a TensorRT engine optimized for the GPU installed and write the serialized engine to `path/to/output/engine.plan`.

After the engine has been generated we can use it to run inference with the TensorRT C++ runtime:
```console
TRTPluginInfer.exe path/to/input/engine.plan batch_size
# or on Linux: ./TRTPluginInfer path/to/input/engine.plan batch_size
```

The `batch_size` parameter specifies the batch size to run inference with and must be in the range [1,64].

