# TensorRT Network Definition API Sample

This sample illustrates how to use the TensorRT C++ Network Definition API to populate an INetworkDefinition with a single convolutional layer.
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
* Essential utilities
  * [git](https://git-scm.com/downloads)

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

To run the sample cd to the directory containing the built executable and call

```bash
TRTNetworkDefinitionSample.exe path/to/save/engine.plan
# or on Linux:
./TRTNetworkDefinitionSample path/to/save/engine.plan
```

This will create a TensorRT engine optimized for the GPU installed and write the serialized engine to `path/to/engine.plan`.
