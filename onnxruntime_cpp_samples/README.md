# ONNXRuntime CPP Samples

To run the samples simply run `cmake -Bbuild . && cmake --build build` inside this folder.
Afterwards the applications can be executed from this directory using `./build/cuda_provider/cuda_sample` and `./build/dml_provider/dml_sample` without any command line arguments. 
For any further information please consult the README files in the respective subdirectories.


## Dependencies 

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

Verify which version of onnxruntime is compatible with that CUDA version at [NVIDIA - CUDA | onnxruntime website](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).Then update the `onnxruntime_INSTALL` inside the `CMakeLists.txt` or install the CUDA Toolkit that is compatible with the onnxruntime version that the sample is using.

- [CuDNN](https://developer.nvidia.com/cudnn-downloads)
Make sure that the CuDNN version is compatible with the CUDA version that is installed. Insure that the CuDNN `bin`, `lib`, and `include` directories are in the system variable CUDA_PATH and CUDNN.


