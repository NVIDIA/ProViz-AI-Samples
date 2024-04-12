# ONNXRuntime CPP Samples

To run the samples simply run `cmake -Bbuild . && cmake --build build` inside this folder.
Afterwards the applications can be executed from this directory using `./build/cuda_provider/cuda_sample` and `./build/dml_provider/dml_sample` without any command line arguments. 
For any further information please consult the README files in the respective subdirectories.


## Dependencies 

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

If you want to use another onnxruntime version than the automatic downloaded one please point the CMake variable `onnxruntime_INSTALL` to the root folder of the downloaded package. Or install CUDA Toolkit 11.6, which is compatible with the downloaded ONNX runtime 1.14. At least the major CUDA version has to match.

To check additional onnxruntime versions and their respective CUDA-compatible version reference to the [onnxruntime CUDA Execution Provider documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements). 

- [CuDNN](https://developer.nvidia.com/cudnn-downloads)
Make sure that the CuDNN version is compatible with the CUDA version and at least it's major version is in line with [ONNX Runtime requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements). Ensure that the `bin` folder of cuDNN is on the `PATH` system variable or place all shared object (`*.dll/*.so`) in the same folder as the sample executable. 


