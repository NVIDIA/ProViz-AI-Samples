# ONNXRuntime CPP Samples

To run the samples simply run `cmake -Bbuild . && cmake --build build` inside this repository root.
Afterwards the application can be executed from this directory using `./build/cuda_provider/cuda_sample` without any command line arguments. 
For checking out TensorRT EP please use the `--trt` flag and also try the `--fp16` flag to leverage tensor cores to the best!
For respective command line parameters please refer to the help menu reachable via `-h`.

An accompanying blogpost for this sample can be found in the 4. blogpost of our end-to-end AI deployment series [here](https://developer.nvidia.com/blog/end-to-end-ai-for-workstation-an-introduction-to-optimization/).
For documentation of the code please read inline comments. 
Especially in the `main()` in [cuda_sample.cpp](./cuda_provider/src/cuda_sample.cpp) the core ideas on the processing pipeline are documented.


**NOTE:** It is important to note that asynchronous processing is only supported for CUDA since the 1.14 release. 
To check the difference with and without please uncomment the respective API function [here](./cuda_provider/src/NVIDIAInference.cpp) line 76.
The effect can be best seen by attaching Nsight Systems to the App.

## High level AI pipeline

The idea behind the sample is to achieve a fully utilized GPU with only one CPU thread feeding it data.
By maintaining 2 input and 2 output buffers, we are able to upload data to input buffer 1 on an CUDA stream (`upload_stream`). 
```cpp
CheckCUDA(cudaMemcpyAsync(raw_uint, img_data[0].raw_data, image_bytes,
                          cudaMemcpyHostToDevice, upload_stream));
CheckCUDA(cudaEventRecord(upload_complete_event, upload_stream));
```
We are able to let the compute stream, on which the AI + pre-/post-processing runs wait for the upload to be finished asynchronous to the GPU using:
```cpp
CheckCUDA(cudaStreamWaitEvent(compute_stream, upload_complete_event));
```
After this asynchronous synchronization between the 2 streams the submitted CUDA work to run the preprocessing and network will be run. 
Our pre-processing operation reads the input buffer 1, but writes to input buffer 2, so that we can overwrite input buffer 1 again. 
This is captured by an CUDA event.
The same is done for post-processing - which has to wait for a read output buffer to not overwrite the previous network output.
See `NVInference::runAsync(..)` for the full implementation details, below is a simplified version of it. 
```cpp
CheckCUDA(cudaStreamWaitEvent(compute_stream, upload_complete_event));
m_preprocessing->runAsync(input, run_options);
/// This event guarantees that we can overwrite the input buffer
CheckCUDA(cudaEventRecord(m_ev_inference_in, m_compute_stream));

m_session->Run(*run_options, io_binding);

/// we wait for the signal that the output buffer has been read
CheckCUDA(cudaStreamWaitEvent(m_compute_stream, *m_download_complete_event));
m_postprocessing->runAsync(m_postprocessing->fetchInputTensorAsync(), run_options)
CheckCUDA(cudaEventRecord(m_ev_postprocess, m_compute_stream));
```
When downloading the buffer from the GPU we wait for `m_ev_postprocess` to ensure that the output is ready and afterwards record `m_download_complete_event`.
As seen above `m_download_complete_event` is the event to ensure no output is overwritten.
Also note that for all memory operations we are using `cudaMemcpyAsync` in combination with pinned memory (`cudaMallocHost`).
This ensures less CPU utilization due to not having to copy memory to a [pinned buffer](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/) by the driver as is done implicitly for pageable memory.

An illustration of the processing can be seen below:
![](./resources/pipelined.png)
