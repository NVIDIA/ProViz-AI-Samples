/*###############################################################################
#
# Copyright 2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
###############################################################################*/

#include "CUDAInference.h"
#include "FormatConversion.cuh"
#include <fstream>
#include <iostream>
#include <memory>

CUDAInference::CUDAInference(std::string model_path,
                             std::vector<int64_t> &input_shape,
                             std::vector<int64_t> &output_shape,
                             ProcessingTemplate *preprocessing_module,
                             ProcessingTemplate *postprocessing_module,
                             uint32_t device_id, bool use_exhaustive,
                             cudaStream_t stream,
                             std::unique_ptr<Ort::SessionOptions> opts)
    : NVInference(std::move(opts)) {
  nvtxRangePush("CUDA EP creation");
  CheckCUDA(cudaSetDevice(device_id));
  const char *name = "cuda_inference";
  m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, name);
  if (!m_session_options) {
    m_session_options = std::make_unique<Ort::SessionOptions>();
  }
  /// setup cuda options
  {
    m_cuda_options = std::make_unique<OrtCUDAProviderOptions>();
    m_cuda_options->device_id = int(device_id);
    m_cuda_options->has_user_compute_stream = 1;
    m_cuda_options->do_copy_in_default_stream = 0;
    m_cuda_options->user_compute_stream = stream;
    if (use_exhaustive)
      m_cuda_options->cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    else
      m_cuda_options->cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
    size_t mem_free = 0, mem_tot = 0;
    cudaMemGetInfo(&mem_free, &mem_tot);
    m_cuda_options->gpu_mem_limit = mem_free;
    m_cuda_options->do_copy_in_default_stream = 1;
    if (m_cuda_options->user_compute_stream != nullptr) {
      m_compute_stream =
          static_cast<cudaStream_t>(m_cuda_options->user_compute_stream);
    } else {
      CheckCUDA(cudaStreamCreate(&m_compute_stream));
      m_cuda_options->user_compute_stream = m_compute_stream;
    }

    m_session_options->AppendExecutionProvider_CUDA(*m_cuda_options);
  }
  /// set inference shapes
  {
    // this way we allow tensor core usage for the first layer already
    // https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/
    // https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#tensor-core-shape
    constexpr int64_t minimal_tensor_core_channels = 8;
    input_shape[1] = output_shape[1] = minimal_tensor_core_channels;

    OrtApi const &ortApi = Ort::GetApi();
    auto opts = (OrtSessionOptions *)(*m_session_options);
    ortApi.AddFreeDimensionOverrideByName(opts, "input_N", input_shape[0]);
    ortApi.AddFreeDimensionOverrideByName(opts, "input_C", input_shape[1]);
    ortApi.AddFreeDimensionOverrideByName(opts, "input_H", input_shape[2]);
    ortApi.AddFreeDimensionOverrideByName(opts, "input_W", input_shape[3]);
  }
#if WIN32
  const std::wstring widestr =
      std::wstring(model_path.begin(), model_path.end());
  const ORTCHAR_T *model_path_cstr = widestr.c_str();
#else
  const ORTCHAR_T *model_path_cstr = model_path.c_str();

#endif
  m_session = std::make_unique<Ort::Session>(*m_env, model_path_cstr,
                                             *m_session_options);
  auto input_count = m_session->GetInputCount();
  std::cout << "Input count: " << input_count << std::endl;
  for (int input_idx = 0; input_idx < input_count; ++input_idx) {
    auto input_info = m_session->GetInputTypeInfo(input_idx);
    auto input_shape_info = input_info.GetTensorTypeAndShapeInfo();
    auto input_dtype = input_shape_info.GetElementType();
    if (m_model_dtype != input_dtype && input_idx > 0) {
      std::cerr << "The specified model has mixed input dtypes !" << std::endl;
    } else {
      m_model_dtype = input_dtype;
    }
    std::cout << "Input dtype: " << input_dtype << std::endl;
    std::cout << "Input shape: ";
    for (auto &s : input_shape_info.GetShape()) {
      std::cout << s << ", ";
    }
    std::cout << std::endl;
  }

  Ort::MemoryInfo memory_info_cuda("Cuda", OrtAllocatorType::OrtArenaAllocator,
                                   0, OrtMemTypeDefault);
  m_device_allocator =
      std::make_unique<Ort::Allocator>(*m_session, memory_info_cuda);

  if (postprocessing_module) {
    m_postprocessing = postprocessing_module;
    m_postprocessing->setComputeStream(m_compute_stream);
    m_postprocessing->allocateInputTensor(output_shape, m_model_dtype,
                                          m_device_allocator.get());
  }
  if (preprocessing_module) {
    m_preprocessing = preprocessing_module;
    m_preprocessing->setComputeStream(m_compute_stream);
    m_preprocessing->allocateOutputTensor(input_shape, m_model_dtype,
                                          m_device_allocator.get());
  }
  nvtxRangePop();
}

void CUDAInference::runAsync(Ort::Value *input, Ort::RunOptions *run_options) {
  nvtxRangePush("CUDA EP runAsync");
  NVInference::runAsync(input, run_options);
  nvtxRangePop();
}
