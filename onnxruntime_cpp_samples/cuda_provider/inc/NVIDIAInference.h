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

#pragma once

#include "ProcessingTemplate.h"
#include "cuda_helpers.cuh"
#include <map>
#include <memory>
#include <optional>

class NVInference : public ProcessingTemplate {
public:
  NVInference(std::unique_ptr<Ort::SessionOptions> opts)
      : ProcessingTemplate(), m_session_options(std::move(opts)) {
    CheckCUDA(cudaEventCreate(&m_ev_inference_in));
    CheckCUDA(cudaEventCreate(&m_ev_inference_out));
    CheckCUDA(cudaEventCreate(&m_ev_inference_out));
    CheckCUDA(cudaEventCreate(&m_ev_postprocess));
    CheckCUDA(cudaEventCreate(&m_ev_preprocess));
  }

  virtual cudaEvent_t getConsumedEvent();

  virtual void setDownloadedEvent(cudaEvent_t *ev);

  virtual void allocateOutputTensor(std::vector<int64_t> shape,
                                    ONNXTensorElementDataType type,
                                    Ort::Allocator *allocator = nullptr);

  virtual void allocateInputTensor(std::vector<int64_t> shape,
                                   ONNXTensorElementDataType type,
                                   Ort::Allocator *allocator = nullptr);

  virtual cudaStream_t fetchComputeStream() const { return m_compute_stream; }

  virtual void runAsync(Ort::Value *input, Ort::RunOptions *run_options);

  std::map<std::string, float> getTimings();

  virtual Ort::Value *fetchOutputTensorAsync() const {
    if (m_postprocessing) {
      return m_postprocessing->fetchOutputTensorAsync();
    } else {
      return m_output_tensor.get();
    }
  }

  //  virtual Ort::Value *fetchInputTensorAsync() const {
  //    if (m_preprocessing) {
  //      return m_preprocessing->fetchOutputTensorAsync();
  //    } else {
  //      return m_output_tensor.get();
  //    }
  //  }

  ~NVInference() {
    CheckCUDA(cudaEventSynchronize(m_ev_inference_in));
    CheckCUDA(cudaEventDestroy(m_ev_inference_in));
    CheckCUDA(cudaEventSynchronize(m_ev_inference_out));
    CheckCUDA(cudaEventDestroy(m_ev_inference_out));
    CheckCUDA(cudaEventSynchronize(m_ev_postprocess));
    CheckCUDA(cudaEventDestroy(m_ev_postprocess));
    CheckCUDA(cudaEventSynchronize(m_ev_preprocess));
    CheckCUDA(cudaEventDestroy(m_ev_preprocess));
    CheckCUDA(cudaStreamSynchronize(m_compute_stream));
  }

protected:
  Ort::AllocatorWithDefaultOptions m_default_allocator =
      Ort::AllocatorWithDefaultOptions();
  ProcessingTemplate *m_preprocessing = nullptr, *m_postprocessing = nullptr;
  ONNXTensorElementDataType m_model_dtype =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  std::unique_ptr<Ort::Env> m_env;
  std::unique_ptr<Ort::SessionOptions> m_session_options;
  std::unique_ptr<Ort::Allocator> m_device_allocator;
  std::unique_ptr<Ort::Session> m_session;
  cudaStream_t m_compute_stream;
  cudaEvent_t m_ev_preprocess, m_ev_postprocess, m_ev_inference_in,
      m_ev_inference_out;
  cudaEvent_t *m_download_complete_event;
};
