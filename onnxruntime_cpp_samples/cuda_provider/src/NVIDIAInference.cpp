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

#include "NVIDIAInference.h"

cudaEvent_t NVInference::getConsumedEvent() {
  if (m_preprocessing)
    return m_ev_inference_in;
  else
    return m_ev_inference_out;
}

void NVInference::setDownloadedEvent(cudaEvent_t *ev) {
  m_download_complete_event = ev;
}

void NVInference::allocateOutputTensor(
    std::vector<int64_t> shape, ONNXTensorElementDataType bytes_per_value,
    Ort::Allocator *allocator) {
  if (!allocator) {
    allocator = m_device_allocator.get();
  }
  if (m_postprocessing) {
    m_postprocessing->allocateOutputTensor(shape, bytes_per_value, allocator);
  } else {
    ProcessingTemplate::allocateOutputTensor(shape, bytes_per_value, allocator);
  }
}

void NVInference::allocateInputTensor(std::vector<int64_t> shape,
                                      ONNXTensorElementDataType bytes_per_value,
                                      Ort::Allocator *allocator) {
  if (!allocator) {
    allocator = m_device_allocator.get();
  }
  ProcessingTemplate::allocateInputTensor(shape, bytes_per_value, allocator);
}

void NVInference::runAsync(Ort::Value *input, Ort::RunOptions *run_options) {
  Ort::IoBinding io_binding(*m_session);
  auto name = m_session->GetInputNameAllocated(0, m_default_allocator);
  auto name_out = m_session->GetOutputNameAllocated(0, m_default_allocator);
  if (m_postprocessing) {
    io_binding.BindOutput(name_out.get(),
                          *m_postprocessing->fetchInputTensorAsync());
  } else {
    io_binding.BindOutput(name_out.get(), *m_output_tensor);
  }
  CheckCUDA(cudaEventRecord(m_ev_preprocess, m_compute_stream));
  if (m_preprocessing != nullptr) {
    io_binding.BindInput(name.get(),
                         *m_preprocessing->fetchOutputTensorAsync());
    m_preprocessing->runAsync(input, run_options);
    /// This event guarantees that we can overwrite the input buffer
    CheckCUDA(cudaEventRecord(m_ev_inference_in, m_compute_stream));
  } else {
    io_binding.BindOutput(name.get(), *m_input_tensor);
  }


  run_options->AddConfigEntry("disable_synchronize_execution_providers", "1");
  m_session->Run(*run_options, io_binding);
  if (!m_preprocessing) {
    /// This event guarantees that we can overwrite the input buffer
    CheckCUDA(cudaEventRecord(m_ev_inference_in, m_compute_stream));
  }
  CheckCUDA(cudaEventRecord(m_ev_inference_out, m_compute_stream));

  /// we wait for the signal that the output buffer has been read
  CheckCUDA(cudaStreamWaitEvent(m_compute_stream, *m_download_complete_event));
  if (m_postprocessing != nullptr) {
    m_postprocessing->runAsync(m_postprocessing->fetchInputTensorAsync(),
                               run_options);
  }
  CheckCUDA(cudaEventRecord(m_ev_postprocess, m_compute_stream));
}

std::map<std::string, float> NVInference::getTimings() {
  std::map<std::string, float> results;
  CheckCUDA(cudaEventSynchronize(m_ev_preprocess));
  CheckCUDA(cudaEventSynchronize(m_ev_inference_in));
  CheckCUDA(cudaEventSynchronize(m_ev_inference_out));
  CheckCUDA(cudaEventSynchronize(m_ev_postprocess));
  float elapsed_time = 0;
  if (m_preprocessing) {
    CheckCUDA(cudaEventElapsedTime(&elapsed_time, m_ev_preprocess,
                                   m_ev_inference_in));
    results.insert({"preprocess", elapsed_time});
  }
  {
    CheckCUDA(cudaEventElapsedTime(&elapsed_time, m_ev_inference_in,
                                   m_ev_inference_out));
    results.insert({"inference", elapsed_time});
  }
  if (m_postprocessing) {
    CheckCUDA(cudaEventElapsedTime(&elapsed_time, m_ev_inference_out,
                                   m_ev_postprocess));
    results.insert({"postprocess", elapsed_time});
  }
  return results;
}
