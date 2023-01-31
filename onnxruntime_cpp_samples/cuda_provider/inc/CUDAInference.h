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

#include "NVIDIAInference.h"
#include "ProcessingTemplate.h"
#include "cuda_helpers.cuh"
#include "onnx_helpers.h"

class CUDAInference : public NVInference {
public:
  CUDAInference(std::string model_path, std::vector<int64_t> &input_shape,
                std::vector<int64_t> &output_shape,
                ProcessingTemplate *preprocessing_module,
                ProcessingTemplate *postprocessing_module,
                uint32_t device_id = 0, bool use_exhaustive = true,
                cudaStream_t stream = nullptr,
                std::unique_ptr<Ort::SessionOptions> opts =
                    std::unique_ptr<Ort::SessionOptions>());

  void runAsync(Ort::Value *input, Ort::RunOptions *run_options) override;

private:
  std::unique_ptr<OrtCUDAProviderOptions> m_cuda_options;
};
