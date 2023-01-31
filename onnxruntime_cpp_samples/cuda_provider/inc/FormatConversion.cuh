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
#pragma warning(disable : 4661)

#include "ProcessingTemplate.h"
#include "cuda_fp16.h"
#include <type_traits>

template <typename DTYPE_IN, typename DTYPE_OUT>
class NCHW_to_NHWC : public ProcessingTemplate {
public:
  NCHW_to_NHWC() : ProcessingTemplate(){};

  void runAsync(Ort::Value *input,
                Ort::RunOptions *run_options = nullptr) override;
};

template <typename DTYPE_IN, typename DTYPE_OUT>
class NHWC_to_NCHW : public ProcessingTemplate {
public:
  NHWC_to_NCHW() : ProcessingTemplate(){};

  void runAsync(Ort::Value *input,
                Ort::RunOptions *run_options = nullptr) override;
};

template class NCHW_to_NHWC<uint8_t, float>;

template class NHWC_to_NCHW<float, uint8_t>;

template class NCHW_to_NHWC<uint8_t, half>;

template class NHWC_to_NCHW<half, uint8_t>;
