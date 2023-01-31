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

#include "FormatConversion.cuh"
#include "cuda_fp16.h"
#include "cuda_helpers.cuh"
#include <cstdint>
#include <type_traits>

template <typename INPUT_T, typename OUTPUT_T>
__global__ void conversion_kernel_NCHW_to_NHWC_impl(const INPUT_T *device_in,
                                                    OUTPUT_T *device_out,
                                                    uint32_t h, uint32_t w,
                                                    uint32_t channel,
                                                    uint32_t channel_out) {
  int idx = blockIdx.x * w + blockIdx.y * blockDim.x + threadIdx.x;
  if (idx < h * w) {
    device_in += idx * channel;
    device_out += idx;
    for (int c = 0; c < min(channel_out, channel); ++c) {
      if constexpr (std::is_same<OUTPUT_T, half>::value) {
        *device_out = __hsub(__hdiv(*device_in, half(255)), half(0.5));
      } else if constexpr (std::is_floating_point_v<INPUT_T> ||
                           std::is_same<OUTPUT_T, float>::value) {
        *device_out = *device_in / 255.f - 0.5f;
      } else {
        static_assert(std::is_same<INPUT_T, float>::value,
                      "This kernel does not support float to int conversion.");
        static_assert(std::is_same<INPUT_T, half>::value,
                      "This kernel does not support half to int conversion.");
        *device_out = *device_in;
      }
      device_in++;
      device_out += h * w;
    }
  }
}

template <typename INPUT_T, typename OUTPUT_T>
__global__ void conversion_kernel_NHWC_to_NCHW_impl(const INPUT_T *device_in,
                                                    OUTPUT_T *device_out,
                                                    uint32_t h, uint32_t w,
                                                    uint32_t channel,
                                                    uint32_t channel_out) {
  int idx = blockIdx.x * w + blockIdx.y * blockDim.x + threadIdx.x;
  if (idx < h * w) {
    device_in += idx;
    device_out += idx * channel_out;
    for (int c = 0; c < min(channel_out, channel); ++c) {
      if constexpr (std::is_same<INPUT_T, half>::value) {
        *device_out =
            __half2int_rd(__hmul(__hadd_sat(*device_in, half(0.5)), half(255)));
      } else if constexpr (std::is_same<INPUT_T, float>::value) {
        *device_out = __saturatef(*device_in + 0.5f) * 255.f;
      } else {
        static_assert(std::is_same<OUTPUT_T, float>::value,
                      "This kernel does not support int to float conversion.");
        static_assert(std::is_same<OUTPUT_T, half>::value,
                      "This kernel does not support int to half conversion.");
        *device_out = *device_in;
      }
      device_out++;
      device_in += h * w;
    }
  }
}
template <typename DTYPE_IN, typename DTYPE_OUT>
void NCHW_to_NHWC<DTYPE_IN, DTYPE_OUT>::runAsync(
    Ort::Value *input, [[maybe_unused]] Ort::RunOptions *run_options) {
  // void conversion_kernel_NCHW_to_NHWC(INPUT_T* device_in, OUTPUT_T*
  // device_out, uint32_t h, uint32_t w, uint32_t channel, uint32_t channel_out,
  // cudaStream_t stream) {
  const auto shape = input->GetTensorTypeAndShapeInfo().GetShape();
  const auto *device_in = input->GetTensorData<DTYPE_IN>();
  auto *device_out = m_output_tensor->GetTensorMutableData<DTYPE_OUT>();
  const auto channel_out =
      m_output_tensor->GetTensorTypeAndShapeInfo().GetShape()[1];

  dim3 block(256);
  const int values_per_block = block.x;
  dim3 grid(shape[2], (shape[3] + values_per_block - 1) / values_per_block);
  conversion_kernel_NCHW_to_NHWC_impl<DTYPE_IN, DTYPE_OUT>
      <<<grid, block, 0, m_compute_stream>>>(device_in, device_out, shape[2],
                                             shape[3], shape[1], channel_out);
  cudaCheckError(m_compute_stream);
}

template <typename DTYPE_IN, typename DTYPE_OUT>
void NHWC_to_NCHW<DTYPE_IN, DTYPE_OUT>::runAsync(
    Ort::Value *input, [[maybe_unused]] Ort::RunOptions *run_options) {
  // void conversion_kernel_NHWC_to_NCHW(INPUT_T* device_in, OUTPUT_T*
  // device_out, uint32_t h, uint32_t w, uint32_t channel, uint32_t channel_out,
  // cudaStream_t stream) {
  const auto shape = input->GetTensorTypeAndShapeInfo().GetShape();
  const auto *device_in = input->GetTensorData<DTYPE_IN>();
  auto *device_out = m_output_tensor->GetTensorMutableData<DTYPE_OUT>();
  const auto channel_out =
      m_output_tensor->GetTensorTypeAndShapeInfo().GetShape()[1];

  dim3 block(256);
  const int values_per_block = block.x;
  dim3 grid(shape[2], (shape[3] + values_per_block - 1) / values_per_block);
  conversion_kernel_NHWC_to_NCHW_impl<DTYPE_IN, DTYPE_OUT>
      <<<grid, block, 0, m_compute_stream>>>(device_in, device_out, shape[2],
                                             shape[3], shape[1], channel_out);
  cudaCheckError(m_compute_stream);
}
