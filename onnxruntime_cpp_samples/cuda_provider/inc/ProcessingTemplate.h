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
#include "onnx_helpers.h"
#include <cuda_runtime.h>
#include <map>

class ProcessingTemplate {
public:
  ProcessingTemplate() = default;

  ProcessingTemplate(ProcessingTemplate &rhs) = delete;

  ProcessingTemplate(ProcessingTemplate &&rhs) = delete;

  virtual void runAsync(Ort::Value *in, Ort::RunOptions *opts) = 0;

  void setComputeStream(cudaStream_t str) { m_compute_stream = str; }

  virtual void allocateOutputTensor(std::vector<int64_t> shape,
                                    ONNXTensorElementDataType type,
                                    Ort::Allocator *allocator) {
    m_output_tensor = allocateTensor(shape, type, allocator);
  }

  virtual void allocateInputTensor(std::vector<int64_t> shape,
                                   ONNXTensorElementDataType type,
                                   Ort::Allocator *allocator) {
    m_input_tensor = allocateTensor(shape, type, allocator);
  }

  virtual Ort::Value *fetchOutputTensorAsync() const {
    return m_output_tensor.get();
  }

  virtual Ort::Value *fetchInputTensorAsync() const {
    return m_input_tensor.get();
  }

protected:
  static std::unique_ptr<Ort::Value>
  allocateTensor(std::vector<int64_t> shape, ONNXTensorElementDataType type,
                 Ort::Allocator *allocator) {
    switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
      using T = float;
      size_t tensor_bytes = sizeof(T);
      for (auto v : shape)
        tensor_bytes *= v;
      auto *raw_data_out =
          reinterpret_cast<T *>(allocator->Alloc(tensor_bytes));
      return std::make_unique<Ort::Value>(Ort::Value::CreateTensor<T>(
          allocator->GetInfo(), raw_data_out, tensor_bytes, shape.data(),
          shape.size()));
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
      using T = uint8_t;
      size_t tensor_bytes = sizeof(T);
      for (auto v : shape)
        tensor_bytes *= v;
      auto *raw_data_out =
          reinterpret_cast<T *>(allocator->Alloc(tensor_bytes));
      return std::make_unique<Ort::Value>(Ort::Value::CreateTensor<T>(
          allocator->GetInfo(), raw_data_out, tensor_bytes, shape.data(),
          shape.size()));
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
      using T = Ort::Float16_t;
      size_t tensor_bytes = sizeof(T);
      for (auto v : shape)
        tensor_bytes *= v;
      auto *raw_data_out =
          reinterpret_cast<T *>(allocator->Alloc(tensor_bytes));
      return std::make_unique<Ort::Value>(Ort::Value::CreateTensor<T>(
          allocator->GetInfo(), raw_data_out, tensor_bytes, shape.data(),
          shape.size()));
    }
    default:
      std::cerr << "This type is not supported" << std::endl;
      return nullptr;
    }
  }
  cudaStream_t m_compute_stream = nullptr;
  std::unique_ptr<Ort::Value> m_output_tensor, m_input_tensor;
};
