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
#include <vector>
#include <iostream>
#include <complex>

#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"

static auto operator<<(std::ostream& out, const ONNXTensorElementDataType dtype) -> std::ostream& {
  const std::vector<std::string> strings = {
      "ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED",
      "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT",
      "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8",
      "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8",
      "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16",
      "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16",
      "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32",
      "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64",
      "ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING",
      "ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL",
      "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16",
      "ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE",
      "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32",
      "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64",
      "ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64",
      "ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128",
      "ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16",
  };
  return out << strings[int(dtype)];
}

static size_t sizeof_tensor_element(const ONNXTensorElementDataType dtype) {
  size_t element_size = 0;
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
      element_size = sizeof(float);
      break;
    }  // maps to c type float
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
      element_size = sizeof(uint8_t);
      break;
    }  // maps to c type uint8_t
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
      element_size = sizeof(int8_t);
      break;
    }  // maps to c type int8_t
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: {
      element_size = sizeof(uint16_t);
      break;
    }  // maps to c type uint16_t
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: {
      element_size = sizeof(int16_t);
      break;
    }  // maps to c type int16_t
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
      element_size = sizeof(int32_t);
      break;
    }  // maps to c type int32_t
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
      element_size = sizeof(int64_t);
      break;
    }  // maps to c type int64_t
       //        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: {
       //        }  // maps to c++ type std::string
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
      element_size = sizeof(bool);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
      element_size = 2;
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
      element_size = sizeof(double);
      break;
    }  // maps to c type double
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: {
      element_size = sizeof(uint32_t);
      break;
    }  // maps to c type uint32_t
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: {
      element_size = sizeof(uint64_t);
      break;
    }  // maps to c type uint64_t
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64: {
      element_size = sizeof(std::complex<float>);
      break;
    }  // complex with float32 real and imaginary components
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: {
      element_size = sizeof(std::complex<double>);
      break;
    }  // complex with float64 real and imaginary components
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: {
      element_size = 2;
      break;
    }  // Non-IEEE floating-point format based on IEEE754 single-precision
    default: {
      throw std::runtime_error("unsupported dtype");
    }
  }
  return element_size;
}
