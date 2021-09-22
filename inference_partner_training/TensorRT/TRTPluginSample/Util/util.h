/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include "NvInfer.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>

#define CHECK_CUDA(call)                                                                              \
{                                                                                                     \
  const cudaError_t err = call;                                                                       \
  if (err != cudaSuccess)                                                                             \
  {                                                                                                   \
    std::cerr << "CudaDebugCall() failed at: " << __FILE__ << ":" << __LINE__ << "; ";                \
    std::cerr << "code: " << err << "; description: " << cudaGetErrorString(err) << std::endl;        \
    exit(1);                                                                                          \
  }                                                                                                   \
}                                                                                                     \

inline void write_engine_to_disk(const nvinfer1::ICudaEngine* engine, const std::string& filename)
{
  const nvinfer1::IHostMemory* serialized = engine->serialize();

  // Create missing intermediate directories
  std::filesystem::path path = std::filesystem::path(filename).remove_filename();
  std::error_code ec;
  if (!path.empty() && !std::filesystem::create_directories(path, ec))
  {
    std::cout << ec.message() << std::endl;
  }
  
  // Write to disk
  std::ofstream out(filename.c_str(), std::ios::binary);
  out.write(static_cast<const char*>(serialized->data()), serialized->size());
  out.close();

  if (!out)
  {
    std::cout << "Error: Failed to write engine to disk." << std::endl;
  }
}

inline nvinfer1::ICudaEngine* load_engine_from_disk(const std::string& filename, nvinfer1::IRuntime* runtime)
{
  std::ifstream in(filename.c_str(), std::ios::binary);
  std::vector<char> buffer(std::istreambuf_iterator<char>(in), {});

  return runtime->deserializeCudaEngine(buffer.data(), buffer.size());
}

static size_t bytes_per_date(nvinfer1::DataType dt)
{
  switch (dt)
  {
  case nvinfer1::DataType::kFLOAT:  return 4ull;
  case nvinfer1::DataType::kINT32:  return 4ull;
  case nvinfer1::DataType::kHALF:   return 2ull;
  case nvinfer1::DataType::kINT8:   return 1ull;
  case nvinfer1::DataType::kBOOL:   return 1ull;
  }
  return 0ull;
}

static size_t binding_data_size(nvinfer1::IExecutionContext* context, int binding_idx)
{
  size_t n = static_cast<size_t>(std::max(1, context->getBindingDimensions(binding_idx).d[0]));
  size_t c = static_cast<size_t>(std::max(1, context->getBindingDimensions(binding_idx).d[1]));
  size_t h = static_cast<size_t>(std::max(1, context->getBindingDimensions(binding_idx).d[2]));
  size_t w = static_cast<size_t>(std::max(1, context->getBindingDimensions(binding_idx).d[3]));
  nvinfer1::DataType dt = context->getEngine().getBindingDataType(binding_idx);
  return n * c * h * w * bytes_per_date(dt);
}

class Logger : public nvinfer1::ILogger
{
public:

  virtual void log(Severity severity, const char* msg) noexcept override
  {
    switch (severity)
    {
    case ILogger::Severity::kINTERNAL_ERROR:
      std::cout << "INTERNAL_ERROR: " << msg << std::endl; break;
    case ILogger::Severity::kERROR:
      std::cout << "ERROR: " << msg << std::endl; break;
    case ILogger::Severity::kWARNING:
      std::cout << "WARNING: " << msg << std::endl; break;
    case ILogger::Severity::kINFO:
      std::cout << "INFO: " << msg << std::endl; break;
    case ILogger::Severity::kVERBOSE:
      std::cout << "VERBOSE: " << msg << std::endl; break;
    }
  }
};
