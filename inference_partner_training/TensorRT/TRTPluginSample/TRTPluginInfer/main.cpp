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

#define NOMINMAX 

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "NvInfer.h"

#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

#include "MyPluginLibrary.h"
#include "util.h"

#define IN_TENSOR_NAME "keras_layer_input:0"
#define OUT_TENSOR_NAME "keras_layer"

static Logger gLogger;

void run_inference(nvinfer1::IExecutionContext* context, const int batch_size)
{
  // Set batch size of input tensor
  int input_binding_idx = context->getEngine().getBindingIndex(IN_TENSOR_NAME);
  nvinfer1::Dims dims = context->getBindingDimensions(input_binding_idx);
  dims.d[0] = batch_size;
  context->setBindingDimensions(input_binding_idx, dims);
  
  // Allocate device data for input / output tensors
  std::vector<void*> buffers(context->getEngine().getNbBindings());
  for (int b_idx = 0; b_idx < context->getEngine().getNbBindings(); ++b_idx)
  {
    CHECK_CUDA(cudaMalloc(&buffers[b_idx], binding_data_size(context, b_idx)));
  }

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  context->enqueueV2(buffers.data(), 0, nullptr);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaDeviceSynchronize());

  float ms;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  std::cout << "Inference took " << ms << " milliseconds;" << std::endl;

  // Cleanup
  for (int b_idx = 0; b_idx < context->getEngine().getNbBindings(); ++b_idx)
  {
    CHECK_CUDA(cudaFree(buffers[b_idx]));
  }
}

int main(int argc, char** argv)
{
  if (argc < 3)
  {
    std::cout << "Usage: TRTPluginInfer.exe path_to_trt_engine batch_size" << std::endl;
    return 0;
  }

  int batch_size = std::stoi(argv[2], nullptr);

  // Initialize plugin library
  initMyPluginLib();

  // Create Runtime object
  nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
  
  // Load engine from disk
  nvinfer1::ICudaEngine* engine = load_engine_from_disk(argv[1], runtime);

  // Create execution context
  nvinfer1::IExecutionContext* context = engine->createExecutionContext();

  run_inference(context, batch_size);

#if NV_TENSORRT_MAJOR < 8
  // Cleanup, not required for TRT 8+
  context->destroy();
  engine->destroy();
  runtime->destroy();
#endif

  return 0;
}
