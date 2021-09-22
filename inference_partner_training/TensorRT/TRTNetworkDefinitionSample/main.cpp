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

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "NvInfer.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>

#include "util.h"


#define IN_TENSOR_NAME "INPUT"
#define OUT_TENSOR_NAME "OUTPUT"

#define IN_CHANNELS 3
#define KERNEL_SIZE 3
#define KERNEL_FILTERS 1

static Logger gLogger;

void create_single_conv_network(nvinfer1::INetworkDefinition* network, const std::vector<float>& weight_data, const std::vector<float>& bias_data)
{
  // Set network input
  int32_t in_b = -1;                                      // input batch size (dynamic)
  int32_t in_w = -1;                                      // input width (dynamic)
  int32_t in_h = -1;                                      // input height (dynamic)
  int32_t in_c = IN_CHANNELS;                             // input channels (static)
  nvinfer1::DataType in_dt = nvinfer1::DataType::kFLOAT;  // input data type (static)
  nvinfer1::ITensor* in_tensor = network->addInput(IN_TENSOR_NAME, in_dt, nvinfer1::Dims4(in_b, in_c, in_h, in_w));

  // Add convolution
  nvinfer1::DimsHW ks(KERNEL_SIZE, KERNEL_SIZE);

  nvinfer1::Weights weights;
  weights.count = IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * KERNEL_FILTERS;
  weights.type = nvinfer1::DataType::kFLOAT;
  weights.values = weight_data.data();

  nvinfer1::Weights bias;
  bias.count = KERNEL_FILTERS;
  bias.type = nvinfer1::DataType::kFLOAT;
  bias.values = bias_data.data();

  nvinfer1::IConvolutionLayer* conv = network->addConvolution(*in_tensor, KERNEL_FILTERS, ks, weights, bias);
  conv->setStride(nvinfer1::DimsHW(1, 1));
  conv->setPadding(nvinfer1::DimsHW(1, 1));

  // Set network output
  conv->getOutput(0)->setName(OUT_TENSOR_NAME);
  network->markOutput(*conv->getOutput(0));
  conv->getOutput(0)->setType(nvinfer1::DataType::kFLOAT);
}

int main(int argc, char** argv)
{
  if (argc < 2)
  {
    std::cout << "Usage: TRTNetworkDefinitionSample.exe output_engine_file" << std::endl;
    return 0;
  }

  // Create builder
  nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);

  // Create network definition with dynamic shape support
  nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1u << static_cast<int>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

  // Create weight and bias data
  std::vector<float> weight_data(IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * KERNEL_FILTERS, 1.0f);
  std::vector<float> bias_data(KERNEL_FILTERS, 0.0f);

  // Create single conv network using network definition API
  create_single_conv_network(network, weight_data, bias_data);

  // Set engine parameters and create optimization profile
  nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
  config->setMaxWorkspaceSize(16ull * 1024ull * 1024ull); // use 16MB of workspace max
  config->setFlag(nvinfer1::BuilderFlag::kFP16); // enable mixed precision inference

  nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
  profile->setDimensions(IN_TENSOR_NAME, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 3, 640, 480));
  profile->setDimensions(IN_TENSOR_NAME, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 3, 1920, 1080));
  profile->setDimensions(IN_TENSOR_NAME, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(8, 3, 3840, 2160));
  config->addOptimizationProfile(profile);

  // Build engine
  nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

  // Serialize engine and write to disk
  write_engine_to_disk(engine, argv[1]);

#if NV_TENSORRT_MAJOR < 8
  // Cleanup, not required for TRT 8+
  network->destroy();
  config->destroy();
  builder->destroy();
  engine->destroy();
#endif

  return 0;
}