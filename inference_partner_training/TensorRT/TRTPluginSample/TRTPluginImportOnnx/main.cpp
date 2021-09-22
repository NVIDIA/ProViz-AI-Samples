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
#include "NvOnnxParser.h"

#include "MyPluginLibrary.h"

#include "util.h"

#include <stdio.h>
#include <string>
#include <string.h>
#include <iostream>
#include <vector>

#define IN_TENSOR_NAME "keras_layer_input:0"
#define OUT_TENSOR_NAME "keras_layer"

static Logger gLogger;

nvonnxparser::IParser* parse_onnx(nvinfer1::INetworkDefinition* network, const std::string& onnx_file)
{
  nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
  parser->parseFromFile(onnx_file.c_str(), 0);
  
  if (parser->getNbErrors())
  {
    std::cout << "Found " << parser->getNbErrors() << " errors:" << std::endl;
    for (int i = 0; i < parser->getNbErrors(); ++i)
    {
      std::cout << parser->getError(i)->desc() << std::endl;
    }
  }

  return parser;
}

int main(int argc, char** argv)
{
  if (argc < 3)
  {
    std::cout << "Usage: TRTPluginImportONNX.exe path_to_onnx_model output_engine_file  [--fp16]" << std::endl;
    return 0;
  }

  const bool enable_fp16 = (argc >= 4) && !strcmp(argv[3], "--fp16");

  initMyPluginLib();

  // Create builder
  nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);

  // Create network definition
  nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1u << static_cast<int>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

  // Parse ONNX model
  nvonnxparser::IParser* parser = parse_onnx(network, argv[1]);

  // Set max workspace size and precision
  nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
  config->setMaxWorkspaceSize(16ull * 1024ull * 1024ull); // use 16MB of workspace max
  if (enable_fp16)
  {
    config->setFlag(nvinfer1::BuilderFlag::kFP16); // enable mixed precision inference
  }

  // Since our model was using dynamic batch sizes we specify have to specify an optimization profile
  nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
  profile->setDimensions(IN_TENSOR_NAME, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1,224,224,3));
  profile->setDimensions(IN_TENSOR_NAME, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(32,224,224,3));
  profile->setDimensions(IN_TENSOR_NAME, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(64,224,224,3));
  config->addOptimizationProfile(profile);

  // Build engine
  nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

  // Write engine to disk
  write_engine_to_disk(engine, argv[2]);

#if NV_TENSORRT_MAJOR < 8
  // Cleanup, not required for TRT 8+
  network->destroy();
  config->destroy();
  builder->destroy();
  engine->destroy();
  parser->destroy();
#endif

  return 0;
}
