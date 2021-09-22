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

#include "MyPlugin.h"

#include <cassert>
#include <iostream>
#include <string.h>

#include <cuda_runtime.h>

#include "util.h"

namespace
{
  size_t tensor_size_in_bytes(const nvinfer1::PluginTensorDesc& desc)
  {
    int num_elements = 1;
    for (int j = 0; j < desc.dims.nbDims; ++j)
    {
      num_elements *= desc.dims.d[j];
    }

    switch (desc.type)
    {
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8:
      return num_elements;
    case nvinfer1::DataType::kHALF:
      return num_elements * 2;
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kINT32:
      return num_elements * 4;
    }

    return 0;
  }
}

MyPlugin::MyPlugin(int num_inputs) noexcept :
  m_num_inputs(num_inputs)
{
  assert(num_inputs > 0u);
}

const char* MyPlugin::getPluginType() const noexcept
{
  return "MyPlugin";
}

const char* MyPlugin::getPluginVersion() const noexcept
{
  return "1";
}

int MyPlugin::getNbOutputs() const noexcept
{
  return 1;
}

int MyPlugin::initialize() noexcept
{
  return 0; // SUCCESS
}

void MyPlugin::terminate() noexcept
{
}

size_t MyPlugin::getSerializationSize() const noexcept
{
  return sizeof(m_num_inputs); // m_num_inputs
}

void MyPlugin::serialize(void* buffer) const noexcept
{
  *reinterpret_cast<int*>(buffer) = m_num_inputs;
}

void MyPlugin::destroy() noexcept
{
  delete this;
}

void MyPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
  m_namespace = libNamespace;
}

inline const char* MyPlugin::getPluginNamespace() const noexcept
{
  return m_namespace.c_str();
}

nvinfer1::DataType MyPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
  assert(index < m_num_inputs);
  assert(nbInputs == m_num_inputs);
  return inputTypes[index];
}

nvinfer1::IPluginV2DynamicExt* MyPlugin::clone() const noexcept
{
  nvinfer1::IPluginV2DynamicExt* plugin = new MyPlugin(m_num_inputs);
  plugin->setPluginNamespace(m_namespace.c_str());
  return plugin;
}

nvinfer1::DimsExprs MyPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
  assert(outputIndex < m_num_inputs);
  assert(nbInputs == m_num_inputs);
  return inputs[outputIndex];
}

bool MyPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
  assert(pos < 2 * m_num_inputs);
  assert(nbInputs == m_num_inputs);
  assert(nbOutputs == m_num_inputs);

  // Check whether tensor storage is linear
  if (inOut[pos].format != nvinfer1::TensorFormat::kLINEAR)
    return false;

  if (pos < nbInputs)
    return true; // This is an input, check all properties that need to match for outputs

  // pos belongs to an output tensor
  const nvinfer1::PluginTensorDesc& in = inOut[pos - nbInputs];
  const nvinfer1::PluginTensorDesc& out = inOut[pos];

  // Check whether dimensions match
  if (in.dims.nbDims != out.dims.nbDims)
    return false;

  for (int i = 0; i < in.dims.nbDims; ++i)
  {
    if (in.dims.d[i] != out.dims.d[i])
      return false;
  }

  // Check if scales match in case of INT8 tensors
  if ((in.type == nvinfer1::DataType::kINT8) && in.scale != out.scale)
    return false;

  if ((in.format != out.format) || (in.type != out.type))
    return false;

  // Check if formats and data types match
  return true;
}

void MyPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
  assert(nbInputs == nbOutputs);
}

size_t MyPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
  return 0;
}

int MyPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
  std::cout << "Running MyPlugin with m_num_inputs==" << m_num_inputs << ";" << std::endl;
  for (int i = 0; i < m_num_inputs; ++i)
  {
    // Copying input to output
    CHECK_CUDA(cudaMemcpyAsync(outputs[i], inputs[i], tensor_size_in_bytes(inputDesc[i]), cudaMemcpyDeviceToDevice, stream));
  }

  return 0; // Success
}

MyPluginCreator::MyPluginCreator()
{
  m_fields.emplace_back("num_inputs", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
  m_fc.fields = m_fields.data();
  m_fc.nbFields = m_fields.size();
}

const char* MyPluginCreator::getPluginName() const noexcept
{
    return "MyPlugin";
}

const char* MyPluginCreator::getPluginVersion() const noexcept
{
    return "1";
}

const nvinfer1::PluginFieldCollection* MyPluginCreator::getFieldNames() noexcept
{
  return &m_fc;
}

nvinfer1::IPluginV2DynamicExt* MyPluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept
{
  assert(fc->nbFields == 1);
  assert(!strcmp(fc->fields[0].name, "num_inputs"));
  nvinfer1::IPluginV2DynamicExt* plugin = new MyPlugin(*reinterpret_cast<const int*>(fc->fields[0].data));
  plugin->setPluginNamespace(m_namespace.c_str());
  return plugin;
}

nvinfer1::IPluginV2DynamicExt* MyPluginCreator::deserializePlugin(const char* name, const void* data, size_t data_size) noexcept
{
  assert(data_size >= sizeof(int));
  nvinfer1::IPluginV2DynamicExt* plugin = new MyPlugin(*reinterpret_cast<const int*>(data));
  plugin->setPluginNamespace(m_namespace.c_str());
  return plugin;
}

void MyPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept
{
  m_namespace = pluginNamespace;
}

const char* MyPluginCreator::getPluginNamespace() const noexcept
{
  return m_namespace.c_str();
}
