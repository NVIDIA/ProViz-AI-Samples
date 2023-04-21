/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "OrtBuffer.h"
#include "OrtInference.h"
#include <dml_provider_factory.h>

OrtBuffer::DataType
OrtBuffer::ONNXToInternalDataType(ONNXTensorElementDataType input_data_type) {

  OrtBuffer::DataType buffer_data_type =
      OrtBuffer::DataType::ORT_BUFFER_TYPE_UNDEFINED;

  switch (input_data_type) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    buffer_data_type = OrtBuffer::DataType::ORT_BUFFER_FLOAT32;
    break;

  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    buffer_data_type = OrtBuffer::DataType::ORT_BUFFER_FLOAT16;
    break;

  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    buffer_data_type = OrtBuffer::DataType::ORT_BUFFER_INT8;
    break;

  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    buffer_data_type = OrtBuffer::DataType::ORT_BUFFER_INT32;
    break;

  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    buffer_data_type = OrtBuffer::DataType::ORT_BUFFER_UINT8;
    break;

  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    buffer_data_type = OrtBuffer::DataType::ORT_BUFFER_UINT32;
    break;

  default: {
    LOGE("Input Tensor Unsupported Data Type.");
  } break;
  }

  return buffer_data_type;
}

OrtBuffer::OrtBuffer()
    : m_is_ready(false), m_is_input(true), m_is_output(false),
      m_cpu_data(nullptr), m_type(Type::ORT_BUFFER),
      m_data_type(DataType::ORT_BUFFER_FLOAT32), m_dml_resource(nullptr),
      m_ort_value(nullptr) {}

OrtBuffer::~OrtBuffer() {}

E2EError OrtBuffer::InitBuffer(const DataType inDataType,
                               const size_t inDimensionCount,
                               int64_t *inDimensions, OrtInference *inParent) {
  E2EError out_err = NO_E2E_ERROR;

  m_parent = inParent;

  ComPtr<ID3D12Device2> d3d_device = m_parent->GetD3DDevice();

  // Set Data Type
  m_data_type = inDataType;

  // Set Dimenisons
  SetDimensions(inDimensionCount, inDimensions);

  size_t data_length = GetDataSize();
  size_t element_count = GetElementCount();
  size_t dTypeSize = GetDataTypeSize();

  D3D12_HEAP_PROPERTIES default_props =
      CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
  D3D12_RESOURCE_DESC buffer_desc = CD3DX12_RESOURCE_DESC::Buffer(
      data_length, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

  // Create a committed resource for the final destination.
  ThrowOnFail(d3d_device->CreateCommittedResource(
      &default_props, D3D12_HEAP_FLAG_NONE, &buffer_desc,
      D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_d3d_buffer)));

  return CreateDMLResource();
}

E2EError OrtBuffer::InitBufferWithData(
    const OrtBuffer::DataType inDataType, const size_t inDimensionCount,
    int64_t *inDimensions, void *inData, OrtInference *inParent,
    ComPtr<ID3D12GraphicsCommandList2> inCmdList) {
  E2EError out_err = NO_E2E_ERROR;

  // We Recieve the data as a raw buffer, the dimensions
  // and the data type from which we can calcuate the
  // required data aize.

  // Set the parent so we can access things
  // like the command queue and the updatebufferresources call
  m_parent = inParent;

  // Set Data Type
  m_data_type = inDataType;

  // Set Dimenisons
  SetDimensions(inDimensionCount, inDimensions);

  // Get the data size of the tensor we
  // need to create.
  size_t data_length = GetDataSize();
  size_t element_count = GetElementCount();

  // We have a staging buffer as a class memeber.
  // This is not a good design, neither is using
  // committed resources.
  // TODO - Refactor to use a heap allocator to
  // leverate placed resources rather than
  // committed resources.

  // Get the copy queue from the parent.
  // And get a command list that's ready to
  // record into.
  size_t dTypeSize = GetDataTypeSize();

  // Create the D3DBuffer and queue up the transfer
  // from host to device.
  m_parent->UpdateBufferResource(
      inCmdList.Get(), &m_d3d_buffer, &m_staging_buffer, element_count,
      dTypeSize, inData, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

  return CreateDMLResource();
}

E2EError OrtBuffer::CreateDMLResource() {
  E2EError out_err = NO_E2E_ERROR;
  OrtApi const &ortApi = Ort::GetApi();
  const OrtDmlApi *ort_dml_api;
  OrtStatusPtr rslt = ortApi.GetExecutionProviderApi(
      "DML", ORT_API_VERSION, reinterpret_cast<const void **>(&ort_dml_api));

  // Create the dml resource from the D3D resource.
  ort_dml_api->CreateGPUAllocationFromD3DResource(m_d3d_buffer.Get(),
                                                  &m_dml_resource);
  std::shared_ptr<Ort::MemoryInfo> mem_info = m_parent->GetMemoryInfo();
  size_t data_length = GetDataSize();

  ONNXTensorElementDataType onnx_dtype = GetONNXDataType();
  if (onnx_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
    return UNSUPPORTED_BINDING_DATA_TYPE;
  }

  rslt = ortApi.CreateTensorWithDataAsOrtValue(
      (OrtMemoryInfo *)mem_info.get(), m_dml_resource, data_length,
      m_shape.data(), m_shape.size(), onnx_dtype, &m_ort_value);

  return out_err;
}

E2EError OrtBuffer::ReadBack() {
  E2EError out_err = NO_E2E_ERROR;

  size_t data_size = GetDataSize();
  ComPtr<ID3D12Device2> d3d_device = m_parent->GetD3DDevice();

  CD3DX12_HEAP_PROPERTIES heapProps =
      CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
  CD3DX12_RESOURCE_DESC resDesc = CD3DX12_RESOURCE_DESC::Buffer(data_size);

  ThrowOnFail(d3d_device->CreateCommittedResource(
      &heapProps, D3D12_HEAP_FLAG_NONE, &resDesc,
      D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
      IID_PPV_ARGS(&m_readback_buffer)));

  std::shared_ptr<DXQueue> copy_queue = m_parent->GetCopyQueue();
  ComPtr<ID3D12GraphicsCommandList2> cmdList = copy_queue->GetCmdList();

  CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
      m_d3d_buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
      D3D12_RESOURCE_STATE_COPY_SOURCE);

  cmdList->ResourceBarrier(1, &barrier);

  cmdList->CopyResource(m_readback_buffer.Get(), m_d3d_buffer.Get());
  copy_queue->ExecuteCommandList(cmdList);
  copy_queue->Flush();

  return out_err;
}

float *OrtBuffer::MapFloat() {
  size_t data_size = GetDataSize();
  D3D12_RANGE readbackBufferRange{0, data_size};
  float *mapped = nullptr;
  ThrowOnFail(m_readback_buffer->Map(0, &readbackBufferRange,
                                     reinterpret_cast<void **>(&mapped)));

  return mapped;
}

uint8_t *OrtBuffer::MapUint8() {
  size_t data_size = GetDataSize();
  D3D12_RANGE readbackBufferRange{0, data_size};
  uint8_t *mapped = nullptr;
  ThrowOnFail(m_readback_buffer->Map(0, &readbackBufferRange,
                                     reinterpret_cast<void **>(&mapped)));

  return mapped;
}

bool OrtBuffer::IsReady() const { return m_is_ready; }

bool OrtBuffer::IsInput() const { return m_is_input; }

bool OrtBuffer::IsOutput() const { return m_is_output; }

void OrtBuffer::IsInput(const bool inState) { m_is_input = inState; }

void OrtBuffer::IsOutput(const bool inState) { m_is_output = inState; }

OrtBuffer::Type OrtBuffer::GetType() const { return m_type; }

void OrtBuffer::SetType(const OrtBuffer::Type inType) { m_type = inType; }

OrtBuffer::DataType OrtBuffer::GetDataType() const { return m_data_type; }

void OrtBuffer::SetDataType(const OrtBuffer::DataType inDataType) {
  m_data_type = inDataType;
}

void OrtBuffer::SetDimensions(const size_t inCount, int64_t *inDimensions) {
  m_shape.clear();
  if (!inDimensions)
    return;
  for (size_t i = 0; i < inCount; ++i) {
    m_shape.push_back(inDimensions[i]);
  }
}

size_t OrtBuffer::GetDimensionCount() const { return m_shape.size(); }

size_t OrtBuffer::GetElementCount() const {
  size_t dim_count = m_shape.size();
  if (dim_count == 0)
    return 0;
  size_t out_count = 1;
  for (size_t i = 0; i < dim_count; ++i) {
    out_count *= m_shape[i];
  }

  return out_count;
}

size_t OrtBuffer::GetDataSize() const {
  return GetElementCount() * GetDataTypeSize();
}

OrtValue *OrtBuffer::GetOrtValue() const { return m_ort_value; }

ComPtr<ID3D12Resource> OrtBuffer::GetD3DResource() { return m_d3d_buffer; }

size_t OrtBuffer::GetDataTypeSize() const {

  static map<DataType, size_t> dtype_to_dtype_size = {
      {DataType::ORT_BUFFER_INT32, sizeof(int32_t)},
      {DataType::ORT_BUFFER_UINT32, sizeof(uint32_t)},
      {DataType::ORT_BUFFER_FLOAT32, sizeof(float)},
      {DataType::ORT_BUFFER_FLOAT16, sizeof(uint16_t)},
      {DataType::ORT_BUFFER_INT8, sizeof(int8_t)},
      {DataType::ORT_BUFFER_UINT8, sizeof(uint8_t)}};

  std::map<DataType, size_t>::iterator itr =
      dtype_to_dtype_size.find(m_data_type);
  if (itr == dtype_to_dtype_size.end())
    return 0;
  return itr->second;
}

ONNXTensorElementDataType OrtBuffer::GetONNXDataType() {

  static map<DataType, ONNXTensorElementDataType> dtype_to_onnx_dtype = {
      {DataType::ORT_BUFFER_INT32, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32},
      {DataType::ORT_BUFFER_UINT32, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32},
      {DataType::ORT_BUFFER_FLOAT32, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT},
      {DataType::ORT_BUFFER_FLOAT16, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16},
      {DataType::ORT_BUFFER_INT8, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8},
      {DataType::ORT_BUFFER_UINT8, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8}};

  std::map<DataType, ONNXTensorElementDataType>::iterator itr =
      dtype_to_onnx_dtype.find(m_data_type);
  if (itr == dtype_to_onnx_dtype.end())
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  return itr->second;
}

DXGI_FORMAT OrtBuffer::GetBufferViewFormat() {

  static map<DataType, DXGI_FORMAT> dtype_to_dxgi_format = {
      {DataType::ORT_BUFFER_INT32, DXGI_FORMAT_R32_SINT},
      {DataType::ORT_BUFFER_UINT32, DXGI_FORMAT_R32_UINT},
      {DataType::ORT_BUFFER_FLOAT32, DXGI_FORMAT_R32_FLOAT},
      {DataType::ORT_BUFFER_FLOAT16, DXGI_FORMAT_R16_FLOAT},
      {DataType::ORT_BUFFER_INT8, DXGI_FORMAT_R8_SINT},
      {DataType::ORT_BUFFER_UINT8, DXGI_FORMAT_R8_UINT}};

  std::map<DataType, DXGI_FORMAT>::iterator itr =
      dtype_to_dxgi_format.find(m_data_type);
  if (itr == dtype_to_dxgi_format.end())
    return DXGI_FORMAT_UNKNOWN;
  return itr->second;
}

D3D12_UNORDERED_ACCESS_VIEW_DESC OrtBuffer::GetUAVDesc() {
  D3D12_UNORDERED_ACCESS_VIEW_DESC out_desc = {};
  out_desc.Format = GetBufferViewFormat();
  out_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
  out_desc.Buffer.FirstElement = 0;
  out_desc.Buffer.NumElements = GetElementCount();
  out_desc.Buffer.StructureByteStride = 0;
  out_desc.Buffer.CounterOffsetInBytes = 0;
  out_desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

  return out_desc;
}