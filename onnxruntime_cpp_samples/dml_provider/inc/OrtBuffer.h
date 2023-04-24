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

#ifndef __H_ORT_BUFFER__
#define __H_ORT_BUFFER__

#pragma once

#include "Common.h"
#include "DXQueue.h"
#include "onnxruntime_c_api.h"

class OrtInference;

class OrtBuffer {
public:
  enum Type { ORT_BUFFER = 0, ORT_IMAGE, ORT_BUFFER_TYPE_COUNT };

  enum DataType {
    ORT_BUFFER_INT32 = 0,
    ORT_BUFFER_UINT32,
    ORT_BUFFER_FLOAT32,
    ORT_BUFFER_FLOAT16,
    ORT_BUFFER_INT8,
    ORT_BUFFER_UINT8,
    ORT_BUFFER_TYPE_UNDEFINED,
    ORT_DATA_TYPE_COUNT
  };

  OrtBuffer();
  ~OrtBuffer();

  static OrtBuffer::DataType
  ONNXToInternalDataType(ONNXTensorElementDataType input_data_type);

  E2EError InitBuffer(const DataType inDataType, const size_t inDimensionCount,
                      int64_t *inDimensions, OrtInference *inParent);

  E2EError InitBufferWithData(const DataType inDataType,
                              const size_t inDimensionCount,
                              int64_t *inDimensions, void *inData,
                              OrtInference *inParent,
                              ComPtr<ID3D12GraphicsCommandList2> inCmdList);

  E2EError ReadBack();

  float *MapFloat();
  uint8_t *MapUint8();

  bool IsReady() const;
  bool IsInput() const;
  bool IsOutput() const;

  void IsInput(const bool inState);
  void IsOutput(const bool inState);

  Type GetType() const;
  void SetType(const Type inType);

  DataType GetDataType() const;
  void SetDataType(const DataType inType);

  void SetDimensions(const size_t inCount, int64_t *inDimensions);

  size_t GetDimensionCount() const;
  size_t GetElementCount() const;
  size_t GetDataSize() const;

  ONNXTensorElementDataType GetONNXDataType();
  DXGI_FORMAT GetBufferViewFormat();

  OrtValue *GetOrtValue() const;
  ComPtr<ID3D12Resource> GetD3DResource();

  D3D12_UNORDERED_ACCESS_VIEW_DESC GetUAVDesc();

private:
  size_t GetDataTypeSize() const;

  E2EError CreateDMLResource();

  void *m_cpu_data;

  std::vector<int64_t> m_shape;

  Type m_type;
  DataType m_data_type;

  bool m_is_ready;
  // if it's an input, it needs to be
  // staged before inference starts.
  bool m_is_input;
  // if it's an output it may need to be
  // transfered back after inference has finished.
  bool m_is_output;

  // D3D Resource Objects.
  ComPtr<ID3D12Resource> m_d3d_buffer;
  ComPtr<ID3D12Resource> m_staging_buffer;
  ComPtr<ID3D12Resource> m_readback_buffer;
  ;

  // DML Resource, created from
  // the D3D12 Resource.
  void *m_dml_resource;

  // ORT Value Object
  OrtValue *m_ort_value;

  OrtInference *m_parent;
};

#endif
