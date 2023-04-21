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

#ifndef __H_ORT_INFERENCE__
#define __H_ORT_INFERENCE__

#pragma once

#include "Common.h"
#include "DXQueue.h"
#include "E2EImageUtils.h"
#include <DirectML.h>

using namespace e2eai;

typedef std::string BindingName;
typedef std::string EnginePath;

class OrtBuffer;

class OrtInference {
public:
  OrtInference();
  ~OrtInference();

  struct BindingMetaData {
    ONNXTensorElementDataType data_type;
    std::vector<int64_t> shape;
  };

  typedef std::map<BindingName, BindingMetaData> BindingMetaDataArray;

  struct Requirements {
    size_t input_count;
    size_t output_count;
    size_t overridable_init_count;

    BindingMetaDataArray input_bindings;
    BindingMetaDataArray output_bindings;
  };

  E2EError InitD3DDevice();

  E2EError InitInference(EnginePath &inEnginePath, EnginePath &inEnginePath2);
  E2EError LoadEngine();

  E2EError QuerySessionRequirements();

  E2EError InitBindings();

  E2EError InitPreprocessCommands();

  E2EError InitPostProcessCommands();

  E2EError UpdateBufferResource(ComPtr<ID3D12GraphicsCommandList2> inCmdList,
                                ID3D12Resource **inDestResource,
                                ID3D12Resource **inIntermediateResource,
                                size_t inElementCount, size_t inElementSize,
                                const void *inBufferData,
                                D3D12_RESOURCE_FLAGS inFlags);

  E2EError Run(void **inBuffers);

  E2EError LoadInput(const char *inPath);
  E2EError SaveResult(const char *inPath);

  std::shared_ptr<DXQueue> GetCopyQueue();
  std::shared_ptr<DXQueue> GetComputeQueue();
  std::shared_ptr<Ort::MemoryInfo> GetMemoryInfo();

  ComPtr<ID3D12Device2> GetD3DDevice();

  ONNXTensorElementDataType GetInputDataType();
  ONNXTensorElementDataType GetOutputDataType();

  BindingName GetInputBindingName();
  BindingName GetOutputBindingName();

private:
  std::chrono::high_resolution_clock m_clock;
  std::chrono::steady_clock::time_point m_t0;

  EnginePath m_engine_path;
  EnginePath m_engine_path2;

  std::unique_ptr<Ort::Env> m_env;
  std::unique_ptr<Ort::Session> m_session;
  std::unique_ptr<Ort::Session> m_session_2;
  std::unique_ptr<Ort::IoBinding> m_io_binding;
  std::unique_ptr<Ort::IoBinding> m_io_binding2;

  std::shared_ptr<Ort::MemoryInfo> m_memory_info;

  // DX12 Releated
  ComPtr<IDXGIAdapter4> m_adapter;
  ComPtr<ID3D12Device2> m_device;

  // Buffers..
  OrtBuffer *m_img_input_buffer;

  OrtBuffer *m_ort_input_buffer;

  OrtBuffer *m_ort_output_buffer;

  OrtBuffer *m_img_output_buffer;

  ComPtr<IDMLDevice> m_dml_device;

  std::shared_ptr<DXQueue> m_copy_queue;
  std::shared_ptr<DXQueue> m_compute_queue;

  ComPtr<ID3D12RootSignature> m_pre_process_root_signature;
  ComPtr<ID3D12PipelineState> m_pre_process_pso;

  ComPtr<ID3D12RootSignature> m_post_process_root_signature;
  ComPtr<ID3D12PipelineState> m_post_process_pso;

  ComPtr<ID3D12DescriptorHeap> m_pre_process_heap;
  ComPtr<ID3D12DescriptorHeap> m_post_process_heap;

  ComPtr<ID3D12GraphicsCommandList2> m_cmd_list_stage_input;
  ComPtr<ID3D12GraphicsCommandList2> m_cmd_list_stage_output;
  ComPtr<ID3D12GraphicsCommandList2> m_cmd_list_preprocess_input;
  ComPtr<ID3D12GraphicsCommandList2> m_cmd_list_postprocess_output;

  ComPtr<ID3D12QueryHeap> m_timestamp_query_heap;
  ComPtr<ID3D12Resource> m_timestamp_readback;

  ImageData m_input_image;

  UINT m_uav_descriptor_size;

  Requirements m_requirements;

  int64_t m_output_scale_factor;

  bool m_is_ready;
};

#endif
