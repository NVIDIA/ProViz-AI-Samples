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

#include "OrtInference.h"
#include "OrtDXUtils.h"
#include "onnxruntime_c_api.h"
#include <dml_provider_factory.h>

#include "OrtBuffer.h"

using UniqueOrtAllocator =
    std::unique_ptr<OrtAllocator, decltype(OrtApi::ReleaseAllocator)>;

OrtInference::OrtInference() : m_is_ready(false) {}

OrtInference::~OrtInference() {}

E2EError OrtInference::InitD3DDevice() {

  E2EError out_err = NO_E2E_ERROR;

  EnableDebugLayer();

  m_adapter = GetDXGIAdapter();
  m_device = CreateDXDevice(m_adapter);

  // Create the query heap..
  D3D12_QUERY_HEAP_DESC queryHeapDesc = {};
  queryHeapDesc.Count = 16;
  queryHeapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
  ThrowOnFail(m_device->CreateQueryHeap(&queryHeapDesc,
                                        IID_PPV_ARGS(&m_timestamp_query_heap)));

  CD3DX12_HEAP_PROPERTIES heapProps =
      CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
  CD3DX12_RESOURCE_DESC resDesc =
      CD3DX12_RESOURCE_DESC::Buffer(16 * sizeof(int64_t));

  ThrowOnFail(m_device->CreateCommittedResource(
      &heapProps, D3D12_HEAP_FLAG_NONE, &resDesc,
      D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
      IID_PPV_ARGS(&m_timestamp_readback)));

  DML_CREATE_DEVICE_FLAGS dml_device_flags = DML_CREATE_DEVICE_FLAG_NONE;

#if defined(_DEBUG)
  // If the project is in a debug build, then enable debugging via DirectML
  // debug layers with this flag.
  dml_device_flags |= DML_CREATE_DEVICE_FLAG_DEBUG;
#endif

  ThrowOnFail(DMLCreateDevice(m_device.Get(), dml_device_flags,
                              IID_PPV_ARGS(&m_dml_device)));

  m_copy_queue =
      std::make_shared<DXQueue>(m_device, D3D12_COMMAND_LIST_TYPE_DIRECT);
  m_compute_queue =
      std::make_shared<DXQueue>(m_device, D3D12_COMMAND_LIST_TYPE_COMPUTE);

  // Create the memory info object of type DML.
  m_memory_info = std::shared_ptr<Ort::MemoryInfo>(
      new Ort::MemoryInfo("DML", OrtAllocatorType::OrtDeviceAllocator, 0,
                          OrtMemType::OrtMemTypeDefault));

  return out_err;
}

E2EError OrtInference::InitInference(EnginePath &inEnginePath,
                                     EnginePath &inEnginePath2) {

  E2EError out_err = NO_E2E_ERROR;

  m_output_scale_factor = 2;

  m_engine_path = inEnginePath;
  m_engine_path2 = inEnginePath2;

  out_err = InitD3DDevice();
  if (out_err != NO_E2E_ERROR) {
    return out_err;
  }

  out_err = LoadEngine();
  if (out_err != NO_E2E_ERROR) {
    return out_err;
  }

  return out_err;
}

E2EError OrtInference::LoadEngine() {
  OrtApi const &ortApi = Ort::GetApi();

  E2EError out_err = NO_E2E_ERROR;
  // Using unique pointers here to be consistent with the Trt Engine Approach.
  // No logging yet.
  // Session can be created with logging options and logger ID.
  m_env = std::unique_ptr<Ort::Env>(new Ort::Env());

  // Convert Engine Path to WString for the wchar_t
  wstring w_path = wstring(m_engine_path.begin(), m_engine_path.end());
  wstring w_path2 = wstring(m_engine_path2.begin(), m_engine_path2.end());

  // Create the session options..
  Ort::SessionOptions opts;
  opts.DisableMemPattern();

  ComPtr<ID3D12Device> d3d_device;
  ThrowOnFail(m_device.As(&d3d_device));
  OrtSessionOptionsAppendExecutionProviderEx_DML(
      opts, m_dml_device.Get(), m_copy_queue->GetD3D12CmdQueue().Get());

  ortApi.AddFreeDimensionOverrideByName((OrtSessionOptions *)opts,
                                        "image_input_N", (int64_t)1);
  ortApi.AddFreeDimensionOverrideByName((OrtSessionOptions *)opts,
                                        "image_input_C", (int64_t)8);
  ortApi.AddFreeDimensionOverrideByName((OrtSessionOptions *)opts,
                                        "image_input_H",
                                        (int64_t)m_input_image.height);
  ortApi.AddFreeDimensionOverrideByName(
      (OrtSessionOptions *)opts, "image_input_W", (int64_t)m_input_image.width);

  // Create the session.
  m_session = std::unique_ptr<Ort::Session>(
      new Ort::Session(*(m_env.get()), w_path.c_str(), opts));

  ortApi.AddFreeDimensionOverrideByName((OrtSessionOptions *)opts, "input_N",
                                        (int64_t)1);
  ortApi.AddFreeDimensionOverrideByName((OrtSessionOptions *)opts, "input_C",
                                        (int64_t)8);
  ortApi.AddFreeDimensionOverrideByName((OrtSessionOptions *)opts, "input_H",
                                        (int64_t)m_input_image.height);
  ortApi.AddFreeDimensionOverrideByName((OrtSessionOptions *)opts, "input_W",
                                        (int64_t)m_input_image.width);

  m_session_2 = std::unique_ptr<Ort::Session>(
      new Ort::Session(*(m_env.get()), w_path2.c_str(), opts));

  opts.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

  return QuerySessionRequirements();
}

E2EError OrtInference::QuerySessionRequirements() {
  E2EError out_err = NO_E2E_ERROR;

  m_requirements.input_count = m_session->GetInputCount();
  m_requirements.output_count = m_session->GetOutputCount();
  m_requirements.overridable_init_count =
      m_session->GetOverridableInitializerCount();

  LOGI("Session Requirements           : ");
  LOGI("    Input Count                : " << m_requirements.input_count);
  LOGI("    Output Count               : " << m_requirements.output_count);
  LOGI("    Overridable Init Count     : "
       << m_requirements.overridable_init_count);

  Ort::AllocatorWithDefaultOptions defaultAlloc =
      Ort::AllocatorWithDefaultOptions();

  for (size_t i = 0; i < m_requirements.input_count; ++i) {
    BindingMetaData this_binding_data = {};

    auto this_binding_name =
        m_session->GetInputNameAllocated(i, (OrtAllocator *)defaultAlloc);
    // this_binding_data.name = BindingName(this_binding_name);

    LOGI("      Input  Binding                : " << this_binding_name);

    Ort::TypeInfo typeInfo = m_session->GetInputTypeInfo(i);
    Ort::Unowned<Ort::TensorTypeAndShapeInfo> shape_info =
        typeInfo.GetTensorTypeAndShapeInfo();
    this_binding_data.shape = shape_info.GetShape();
    this_binding_data.data_type = shape_info.GetElementType();

    m_requirements.input_bindings[BindingName(this_binding_name.get())] =
        this_binding_data;
  }

  for (size_t i = 0; i < m_requirements.output_count; ++i) {
    BindingMetaData this_binding_data = {};

    auto this_binding_name =
        m_session->GetOutputNameAllocated(i, (OrtAllocator *)defaultAlloc);
    // this_binding_data.name = BindingName(this_binding_name);

    LOGI("      Output Binding                : " << this_binding_name);

    Ort::TypeInfo typeInfo = m_session->GetOutputTypeInfo(i);
    Ort::Unowned<Ort::TensorTypeAndShapeInfo> shape_info =
        typeInfo.GetTensorTypeAndShapeInfo();
    this_binding_data.shape = shape_info.GetShape();
    this_binding_data.data_type = shape_info.GetElementType();

    m_requirements.output_bindings[BindingName(this_binding_name.get())] =
        this_binding_data;
  }

  return out_err;
}

E2EError OrtInference::LoadInput(const char *inPath) {
  E2EError out_err = NO_E2E_ERROR;

  // TODO - Handle Errors.
  LoadPNG(inPath, &m_input_image);

  return out_err;
}

E2EError OrtInference::InitBindings() {
  E2EError out_err = NO_E2E_ERROR;

  // Create the binding object for this session.
  m_io_binding =
      std::unique_ptr<Ort::IoBinding>(new Ort::IoBinding(*(m_session.get())));
  m_io_binding2 =
      std::unique_ptr<Ort::IoBinding>(new Ort::IoBinding(*(m_session_2.get())));

  OrtValue *ort_input_val = m_ort_input_buffer->GetOrtValue();
  OrtValue *ort_output_val = m_ort_output_buffer->GetOrtValue();

  m_io_binding->BindInput(GetInputBindingName().c_str(),
                          (Ort::Value &)ort_input_val);
  m_io_binding->BindOutput(GetOutputBindingName().c_str(),
                           (Ort::Value &)ort_input_val);

  m_io_binding2->BindInput("input", (Ort::Value &)ort_input_val);
  m_io_binding2->BindOutput("out_element_wise_sum",
                            (Ort::Value &)ort_output_val);

  return out_err;
}

E2EError OrtInference::InitPostProcessCommands() {

  E2EError out_err = NO_E2E_ERROR;
  int64_t wh_dim = m_input_image.height * m_output_scale_factor;
  int64_t dml_shape_uint[] = {1, m_input_image.channels, wh_dim, wh_dim};
  int64_t dml_shape_float[] = {1, 8, wh_dim, wh_dim};

  m_cmd_list_stage_output = m_copy_queue->GetCmdList();

  ONNXTensorElementDataType output_data_type = GetInputDataType();

  OrtBuffer::DataType buffer_data_type =
      OrtBuffer::ONNXToInternalDataType(output_data_type);

  if (buffer_data_type == OrtBuffer::DataType::ORT_BUFFER_TYPE_UNDEFINED) {
    return UNSUPPORTED_BINDING_DATA_TYPE;
  }

  m_img_output_buffer = new OrtBuffer();
  m_img_output_buffer->InitBuffer(OrtBuffer::DataType::ORT_BUFFER_UINT8, 4,
                                  dml_shape_uint, this);

  CD3DX12_RESOURCE_BARRIER uav_barrier1 = CD3DX12_RESOURCE_BARRIER::UAV(
      m_img_output_buffer->GetD3DResource().Get());

  m_cmd_list_stage_output->ResourceBarrier(1, &uav_barrier1);

  m_ort_output_buffer = new OrtBuffer();
  m_ort_output_buffer->InitBuffer(buffer_data_type, 4, dml_shape_float, this);

  CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
      m_ort_output_buffer->GetD3DResource().Get(),
      D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

  m_cmd_list_stage_output->ResourceBarrier(1, &barrier);

  CD3DX12_RESOURCE_BARRIER uav_barrier2 = CD3DX12_RESOURCE_BARRIER::UAV(
      m_ort_output_buffer->GetD3DResource().Get());

  m_cmd_list_stage_output->ResourceBarrier(1, &uav_barrier2);

  D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc_input =
      m_ort_output_buffer->GetUAVDesc();
  D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc_output =
      m_img_output_buffer->GetUAVDesc();

  // Create a root signature that transforms one lot
  // of input into one lot of output. Also creates heap.
  CreateTransformingRootSignature(m_device, m_post_process_root_signature,
                                  m_post_process_heap);

  // Create the UAV for the Input buffer.
  D3D12_CPU_DESCRIPTOR_HANDLE hdl =
      m_post_process_heap->GetCPUDescriptorHandleForHeapStart();
  m_device->CreateUnorderedAccessView(
      m_ort_output_buffer->GetD3DResource().Get(), nullptr, &uavDesc_input,
      hdl);
  hdl.ptr += m_uav_descriptor_size;

  // Create the UAV for the Output buffer.
  m_device->CreateUnorderedAccessView(
      m_img_output_buffer->GetD3DResource().Get(), nullptr, &uavDesc_output,
      hdl);
  hdl.ptr += m_uav_descriptor_size;

  /*
  Load the shader.
  */
  ComPtr<ID3DBlob> computeShaderBlob;
  out_err = GetShaderForDataType(
      buffer_data_type, L"dml_provider/shader/postprocess_scale_nchw_to_nhwc_fp16.cso",
      L"dml_provider/shader/postprocess_scale_nchw_to_nhwc_fp32.cso", computeShaderBlob);
  CreatePSO(m_device, m_post_process_root_signature, computeShaderBlob,
            m_post_process_pso);

  m_cmd_list_postprocess_output = m_compute_queue->GetCmdList();

  /*
  Create a list of descriptor heaps (1)
  to bind to the pipeline.
  */
  ID3D12DescriptorHeap *heaps[] = {m_post_process_heap.Get()};

  /*
          Bind the descriptor heaps.
  */
  m_cmd_list_postprocess_output->SetDescriptorHeaps(1, heaps);
  m_cmd_list_postprocess_output->SetPipelineState(m_post_process_pso.Get());
  m_cmd_list_postprocess_output->SetComputeRootSignature(
      m_post_process_root_signature.Get());
  m_cmd_list_postprocess_output->SetComputeRootDescriptorTable(
      0, m_post_process_heap->GetGPUDescriptorHandleForHeapStart());
  m_cmd_list_postprocess_output->SetComputeRoot32BitConstant(
      1, (wh_dim * wh_dim), 0);
  m_cmd_list_postprocess_output->SetComputeRoot32BitConstant(1, m_input_image.channels, 1);
  m_cmd_list_postprocess_output->SetComputeRoot32BitConstant(1, m_input_image.channels, 2);
  m_cmd_list_postprocess_output->SetComputeRoot32BitConstant(1, 0, 3);

  /*
          Dispatch the kernel.
  */

  size_t total_threads = 1 * m_input_image.channels * wh_dim * wh_dim;
  size_t group_size = 256;
  size_t group_count = (total_threads + group_size - 1) / group_size;

  m_cmd_list_postprocess_output->EndQuery(m_timestamp_query_heap.Get(),
                                          D3D12_QUERY_TYPE_TIMESTAMP, 2);

  m_cmd_list_postprocess_output->Dispatch(group_count, 1, 1);

  m_cmd_list_postprocess_output->EndQuery(m_timestamp_query_heap.Get(),
                                          D3D12_QUERY_TYPE_TIMESTAMP, 3);

  m_cmd_list_postprocess_output->ResolveQueryData(
      m_timestamp_query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2, 2,
      m_timestamp_readback.Get(), 16);

  return out_err;
}

E2EError OrtInference::InitPreprocessCommands() {
  E2EError out_err = NO_E2E_ERROR;
  m_cmd_list_stage_input = m_copy_queue->GetCmdList();

  int64_t dml_shape_uint[] = {1, m_input_image.channels, m_input_image.height, m_input_image.width};
  int64_t dml_shape_float[] = {1, 8, m_input_image.height, m_input_image.width};

  ONNXTensorElementDataType input_data_type = GetInputDataType();

  OrtBuffer::DataType buffer_data_type =
      OrtBuffer::ONNXToInternalDataType(input_data_type);

  if (buffer_data_type == OrtBuffer::DataType::ORT_BUFFER_TYPE_UNDEFINED) {
    return UNSUPPORTED_BINDING_DATA_TYPE;
  }

  m_img_input_buffer = new OrtBuffer();
  m_img_input_buffer->InitBufferWithData(
      OrtBuffer::DataType::ORT_BUFFER_UINT8, 4, dml_shape_uint,
      m_input_image.raw_data, this, m_cmd_list_stage_input);

  CD3DX12_RESOURCE_BARRIER uav_barrier1 =
      CD3DX12_RESOURCE_BARRIER::UAV(m_img_input_buffer->GetD3DResource().Get());

  m_cmd_list_stage_input->ResourceBarrier(1, &uav_barrier1);

  m_ort_input_buffer = new OrtBuffer();
  m_ort_input_buffer->InitBuffer(buffer_data_type, 4, dml_shape_float, this);

  CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
      m_ort_input_buffer->GetD3DResource().Get(),
      D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

  m_cmd_list_stage_input->ResourceBarrier(1, &barrier);

  CD3DX12_RESOURCE_BARRIER uav_barrier2 =
      CD3DX12_RESOURCE_BARRIER::UAV(m_ort_input_buffer->GetD3DResource().Get());

  m_cmd_list_stage_input->ResourceBarrier(1, &uav_barrier2);

  // Now we have a buffer that contains the original
  // image data on the GPU as uint8 NHWC.

  // Get UAV Descriptor size.
  m_uav_descriptor_size = m_device->GetDescriptorHandleIncrementSize(
      D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

  /*
  Create an unordered access view.
  - Format unknown
  - Buffer
  */
  D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc_input =
      m_img_input_buffer->GetUAVDesc();
  D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc_output =
      m_ort_input_buffer->GetUAVDesc();

  // Create a root signature that transforms one lot
  // of input into one lot of output. Also creates heap.
  CreateTransformingRootSignature(m_device, m_pre_process_root_signature,
                                  m_pre_process_heap);

  /*
          Create the UAV for the Input buffer.
  */
  D3D12_CPU_DESCRIPTOR_HANDLE hdl =
      m_pre_process_heap->GetCPUDescriptorHandleForHeapStart();
  m_device->CreateUnorderedAccessView(
      m_img_input_buffer->GetD3DResource().Get(), nullptr, &uavDesc_input, hdl);
  hdl.ptr += m_uav_descriptor_size;

  /*
          Create the UAV for the Output buffer.
  */
  m_device->CreateUnorderedAccessView(
      m_ort_input_buffer->GetD3DResource().Get(), nullptr, &uavDesc_output,
      hdl);
  hdl.ptr += m_uav_descriptor_size;

  /*
          Load the shader.
  */
  ComPtr<ID3DBlob> computeShaderBlob;
  out_err = GetShaderForDataType(
      buffer_data_type, L"dml_provider/shader/preprocess_scale_nhwc_to_nchw_fp16.cso",
      L"dml_provider/shader/preprocess_scale_nhwc_to_nchw_fp32.cso", computeShaderBlob);

  CreatePSO(m_device, m_pre_process_root_signature, computeShaderBlob,
            m_pre_process_pso);
  m_cmd_list_preprocess_input = m_compute_queue->GetCmdList();

  /*
  Create a list of descriptor heaps (1)
  to bind to the pipeline.
  */
  ID3D12DescriptorHeap *heaps[] = {m_pre_process_heap.Get()};

  m_cmd_list_preprocess_input->SetDescriptorHeaps(1, heaps);
  m_cmd_list_preprocess_input->SetPipelineState(m_pre_process_pso.Get());
  m_cmd_list_preprocess_input->SetComputeRootSignature(
      m_pre_process_root_signature.Get());
  m_cmd_list_preprocess_input->SetComputeRootDescriptorTable(
      0, m_pre_process_heap->GetGPUDescriptorHandleForHeapStart());
  m_cmd_list_preprocess_input->SetComputeRoot32BitConstant(
      1, (m_input_image.height * m_input_image.width), 0);
  m_cmd_list_preprocess_input->SetComputeRoot32BitConstant(1, m_input_image.channels, 1);
  m_cmd_list_preprocess_input->SetComputeRoot32BitConstant(1, 8, 2);
  m_cmd_list_preprocess_input->SetComputeRoot32BitConstant(1, 0, 3);

  /*
          Dispatch the kernel.
  */
  size_t total_threads = 1 * 8 * m_input_image.height * m_input_image.width;
  size_t group_size = 256;
  size_t group_count = (total_threads + group_size - 1) / group_size;

  m_cmd_list_preprocess_input->EndQuery(m_timestamp_query_heap.Get(),
                                        D3D12_QUERY_TYPE_TIMESTAMP, 0);

  m_cmd_list_preprocess_input->Dispatch(group_count, 1, 1);

  m_cmd_list_preprocess_input->EndQuery(m_timestamp_query_heap.Get(),
                                        D3D12_QUERY_TYPE_TIMESTAMP, 1);

  m_cmd_list_preprocess_input->ResolveQueryData(
      m_timestamp_query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2,
      m_timestamp_readback.Get(), 0);

  return out_err;
}

E2EError OrtInference::UpdateBufferResource(
    ComPtr<ID3D12GraphicsCommandList2> inCmdList,
    ID3D12Resource **inDestResource, ID3D12Resource **inIntermediateResource,
    size_t inElementCount, size_t inElementSize, const void *inBufferData,
    D3D12_RESOURCE_FLAGS inFlags) {

  E2EError out_err = NO_E2E_ERROR;

  // Calculate the buffer size
  size_t buffer_size = inElementCount * inElementSize;

  // Create the heap props and the resource descriptor
  D3D12_HEAP_PROPERTIES default_props =
      CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
  D3D12_RESOURCE_DESC buffer_desc =
      CD3DX12_RESOURCE_DESC::Buffer(buffer_size, inFlags);

  ThrowOnFail(m_device->CreateCommittedResource(
      &default_props, D3D12_HEAP_FLAG_NONE, &buffer_desc,
      D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(inDestResource)));

  // If we passed in some data to stage then we need
  // to initialize the intermediate resource to be used for the staging.
  if (inBufferData) {

    D3D12_HEAP_PROPERTIES upload_props =
        CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    D3D12_RESOURCE_DESC buffer_desc2 =
        CD3DX12_RESOURCE_DESC::Buffer(buffer_size);

    ThrowOnFail(m_device->CreateCommittedResource(
        &upload_props, D3D12_HEAP_FLAG_NONE, &buffer_desc2,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(inIntermediateResource)));

    D3D12_SUBRESOURCE_DATA subResData = {};
    subResData.pData = inBufferData;
    subResData.RowPitch = buffer_size;
    subResData.SlicePitch = buffer_size;

    UpdateSubresources(inCmdList.Get(), *inDestResource,
                       *inIntermediateResource, 0, 0, 1, &subResData);
  }

  return out_err;
}

E2EError OrtInference::Run(void **inBuffers) {
  E2EError out_err = NO_E2E_ERROR;
  Ort::RunOptions run_options;

  // Transfer the input data to the GPU
  // Also performs the necessary state changes
  // for the m_ort_input_buffer

  m_copy_queue->ExecuteCommandList(m_cmd_list_stage_input);
  // No data to transfer here.
  // but does perform the necessary state changes
  // for the m_ort_output_buffer.
  m_copy_queue->ExecuteCommandList(m_cmd_list_stage_output);
  m_copy_queue->Flush();

    // UINT64 gpuTimestampBegin;
    // UINT64 cpuTimestampBegin;
    // UINT64 gpuTimestampEnd;
    // UINT64 cpuTimestampEnd;

    // m_compute_queue->GetD3D12CmdQueue()->GetClockCalibration(&gpuTimestampBegin,
    // &cpuTimestampBegin); Perform preprocessing on the input data.
    m_compute_queue->ExecuteCommandList(m_cmd_list_preprocess_input);
    m_compute_queue->Flush();
    // m_compute_queue->GetD3D12CmdQueue()->GetClockCalibration(&gpuTimestampEnd,
    // &cpuTimestampEnd);

    size_t data_size = 16 * sizeof(int64_t);
    D3D12_RANGE buf_range{0, data_size};
    int64_t *mapped = nullptr;

    m_session->Run(run_options, *(m_io_binding.get()));
    m_session_2->Run(run_options, *(m_io_binding2.get()));
    m_copy_queue->Flush();

    // post process the output data.
    m_compute_queue->ExecuteCommandList(m_cmd_list_postprocess_output);
    m_compute_queue->Flush();

    ThrowOnFail(m_timestamp_readback->Map(0, &buf_range,
                                          reinterpret_cast<void **>(&mapped)));

    LOGI("Preprocessing Pass Compute Time : "
         << ((double)mapped[1] - (double)mapped[0]) / (double)1000000 << "ms");
    LOGI("Postprocessing Pass Compute Time : "
         << ((double)mapped[3] - (double)mapped[2]) / (double)1000000 << "ms");

    // LOGI("Time In Ticks : CPU : " << (cpuTimestampEnd - cpuTimestampBegin) <<
    // " GPU : " << (gpuTimestampEnd - gpuTimestampBegin));

    // Run the model

  return out_err;
}

E2EError OrtInference::SaveResult(const char *inPath) {
  E2EError out_err = NO_E2E_ERROR;

  // Read back the post processed data
  m_img_output_buffer->ReadBack();

  // Save as a PNG.
  m_input_image.raw_data = m_img_output_buffer->MapUint8();
  m_input_image.width *= m_output_scale_factor;
  m_input_image.height *= m_output_scale_factor;

  SavePNG(inPath, m_input_image);

  return out_err;
}

std::shared_ptr<DXQueue> OrtInference::GetCopyQueue() { return m_copy_queue; }

std::shared_ptr<DXQueue> OrtInference::GetComputeQueue() {
  return m_compute_queue;
}

std::shared_ptr<Ort::MemoryInfo> OrtInference::GetMemoryInfo() {
  return m_memory_info;
}

ComPtr<ID3D12Device2> OrtInference::GetD3DDevice() { return m_device; }

ONNXTensorElementDataType OrtInference::GetInputDataType() {
  BindingMetaDataArray::iterator itr = m_requirements.input_bindings.begin();
  if (itr == m_requirements.input_bindings.end())
    return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

  BindingMetaData meta_data = itr->second;
  return meta_data.data_type;
}

ONNXTensorElementDataType OrtInference::GetOutputDataType() {
  BindingMetaDataArray::iterator itr = m_requirements.output_bindings.begin();
  if (itr == m_requirements.output_bindings.end())
    return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

  BindingMetaData meta_data = itr->second;
  return meta_data.data_type;
}

BindingName OrtInference::GetInputBindingName() {
  BindingMetaDataArray::iterator itr = m_requirements.input_bindings.begin();
  if (itr == m_requirements.input_bindings.end())
    return BindingName("none");
  BindingName out_name = itr->first;
  return out_name;
}

BindingName OrtInference::GetOutputBindingName() {
  BindingMetaDataArray::iterator itr = m_requirements.output_bindings.begin();
  if (itr == m_requirements.output_bindings.end())
    return BindingName("none");
  BindingName out_name = itr->first;
  return out_name;
}