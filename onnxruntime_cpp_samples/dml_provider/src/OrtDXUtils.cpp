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

#include <Windows.h>

#include "OrtDXUtils.h"
#include "OrtBuffer.h"

ComPtr<IDXGIAdapter4> GetDXGIAdapter(bool useWarp) {

  /*
          We start by creating a ComPtr to an IDXGIFactory interface.
          We will use this to create the DXGI (DirectX Graphics
          Interface) objects that we need.
  */
  ComPtr<IDXGIFactory4> dxgiFactory;
  UINT createFactoryFlags = 0;

#if defined(_DEBUG)
  createFactoryFlags = DXGI_CREATE_FACTORY_DEBUG;
#endif

  ThrowOnFail(
      CreateDXGIFactory2(createFactoryFlags, IID_PPV_ARGS(&dxgiFactory)));

  /*
          We also need ComPtrs to 2 IDXGIAdapter interfaces.
          We will use the first adapter, version 1, (IDXGIAdapter1)
          to query all the adapters  that are available to us. Any
          adapters that are suitable to our needs will be converted
          to version 4 adapter (IDXGIAdapter4) which supports the
          features that we need.
  */

  ComPtr<IDXGIAdapter1> dxgiAdapter1;
  ComPtr<IDXGIAdapter4> dxgiAdapter4;

  if (useWarp) {
    ThrowOnFail(dxgiFactory->EnumWarpAdapter(IID_PPV_ARGS(&dxgiAdapter1)));
    ThrowOnFail(dxgiAdapter1.As(&dxgiAdapter4));
  } else {

    /*
            Next we will enumerate all of the adapters connected to our
            system. We would like to find the best adapter for our needs,
            in this case we will find the adapter that has the most dedicated
            video memory available to it.
    */

    size_t maxDedicatedVideoMemory = 0;

    for (UINT i = 0;
         dxgiFactory->EnumAdapters1(i, &dxgiAdapter1) != DXGI_ERROR_NOT_FOUND;
         ++i) {

      /*
              For each adapter that we find as we call
         dxgiFactory->EnumAdapter1(i,&dgxiAdapter) we will get the corresponding
         descriptor object that we can query for the capabilities.
      */
      DXGI_ADAPTER_DESC1 adapterDesc1;
      dxgiAdapter1->GetDesc1(&adapterDesc1);

      if ((adapterDesc1.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0 &&
          SUCCEEDED(D3D12CreateDevice(dxgiAdapter1.Get(),
                                      D3D_FEATURE_LEVEL_11_0,
                                      __uuidof(ID3D12Device), nullptr)) &&
          adapterDesc1.DedicatedVideoMemory > maxDedicatedVideoMemory) {
        maxDedicatedVideoMemory = adapterDesc1.DedicatedVideoMemory;
        ThrowOnFail(dxgiAdapter1.As(&dxgiAdapter4));
      }
    }
  }

  return dxgiAdapter4;
}

ComPtr<ID3D12Device2> CreateDXDevice(ComPtr<IDXGIAdapter4> inAdapter) {

  ComPtr<ID3D12Device2> d3d12Device2;

  ThrowOnFail(D3D12CreateDevice(inAdapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                IID_PPV_ARGS(&d3d12Device2)));

#if defined(_DEBUG)

  ComPtr<ID3D12InfoQueue> pInfoQueue;
  if (SUCCEEDED(d3d12Device2.As(&pInfoQueue))) {
    pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
    pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
    pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, TRUE);

    D3D12_MESSAGE_SEVERITY Severities[] = {D3D12_MESSAGE_SEVERITY_INFO};

    D3D12_MESSAGE_ID DenyIds[] = {
        D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE,
        D3D12_MESSAGE_ID_MAP_INVALID_NULLRANGE,
        D3D12_MESSAGE_ID_UNMAP_INVALID_NULLRANGE};

    D3D12_INFO_QUEUE_FILTER NewFilter = {};
    NewFilter.DenyList.NumSeverities = _countof(Severities);
    NewFilter.DenyList.pSeverityList = Severities;
    NewFilter.DenyList.NumIDs = _countof(DenyIds);
    NewFilter.DenyList.pIDList = DenyIds;

    ThrowOnFail(pInfoQueue->PushStorageFilter(&NewFilter));
  }

#endif

  return d3d12Device2;
}

void EnableDebugLayer() {

#if defined(_DEBUG)
  ComPtr<ID3D12Debug> debugInterface;
  ThrowOnFail(D3D12GetDebugInterface(IID_PPV_ARGS(&debugInterface)));
  debugInterface->EnableDebugLayer();
#endif
}

ComPtr<ID3D12DescriptorHeap> CreateDescriptorHeap(
    ComPtr<ID3D12Device2> inDevice, D3D12_DESCRIPTOR_HEAP_TYPE inType,
    uint32_t inNumDescriptors, D3D12_DESCRIPTOR_HEAP_FLAGS inFlags) {

  ComPtr<ID3D12DescriptorHeap> descriptorHeap;
  D3D12_DESCRIPTOR_HEAP_DESC desc = {};

  desc.NumDescriptors = inNumDescriptors;
  desc.Type = inType;
  desc.Flags = inFlags;

  ThrowOnFail(
      inDevice->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&descriptorHeap)));
  return descriptorHeap;
}

void ConvertToFloat(uint8_t *inInput, float *inOutput,
                    const size_t inElementCount) {

  for (size_t i = 0; i < inElementCount; ++i) {

    int input_channel = i % 3;
    unsigned int slice_size = 512 * 512;
    unsigned int output_idx = (i / 3) + (input_channel * slice_size);

    inOutput[output_idx] = ((float)inInput[i]) / 255.0 - 0.5;
  }
}

void ConvertToUint8(float *inInput, uint8_t *inOutput,
                    const size_t inElementCount) {

  float out_min = FLT_MAX;
  float out_max = FLT_MIN;

  for (size_t i = 0; i < inElementCount; ++i) {

    int input_channel = i % 3;
    unsigned int slice_size = 512 * 512;

    unsigned int input_idx = (i / 3) + (input_channel * slice_size);

    float input_val = inInput[input_idx];

    if (input_val > out_max)
      out_max = input_val;
    if (input_val < out_min)
      out_min = input_val;

    if (input_val < -0.5f)
      input_val = -0.5f;

    if (input_val > 0.5f)
      input_val = 0.5f;

    inOutput[i] = (uint8_t)((input_val + 0.5) * 255.0);
  }

  LOGI("MIN VAL : {}" << out_min);
  LOGI("MAX VAL : {}" << out_max);
}

void CreateTransformingRootSignature(ComPtr<ID3D12Device2> inDevice,
                                     ComPtr<ID3D12RootSignature> &outRootSig,
                                     ComPtr<ID3D12DescriptorHeap> &outHeap) {

  /*
          Create the root signature.
          - Create a descriptor range for the UAV Table.
          - Contains 2 descriptors.
          - Create one root parameter for the descriptor table.
          - Create a root signature descriptor.
  */
  CD3DX12_DESCRIPTOR_RANGE uavTable;
  uavTable.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 0);
  CD3DX12_ROOT_PARAMETER slotParams[2];
  slotParams[0].InitAsDescriptorTable(1, &uavTable);
  slotParams[1].InitAsConstants(4, 0);

  CD3DX12_ROOT_SIGNATURE_DESC rootSigDesc(2, slotParams, 0, nullptr);
  ComPtr<ID3DBlob> serializedRootSig = nullptr;
  ComPtr<ID3DBlob> errorBlob = nullptr;

  /*
  Serialize the root signature.
  */
  HRESULT hr = D3D12SerializeRootSignature(
      &rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1,
      serializedRootSig.GetAddressOf(), errorBlob.GetAddressOf());

  if (nullptr != errorBlob) {
    ::OutputDebugStringA((char *)errorBlob->GetBufferPointer());
  }

  ThrowOnFail(hr);

  ThrowOnFail(inDevice->CreateRootSignature(
      0, serializedRootSig->GetBufferPointer(),
      serializedRootSig->GetBufferSize(), IID_PPV_ARGS(&outRootSig)));

  outHeap =
      ::CreateDescriptorHeap(inDevice, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                             2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);
}

E2EError GetShaderForDataType(uint32_t inDataType, const wchar_t *inFP16Path,
                              const wchar_t *inFP32Path,
                              ComPtr<ID3DBlob> &outBlob) {
  switch (inDataType) {
  case OrtBuffer::DataType::ORT_BUFFER_FLOAT16:
    ThrowOnFail(D3DReadFileToBlob(inFP16Path, &outBlob));
    break;
  case OrtBuffer::DataType::ORT_BUFFER_FLOAT32:
    ThrowOnFail(D3DReadFileToBlob(inFP32Path, &outBlob));
    break;

  default:
    return UNSUPPORTED_BINDING_DATA_TYPE;
    break;
  }

  return NO_E2E_ERROR;
}

E2EError CreatePSO(ComPtr<ID3D12Device2> inDevice,
                   ComPtr<ID3D12RootSignature> inRootSignature,
                   ComPtr<ID3DBlob> inShader,
                   ComPtr<ID3D12PipelineState> &outPso) {

  /*
          Create the compute pipeline state descriptor.
          - assign root signature
          - assign the compute shader
  */
  D3D12_COMPUTE_PIPELINE_STATE_DESC cpDesc = {};
  cpDesc.pRootSignature = inRootSignature.Get();
  cpDesc.CS = CD3DX12_SHADER_BYTECODE(inShader.Get());
  cpDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
  cpDesc.NodeMask = 0;

  /*
          Create the compute pipeline state.
  */

  ThrowOnFail(
    inDevice->CreateComputePipelineState(&cpDesc, IID_PPV_ARGS(&outPso)));

  return NO_E2E_ERROR;
}