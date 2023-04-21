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

#ifndef __H_ORT_DX_UTILS__
#define __H_ORT_DX_UTILS__

#pragma once

#include "Common.h"

void EnableDebugLayer();

void ConvertToFloat(uint8_t *inInput, float *inOutput,
                    const size_t inElementCount);
void ConvertToUint8(float *inInput, uint8_t *inOutput,
                    const size_t inElementCount);

ComPtr<IDXGIAdapter4> GetDXGIAdapter(bool useWarp = false);

ComPtr<ID3D12Device2> CreateDXDevice(ComPtr<IDXGIAdapter4> inAdapter);

ComPtr<ID3D12DescriptorHeap> CreateDescriptorHeap(
    ComPtr<ID3D12Device2> inDevice, D3D12_DESCRIPTOR_HEAP_TYPE inType,
    uint32_t inNumDescriptors,
    D3D12_DESCRIPTOR_HEAP_FLAGS inFlags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE);

void CreateTransformingRootSignature(ComPtr<ID3D12Device2> inDevice,
                                     ComPtr<ID3D12RootSignature> &outRootSig,
                                     ComPtr<ID3D12DescriptorHeap> &outHeap);

E2EError GetShaderForDataType(uint32_t inDataType, const wchar_t *inFP16Path,
                              const wchar_t *inFP32Path,
                              ComPtr<ID3DBlob> &outBlob);

E2EError CreatePSO(ComPtr<ID3D12Device2> inDevice,
                   ComPtr<ID3D12RootSignature> inRootSignature,
                   ComPtr<ID3DBlob> inShader,
                   ComPtr<ID3D12PipelineState> &outPso);

#endif