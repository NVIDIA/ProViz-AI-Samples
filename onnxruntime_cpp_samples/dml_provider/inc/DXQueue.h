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

#ifndef __H_DX_QUEUE__
#define __H_DX_QUEUE__

#pragma once

#include "common.h"

class DXQueue {
public:
  DXQueue(ComPtr<ID3D12Device2> inDevice, D3D12_COMMAND_LIST_TYPE inType);

  virtual ~DXQueue();

  ComPtr<ID3D12GraphicsCommandList2> GetCmdList();

  uint64_t ExecuteCommandList(ComPtr<ID3D12GraphicsCommandList2> inCmdList);

  uint64_t Signal();

  bool IsFenceComplete(uint64_t inFenceValue);
  void WaitForFenceValue(uint64_t inFenceValue);
  void Flush();

  ComPtr<ID3D12CommandQueue> GetD3D12CmdQueue() const;

protected:
  ComPtr<ID3D12CommandAllocator> CreateCmdAllocator();
  ComPtr<ID3D12GraphicsCommandList2>
  CreateCmdList(ComPtr<ID3D12CommandAllocator> inAllocator);

private:
  struct CmdAllocEntry {
    uint64_t fence_value;
    ComPtr<ID3D12CommandAllocator> cmd_allocator;
  };

  using CmdAllocatorQueue = std::queue<CmdAllocEntry>;
  using CmdListQueue = std::queue<ComPtr<ID3D12GraphicsCommandList2>>;

  D3D12_COMMAND_LIST_TYPE m_cmd_list_type;
  ComPtr<ID3D12Device2> m_device;
  ComPtr<ID3D12CommandQueue> m_queue;
  ComPtr<ID3D12Fence> m_fence;
  HANDLE m_fence_event;
  uint64_t m_fence_value;

  CmdAllocatorQueue m_cmd_allocator_queue;
  CmdListQueue m_cmd_list_queue;
};

#endif
