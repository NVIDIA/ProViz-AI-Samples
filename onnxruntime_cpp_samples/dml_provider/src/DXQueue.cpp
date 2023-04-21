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

#include "DXQueue.h"

DXQueue::DXQueue(ComPtr<ID3D12Device2> inDevice, D3D12_COMMAND_LIST_TYPE inType)
    : m_fence_value(0), m_cmd_list_type(inType), m_device(inDevice) {

  D3D12_COMMAND_QUEUE_DESC cqDsc = {};
  cqDsc.Type = inType;
  cqDsc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
  cqDsc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
  cqDsc.NodeMask = 0;

  ThrowOnFail(m_device->CreateCommandQueue(&cqDsc, IID_PPV_ARGS(&m_queue)));
  ThrowOnFail(m_device->CreateFence(m_fence_value, D3D12_FENCE_FLAG_NONE,
                                    IID_PPV_ARGS(&m_fence)));

  m_fence_event = ::CreateEvent(NULL, FALSE, FALSE, NULL);

  assert(m_fence_event && "Epic Fence Event Fail.");
}

DXQueue::~DXQueue() {}

ComPtr<ID3D12CommandAllocator> DXQueue::CreateCmdAllocator() {
  ComPtr<ID3D12CommandAllocator> cmdAlloc;
  ThrowOnFail(m_device->CreateCommandAllocator(m_cmd_list_type,
                                               IID_PPV_ARGS(&cmdAlloc)));
  return cmdAlloc;
}

ComPtr<ID3D12GraphicsCommandList2>
DXQueue::CreateCmdList(ComPtr<ID3D12CommandAllocator> inAlloc) {
  ComPtr<ID3D12GraphicsCommandList2> cmdList;
  ThrowOnFail(m_device->CreateCommandList(0, m_cmd_list_type, inAlloc.Get(),
                                          nullptr, IID_PPV_ARGS(&cmdList)));
  return cmdList;
}

ComPtr<ID3D12GraphicsCommandList2> DXQueue::GetCmdList() {
  ComPtr<ID3D12CommandAllocator> cmdAlloc;
  ComPtr<ID3D12GraphicsCommandList2> cmdList;

  if (!m_cmd_allocator_queue.empty() &&
      IsFenceComplete(m_cmd_allocator_queue.front().fence_value)) {
    cmdAlloc = m_cmd_allocator_queue.front().cmd_allocator;
    m_cmd_allocator_queue.pop();

    ThrowOnFail(cmdAlloc->Reset());
  } else {
    cmdAlloc = CreateCmdAllocator();
  }

  if (!m_cmd_list_queue.empty()) {
    cmdList = m_cmd_list_queue.front();
    m_cmd_list_queue.pop();

    ThrowOnFail(cmdList->Reset(cmdAlloc.Get(), nullptr));
  } else {
    cmdList = CreateCmdList(cmdAlloc);
  }

  ThrowOnFail(cmdList->SetPrivateDataInterface(__uuidof(ID3D12CommandAllocator),
                                               cmdAlloc.Get()));

  return cmdList;
}

uint64_t
DXQueue::ExecuteCommandList(ComPtr<ID3D12GraphicsCommandList2> inCmdList) {

  ID3D12CommandAllocator *cmdAlloc;
  UINT dataSize = sizeof(cmdAlloc);
  ThrowOnFail(inCmdList->GetPrivateData(__uuidof(ID3D12CommandAllocator),
                                        &dataSize, &cmdAlloc));

  inCmdList->Close();

  ID3D12CommandList *const ppCommandLists[] = {inCmdList.Get()};

  m_queue->ExecuteCommandLists(1, ppCommandLists);
  uint64_t fenceVal = Signal();
  m_cmd_allocator_queue.emplace(CmdAllocEntry{fenceVal, cmdAlloc});
  m_cmd_list_queue.push(inCmdList);

  cmdAlloc->Release();

  return fenceVal;
}

uint64_t DXQueue::Signal() {
  uint64_t fenceValueForSignal = ++m_fence_value;
  ThrowOnFail(m_queue->Signal(m_fence.Get(), fenceValueForSignal));
  return fenceValueForSignal;
}

bool DXQueue::IsFenceComplete(uint64_t inFenceValue) {
  return m_fence->GetCompletedValue() >= inFenceValue;
}

void DXQueue::WaitForFenceValue(uint64_t inFenceValue) {
  const std::chrono::milliseconds duration = std::chrono::milliseconds::max();
  if (!IsFenceComplete(inFenceValue)) {
    ThrowOnFail(m_fence->SetEventOnCompletion(inFenceValue, m_fence_event));
    ::WaitForSingleObject(m_fence_event, static_cast<DWORD>(duration.count()));
  }
}

void DXQueue::Flush() {
  uint64_t fenceValForSignal = Signal();
  WaitForFenceValue(fenceValForSignal);
}

ComPtr<ID3D12CommandQueue> DXQueue::GetD3D12CmdQueue() const { return m_queue; }