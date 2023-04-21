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

#ifndef __H_COMMON__
#define __H_COMMON__

#pragma once

#include <Windows.h>
#include <shellapi.h>

#include "png.h"
#include <iostream>
#include <map>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <queue>
#include <string>

// DX12 Headers
#include "d3dx12.h"
#include <DirectXMath.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl.h>

// stl
#include <algorithm>
#include <cassert>
#include <chrono>

#if defined(min)
#undef min
#endif

#if defined(max)
#undef max
#endif

inline void ThrowOnFail(HRESULT hr) {
  if (FAILED(hr)) {
    throw std::exception();
  }
}

using namespace Microsoft::WRL;
using namespace DirectX;

typedef enum _E2EError {
  NO_E2E_ERROR = 0,
  ENGINE_NOT_FOUND,
  CANT_ALLOCATE_ENGINE,
  CONTEXT_NOT_READY,
  INFERENCE_FAILED,
  UNSUPPORTED_BINDING_DATA_TYPE,
  ERROR_COUNT
} E2EError;

#define LOGI(msg) std::cout << msg << "." << std::endl;
#define LOGE(msg)                                                              \
  std::cout << msg << " : " << __FILE__ << " : " << __LINE__ << std::endl;

using namespace std;

#endif
