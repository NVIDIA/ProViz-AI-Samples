/*###############################################################################
#
# Copyright 2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
###############################################################################*/
#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <string>

#ifdef NDEBUG

#define CheckCUDA(call) call;
#define cudaCheckError(stream) ;

#else

#define CheckCUDA(call)                                                        \
  {                                                                            \
    const cudaError_t err = call;                                              \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CudaDebugCall() failed at: " << __FILE__ << ":"            \
                << __LINE__ << "; ";                                           \
      std::cerr << "code: " << err                                             \
                << "; description: " << cudaGetErrorString(err) << std::endl;  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError(stream)                                                 \
  {                                                                            \
    cudaStreamSynchronize(stream);                                             \
    cudaError_t e = cudaGetLastError();                                        \
    if (e != cudaSuccess) {                                                    \
      std::cerr << "Cuda failure :" << __FILE__ << ":" << __LINE__             \
                << " with error: " << cudaGetErrorString(e) << std::endl;      \
      exit(0);                                                                 \
    }                                                                          \
  }

#endif
