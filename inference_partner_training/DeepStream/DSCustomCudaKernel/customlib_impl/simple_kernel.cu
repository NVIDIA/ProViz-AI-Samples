/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
 
#include "simple_kernel.hpp"
#include <stdint.h>
#include <iostream>
#include <cuda.h>

namespace
{
const int BLOCK_X = 32;
const int BLOCK_Y = 32;

__global__ void kernel(void* in_data, void* out_data, int width, int height, int pitch)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width)
        return;
    if (y >= height)
        return;

    uchar4 input_pixel = static_cast<uchar4*>(in_data)[y * (pitch/sizeof(uchar4)) + x];
    uint8_t gray_value = static_cast<uint8_t>(input_pixel.x * 0.2989f + input_pixel.y *0.587f +  input_pixel.z * 0.114f);
    uchar4 output_pixel = {gray_value, gray_value, gray_value, 0xFF};

    static_cast<uchar4*>(out_data)[y * (pitch/sizeof(uchar4)) + x] = output_pixel;
}


template<typename X, typename Y, typename std::enable_if_t<std::is_integral_v<X> && std::is_integral_v<Y>, int> = 0>
__host__ __device__ X div_up(const X& x, const Y& y)
{
    return (x + y - 1) / static_cast<X>(y);
}
} // namespace

void simple_kernel(NvBufSurface* in, NvBufSurface* out)
{

    NvBufSurfaceParams in_params = in->surfaceList[0];
    NvBufSurfaceParams out_params = out->surfaceList[0];

    dim3 grid(div_up(in_params.width, BLOCK_X), div_up(in_params.height, BLOCK_Y), 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);
    kernel<<<grid, block>>>(
        in_params.dataPtr,
        out_params.dataPtr,
        in_params.width,
        in_params.height,
        in_params.pitch);
}
