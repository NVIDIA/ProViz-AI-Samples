# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

add_library(customlib SHARED
    customlib_impl.cpp
    simple_kernel.cu
    simple_kernel.hpp
    )

target_include_directories(
    customlib
PRIVATE
    ../includes
    ${DEEP_STREAM_ROOT}/sources/includes/
    ${GST_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
	)

target_link_libraries(customlib
    PUBLIC
    nvdsgst_helper
    nvdsgst_meta
    nvds_meta
    nvbufsurface
    nvbufsurftransform
    nvdsbufferpool
    ${GST_LIBRARIES}
    ${CUDA_LIBRARIES}
)

target_compile_options(customlib PRIVATE
    ${GST_CFLAGS} 
    )

set_target_properties(customlib PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
    #CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}
    )
