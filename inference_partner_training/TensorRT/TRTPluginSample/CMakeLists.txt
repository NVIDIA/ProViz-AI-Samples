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

cmake_minimum_required(VERSION 3.17)

project(TRTPluginSample CXX)

find_package(CUDAToolkit )

if(MSVC)
  # Force to always compile with W4
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif()
  set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT TRTPluginImportOnnx)
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  # Update if necessary
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")
endif()

add_library(MyPluginLibrary SHARED
    Util/util.h
    MyPluginLibrary/inc/MyPluginLibrary.h
    MyPluginLibrary/cpp/MyPluginLibrary.cpp
    MyPluginLibrary/MyPlugin/MyPlugin.h
    MyPluginLibrary/MyPlugin/MyPlugin.cpp
    )

target_compile_definitions(MyPluginLibrary PRIVATE MYPLUGIN_BUILD_LIB)
target_compile_features(MyPluginLibrary PRIVATE cxx_std_17)

target_include_directories(MyPluginLibrary
PRIVATE
    MyPluginLibrary/inc
    MyPluginLibrary/MyPlugin
    Util
)

find_library(TRT NAMES nvinfer PATHS ${CUDAToolkit_LIBRARY_DIR})
target_link_libraries(MyPluginLibrary
  PRIVATE
  ${TRT_PLUGIN}
  ${TRT}
  CUDA::cudart
  )

set_target_properties(MyPluginLibrary PROPERTIES
            CXX_EXTENSIONS OFF
            )

add_executable(TRTPluginImportOnnx
    TRTPluginImportOnnx/main.cpp
    Util/util.h
)

target_compile_features(TRTPluginImportOnnx PRIVATE cxx_std_17)

target_include_directories(TRTPluginImportOnnx
PRIVATE
    MyPluginLibrary/inc
    Util
)

find_library(TRT_ONNX NAMES nvonnxparser PATHS ${CUDAToolkit_LIBRARY_DIR})
find_library(TRT NAMES nvinfer PATHS ${CUDAToolkit_LIBRARY_DIR})
target_link_libraries(TRTPluginImportOnnx
  PRIVATE
  ${TRT}
  ${TRT_ONNX}
  MyPluginLibrary
  CUDA::cudart
)

add_executable(TRTPluginInfer
    TRTPluginInfer/main.cpp
    Util/util.h
)

target_compile_features(TRTPluginInfer PRIVATE cxx_std_17)

target_include_directories(TRTPluginInfer
PRIVATE
    MyPluginLibrary/inc
    Util
)

target_link_libraries(TRTPluginInfer
  PRIVATE
  ${TRT}
  ${CMAKE_DL_LIBS}
  MyPluginLibrary
  CUDA::cudart
)
