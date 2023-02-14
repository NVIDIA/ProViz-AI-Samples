cmake_minimum_required(VERSION 3.24)
cmake_policy(SET CMP0077 NEW)

project(onnx_samples)
set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(CXX_STANDARD 20)

if (NOT DEFINED onnxruntime_INSTALL)
    if (UNIX)
        set(ORT_RELEASE "https://github.com/microsoft/onnxruntime/releases/download/v1.14.0/onnxruntime-linux-x64-gpu-1.14.0.tgz")
    else ()
        set(ORT_RELEASE "https://github.com/microsoft/onnxruntime/releases/download/v1.14.0/onnxruntime-win-x64-gpu-1.14.0.zip")
    endif ()

    include(FetchContent)

    FetchContent_Declare(onnxruntime
            URL ${ORT_RELEASE})
    FetchContent_MakeAvailable(onnxruntime)
    set(onnxruntime_INSTALL ${onnxruntime_SOURCE_DIR})
endif ()

add_library(onnxruntime_interface INTERFACE)
target_include_directories(onnxruntime_interface INTERFACE
        "${onnxruntime_INSTALL}/include"
        "${onnxruntime_INSTALL}/include/onnxruntime"
        "${onnxruntime_INSTALL}/include/onnxruntime/core/session"
        "${onnxruntime_INSTALL}/include/onnxruntime/core")
target_link_directories(onnxruntime_interface INTERFACE "${onnxruntime_INSTALL}/lib")
target_link_libraries(onnxruntime_interface INTERFACE onnxruntime)


add_subdirectory(common)
add_subdirectory(cuda_provider)
