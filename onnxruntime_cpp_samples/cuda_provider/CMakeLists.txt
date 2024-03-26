project(cuda_sample CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 70 72 75 80 86 89)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

add_executable(cuda_sample src/cuda_sample.cpp
        src/CUDAInference.cpp
        src/TRTInference.cpp
        src/FormatConversion.cu src/NVIDIAInference.cpp)

set_property(TARGET cuda_sample PROPERTY CXX_STANDARD 20)
target_include_directories(cuda_sample PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
target_link_libraries(cuda_sample PRIVATE
        onnxruntime_interface
        CUDA::cudart
        CUDA::nvToolsExt
        image_utils
        argparse::argparse
        fmt::fmt
        )

# https://github.com/microsoft/onnxruntime/issues/11236
if (WIN32)
    include(FetchContent)
    FetchContent_Declare(zlib_dll
            URL http://www.winimage.com/zLibDll/zlib123dllx64.zip)
    FetchContent_MakeAvailable(zlib_dll)
    file(GLOB ZLIB_DLL "${zlib_dll_SOURCE_DIR}/dll_x64/*.dll")
    add_custom_command(TARGET cuda_sample PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${ZLIB_DLL} $<TARGET_FILE_DIR:cuda_sample>)
    file(GLOB RUNTIME_PATH "${onnxruntime_INSTALL}/lib/*.dll")
    list(LENGTH RUNTIME_PATH RES_LEN)
    if (RES_LEN GREATER 0)
        add_custom_command(TARGET cuda_sample PRE_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                ${RUNTIME_PATH} $<TARGET_FILE_DIR:cuda_sample>)
    endif ()
    file(GLOB RUNTIME_PATH "${onnxruntime_INSTALL}/bin/*.dll")
    list(LENGTH RUNTIME_PATH RES_LEN)
    if (RES_LEN GREATER 0)
        add_custom_command(TARGET cuda_sample PRE_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                ${RUNTIME_PATH} $<TARGET_FILE_DIR:cuda_sample>)
    endif ()
endif ()
