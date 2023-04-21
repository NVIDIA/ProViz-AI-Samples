
project(dml_sample CXX)

file(GLOB SOURCES "src/*.cpp")

add_executable(dml_sample ${SOURCES})


set(ORT_RELEASE_DML "https://github.com/microsoft/onnxruntime/releases/download/v1.13.1/Microsoft.ML.OnnxRuntime.DirectML.1.13.1.zip")

FetchContent_Declare(dml
    URL "https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/1.10.1")
FetchContent_MakeAvailable(dml)

FetchContent_Declare(onnxruntime_dml
        URL ${ORT_RELEASE_DML})
FetchContent_MakeAvailable(onnxruntime_dml)
set(onnxruntime_dml_INSTALL ${onnxruntime_dml_SOURCE_DIR})
add_library(onnxruntime_interface_dml INTERFACE)
target_include_directories(onnxruntime_interface_dml INTERFACE
        "${onnxruntime_dml_INSTALL}/build/native/include")
target_link_directories(onnxruntime_interface_dml INTERFACE "${onnxruntime_dml_INSTALL}/runtimes/win-x64/native")
target_link_libraries(onnxruntime_interface_dml INTERFACE onnxruntime)

FetchContent_Declare(dmlheaders
        GIT_REPOSITORY https://github.com/microsoft/DirectX-Headers.git
        GIT_TAG v1.600.10
        )
FetchContent_MakeAvailable(dmlheaders)

set_property(TARGET dml_sample PROPERTY CXX_STANDARD 20)
target_compile_definitions(dml_sample PRIVATE PROJECT_ROOT="${PROJECT_ROOT}")
target_include_directories(dml_sample PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        ${dmlheaders_SOURCE_DIR}/include/directx
        )
target_link_libraries(dml_sample PRIVATE
        onnxruntime_interface_dml
        image_utils
        d3d12.lib
        dxgi.lib
        dxguid.lib
        d3dcompiler.lib
        DirectML.lib
        argparse::argparse
        )

file(GLOB RUNTIME_PATH "${onnxruntime_dml_INSTALL}/runtimes/win-x64/native/*.dll")
list(LENGTH RUNTIME_PATH RES_LEN)
if (RES_LEN GREATER 0)
    add_custom_command(TARGET dml_sample PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${RUNTIME_PATH} $<TARGET_FILE_DIR:dml_sample>)
endif ()


file(GLOB DML_DLL_PATH "${dml_SOURCE_DIR}/bin/x64-win/*.dll")
list(LENGTH DML_DLL_PATH RES_LEN)
if (RES_LEN GREATER 0)
    add_custom_command(TARGET dml_sample PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${DML_DLL_PATH} $<TARGET_FILE_DIR:dml_sample>)
endif ()

# compile shaders
add_custom_target(shaders)
file(GLOB SHADER "${CMAKE_CURRENT_SOURCE_DIR}/shader/*.hlsl")

set_source_files_properties(VertexShader.hlsl PROPERTIES ShaderType "vs")
set_source_files_properties(PixelShader.hlsl PROPERTIES ShaderType "ps")
set_source_files_properties(${HLSL_SHADER_FILES} PROPERTIES ShaderModel "4_0")

FetchContent_Declare(dxc
        URL https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.7.2207/dxc_2022_07_18.zip
        )
FetchContent_MakeAvailable(dxc)

foreach (FILE ${SHADER})
    get_filename_component(FILE_WE ${FILE} NAME_WE)
    add_custom_command(TARGET shaders
            COMMAND ${dxc_SOURCE_DIR}/bin/x64/dxc.exe -T cs_6_6 -O3 ${FILE} -enable-16bit-types -Fo shader/${FILE_WE}.cso
            MAIN_DEPENDENCY ${FILE}
            COMMENT "HLSL ${FILE}"
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            VERBATIM)
endforeach (FILE)

add_dependencies(dml_sample shaders)
