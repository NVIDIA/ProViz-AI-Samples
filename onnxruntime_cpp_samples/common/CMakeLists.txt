option(FETCHCONTENT_QUIET OFF)
include(FetchContent)

set(ZLIB_USE_STATIC_LIBS "ON")
FetchContent_Declare(zlib
        GIT_REPOSITORY https://github.com/madler/zlib.git
        GIT_TAG v1.2.11
)
FetchContent_MakeAvailable(zlib)
set(ZLIB_INCLUDE_DIR ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})
set(ZLIB_LIBRARY zlibstatic)

set(SKIP_INSTALL_ALL ON)
option(PNG_BUILD_ZLIB ON)
option(PNG_SHARED OFF)
set(PNG_STATIC ON)
FetchContent_Declare(libpng
        GIT_REPOSITORY https://github.com/glennrp/libpng.git
        GIT_TAG v1.6.37
)
FetchContent_MakeAvailable(libpng)

FetchContent_Declare(argparse
        URL https://github.com/p-ranav/argparse/archive/v2.2.tar.gz
        URL_HASH SHA256=f0fc6ab7e70ac24856c160f44ebb0dd79dc1f7f4a614ee2810d42bb73799872b
        )
FetchContent_MakeAvailable(argparse)

FetchContent_Declare(fmt
        URL https://github.com/fmtlib/fmt/archive/8.0.1.tar.gz
        URL_HASH SHA256=b06ca3130158c625848f3fb7418f235155a4d389b2abc3a6245fb01cb0eb1e01
        )
FetchContent_MakeAvailable(fmt)


add_library(image_utils STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/E2EImageUtils.cpp)
set_property(TARGET image_utils PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(image_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc
        ${libpng_SOURCE_DIR}
        ${libpng_BINARY_DIR}
        )
target_link_directories(image_utils PRIVATE ${libpng_BINARY_DIR})
target_link_libraries(image_utils PRIVATE png_static)