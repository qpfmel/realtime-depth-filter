cmake_minimum_required(VERSION 3.20)
project(RealtimeDepthFilter LANGUAGES CXX )

find_package(OpenCV REQUIRED)

add_executable(depth_filter src/main.cpp)
target_compile_features(depth_filter PRIVATE cxx_std_17)
target_link_libraries(depth_filter PRIVATE ${OpenCV_LIBS})
target_compile_options(depth_filter PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

# TODO: 경로가 이 컴퓨터에 고정돼 있음. 나중에 설정 가능하게 바꿀 것
set(ORT_ROOT   "Q:/sdk/onnxruntime-win-x64-gpu_cuda12-1.30.0")
set(CUDNN_ROOT "Q:/sdk/cudnn-windows-x86_64-9.26.0.51_cuda12-archive")
set(CUDART_DLL "Q:/sdk/cuda_cudart-windows-x86_64-12.9.79-archive/bin/cudart64_12.dll")

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION             "${ORT_ROOT}/lib/onnxruntime.dll"
    IMPORTED_IMPLIB               "${ORT_ROOT}/lib/onnxruntime.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_ROOT}/include")

target_link_libraries(depth_filter PRIVATE onnxruntime)

file(GLOB CUDNN_DLLS "${CUDNN_ROOT}/bin/x64/*.dll")
add_custom_command(TARGET depth_filter POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ORT_ROOT}/lib/onnxruntime.dll"
            "${ORT_ROOT}/lib/onnxruntime_providers_shared.dll"
            "${ORT_ROOT}/lib/onnxruntime_providers_cuda.dll"
            "${CUDART_DLL}"
            ${CUDNN_DLLS}
            "$<TARGET_FILE_DIR:depth_filter>")