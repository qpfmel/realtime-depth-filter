cmake_minimum_required(VERSION 3.20)
project(RealtimeDepthFilter LANGUAGES CXX )

find_package(OpenCV REQUIRED)

add_executable(depth_filter src/main.cpp)
target_compile_features(depth_filter PRIVATE cxx_std_17)
target_link_libraries(depth_filter PRIVATE ${OpenCV_LIBS})