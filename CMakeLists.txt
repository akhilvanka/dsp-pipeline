cmake_minimum_required(VERSION 3.16)
project(DSPPipeline VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wno-unused-parameter)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native)
endif()

include_directories(include)

set(DSP_SOURCES src/fft.cpp src/anomaly.cpp)

add_executable(dsp_demo   ${DSP_SOURCES} src/main.cpp)
add_executable(dsp_tests  ${DSP_SOURCES} tests/test_dsp.cpp)

enable_testing()
add_test(NAME dsp_tests COMMAND dsp_tests)
