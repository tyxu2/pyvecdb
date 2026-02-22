cmake_minimum_required(VERSION 3.18)
project(pyvecdb LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Check for CUDA (disabled for now due to linking issues)
set(DISABLE_CUDA ON)

if(NOT DISABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        add_definitions(-DUSE_CUDA)
        find_package(CUDA REQUIRED)
        message(STATUS "CUDA found. Building with GPU support.")
    else()
        message(STATUS "CUDA not found. Building CPU-only.")
    endif()
else()
    message(STATUS "CUDA disabled. Building CPU-only.")
endif()

# Find pybind11
find_package(pybind11 REQUIRED)

# Source files
set(SOURCES
    src/bindings.cpp
    src/IndexFlat.cpp
    src/IndexIVF.cpp
    src/IndexHNSW.cpp
    src/Utils.cpp
)

if(CMAKE_CUDA_COMPILER AND NOT DISABLE_CUDA)
    list(APPEND SOURCES src/Distance.cu)
else()
    list(APPEND SOURCES src/Distance.cpp)
endif()

pybind11_add_module(_pyvecdb ${SOURCES})

target_include_directories(_pyvecdb PRIVATE src)

if(CMAKE_CUDA_COMPILER AND NOT DISABLE_CUDA)
    set_target_properties(_pyvecdb PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(_pyvecdb PRIVATE ${CUDA_LIBRARIES} cuda)
else()
    # CPU-only build
endif()
