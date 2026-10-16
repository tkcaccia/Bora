cmake_minimum_required(VERSION 3.16)
project(bora_tri_mesh LANGUAGES CXX)
find_package(OpenMP REQUIRED)
add_library(bora_tri_mesh SHARED tri_mesh_refine.cpp)
target_compile_features(bora_tri_mesh PRIVATE cxx_std_17)
target_compile_options(bora_tri_mesh PRIVATE -O3)
target_link_libraries(bora_tri_mesh PRIVATE OpenMP::OpenMP_CXX)
set_target_properties(bora_tri_mesh PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
