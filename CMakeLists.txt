cmake_minimum_required( VERSION 3.0 )
project(myKLT)

# set(CMAKE_BUILD_TYPE Debug)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Release 模式下最大优化选项
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops")
endif()

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)

add_subdirectory(${PROJECT_SOURCE_DIR}/src)

add_executable(myKLT main.cpp)
target_link_libraries(myKLT klt_lib)
