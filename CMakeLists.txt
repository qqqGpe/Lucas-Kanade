cmake_minimum_required( VERSION 3.15 )
project(myKLT)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# Make Eigen's default Matrix storage order row-major so gradient maps and other
# OpenCV-sourced matrices line up with cv::Mat without an explicit transpose.
add_compile_definitions(EIGEN_DEFAULT_TO_ROW_MAJOR)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)

add_subdirectory(${PROJECT_SOURCE_DIR}/src)

add_executable(myKLT main.cpp)
target_link_libraries(myKLT klt_lib)
