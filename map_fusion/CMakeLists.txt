cmake_minimum_required(VERSION 2.8.3)
project(map_fusion)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -rdynamic")

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")
endif()
find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

set(cv_bridge_DIR "/usr/local/share/cv_bridge/cmake")
find_package(catkin REQUIRED COMPONENTS
    roscpp
    tf
    cv_bridge
    camera_model
    afm
    )

catkin_package()
include_directories(${catkin_INCLUDE_DIRS}) 

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


find_package(Boost REQUIRED COMPONENTS filesystem)
include_directories(${Boost_INCLUDE_DIRS})

# 2D to 3D tracking with slidewindow
add_executable(window_tracking_node
    src/tracking_node.cpp
    src/estimator.cpp
    src/line.cpp
    src/CameraPoseVisualization.cpp
    )
add_dependencies(window_tracking_node afm_gencpp)
target_link_libraries(window_tracking_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${CERES_LIBRARIES})
