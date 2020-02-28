cmake_minimum_required(VERSION 2.8.3)
project(initialization_publisher)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -DEIGEN_DONT_PARALLELIZE")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -rdynamic")

set(cv_bridge_DIR "/usr/local/share/cv_bridge/cmake")
find_package(catkin REQUIRED COMPONENTS
    roscpp
    pcl_ros
    camera_model
    cv_bridge
    tf
    afm
    )

    
catkin_package()
include_directories(${catkin_INCLUDE_DIRS})

find_package(OpenCV)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(euroc_publisher
    src/euroc_publisher_node.cpp
    )
add_dependencies(euroc_publisher afm_gencpp)
target_link_libraries(euroc_publisher ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(realsense_d435i_publisher
    src/realsense_d435i.cpp
    )

target_link_libraries(realsense_d435i_publisher ${catkin_LIBRARIES} ${OpenCV_LIBS})
