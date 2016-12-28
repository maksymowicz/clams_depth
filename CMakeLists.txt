cmake_minimum_required(VERSION 2.8.3)
project(clams_ros)

set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")
set(CMAKE_BUILD_TYPE Debug)

find_package(catkin REQUIRED COMPONENTS
    image_transport
    nodelet
    roscpp
    sensor_msgs
)

find_package(Boost REQUIRED COMPONENTS system thread)
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)


catkin_package(
    INCLUDE_DIRS include
    LIBRARIES clams_depth_nodelet
    CATKIN_DEPENDS image_transport nodelet roscpp sensor_msgs
    DEPENDS Boost OpenMP Eigen3 OpenCV
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

add_library(clams_depth_nodelet src/clams_depth_nodelet.cpp src/clams_depth.cpp
    src/discrete_depth_distortion_model.cpp)
target_link_libraries(clams_depth_nodelet ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(clams_depth_node src/clams_depth_node.cpp src/clams_depth.cpp
    src/discrete_depth_distortion_model.cpp)
target_link_libraries(clams_depth_node ${catkin_LIBRARIES} ${OpenCV_LIBS})
