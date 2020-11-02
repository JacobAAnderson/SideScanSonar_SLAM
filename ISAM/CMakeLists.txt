cmake_minimum_required(VERSION 2.8.3)
project(sidescansoanr_slam)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  message_filters
)

catkin_package()


## Specify additional locations of header files --> Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
  /home/jake/Cpp/isam_v1_7/include
  /usr/include/eigen3
)

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
add_executable(${PROJECT_NAME}_node src/sss_slam.cpp)
# add_executable(${PROJECT_NAME}_node src/Demo_isam.cpp )

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
	/home/jake/Cpp/isam_v1_7/lib/libisam.a
	cholmod
)
