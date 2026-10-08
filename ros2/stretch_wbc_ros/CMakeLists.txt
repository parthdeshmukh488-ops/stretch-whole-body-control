cmake_minimum_required(VERSION 3.16)
project(stretch_wbc_ros LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# The safety filter is shared with the standalone C++ port in the repository's cpp/.
set(STRETCHWBC_CPP ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp)

add_library(stretch_wbc_core src/controller.cpp ${STRETCHWBC_CPP}/src/safety_filter.cpp)
target_include_directories(stretch_wbc_core PUBLIC include ${STRETCHWBC_CPP}/include)
target_link_libraries(stretch_wbc_core PUBLIC Eigen3::Eigen)

add_executable(wbc_node src/wbc_node.cpp)
target_link_libraries(wbc_node stretch_wbc_core)
ament_target_dependencies(wbc_node rclcpp tf2 tf2_ros geometry_msgs sensor_msgs)

install(TARGETS wbc_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_controller test/test_controller.cpp)
  target_link_libraries(test_controller stretch_wbc_core)
  target_include_directories(test_controller PRIVATE test)
endif()

ament_package()
