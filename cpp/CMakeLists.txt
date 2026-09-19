cmake_minimum_required(VERSION 3.16)
project(stretchwbc_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0)
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(eigen)
endif()

add_library(stretchwbc_safety src/safety_filter.cpp)
target_include_directories(stretchwbc_safety PUBLIC include)
target_link_libraries(stretchwbc_safety PUBLIC Eigen3::Eigen)

# ---------------------------------------------------------------- tests

option(STRETCHWBC_BUILD_TESTS "Build the gtest suite" ON)

if(STRETCHWBC_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()

  add_executable(test_safety_filter test/test_safety_filter.cpp)
  target_link_libraries(test_safety_filter PRIVATE stretchwbc_safety GTest::gtest_main)
  target_include_directories(test_safety_filter PRIVATE test)

  include(GoogleTest)
  gtest_discover_tests(test_safety_filter)
endif()
