cmake_minimum_required(VERSION 3.10)
project(HuffmanNodeTests)

set(CMAKE_CXX_STANDARD 11)

# add gtest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Add your test file
add_executable(runTests TestHuffman.cpp)
target_link_libraries(runTests GTest::Main pthread)
# Add decode executable
add_executable(decode decode.cpp)

# Add encode executable
add_executable(encode encode.cpp)
