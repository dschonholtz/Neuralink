cmake_minimum_required(VERSION 3.10)
project(HuffmanNodeTests)

set(CMAKE_CXX_STANDARD 11)

# add gtest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Include directories
include_directories(include)

# Create a library for the common code
add_library(DiskHuffmanTreeLib src/DiskHuffmanTree.cpp src/WavReadWrite.cpp)

# Add decode executable
add_executable(decode src/decode.cpp)
target_link_libraries(decode DiskHuffmanTreeLib)

# Add encode executable
add_executable(encode src/encode.cpp)
target_link_libraries(encode DiskHuffmanTreeLib)

# Add diskHuffmanTree executable
add_executable(diskHuffmanTree src/BuildDiskHuffmanTree.cpp)
target_link_libraries(diskHuffmanTree DiskHuffmanTreeLib)

# Add test executable
add_executable(testHuffman tests/TestHuffman.cpp)
target_link_libraries(testHuffman DiskHuffmanTreeLib GTest::GTest GTest::Main)
