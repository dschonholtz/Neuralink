#include <iostream>
#include <vector>
#include <string>
#include "DiskHuffmanTree.h"
#include <gtest/gtest.h>

TEST(HuffmanTest, CompressionDecompression) {
    // Generate some raw data
    std::vector<int16_t> rawData = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5};

    // Create a DiskHuffmanTree object
    DiskHuffmanTree huffmanTree;

    // Build the Huffman tree from the raw data
    huffmanTree.buildTreeFromData(rawData);

    // Encode the raw data
    std::string encodedData;
    huffmanTree.encode(rawData, encodedData);

    // Decode the encoded data
    std::vector<int16_t> decodedData;
    huffmanTree.decode(encodedData, decodedData);

    // Verify that the input and output are the same
    ASSERT_EQ(rawData, decodedData);
}

