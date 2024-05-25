// HuffmanCompressor.h
#ifndef HUFFMANCOMPRESSOR_H
#define HUFFMANCOMPRESSOR_H

#include <vector>
#include <bitset>
#include <unordered_map>
#include <string>
#include <queue>
#include <memory>
#include <iostream>

/**
 * @class HuffmanNode
 * @brief Represents a node in the Huffman Tree.
 */
class HuffmanNode {
public:
    int16_t value; ///< The value stored in the node.
    int frequency; ///< The frequency of the value.
    std::shared_ptr<HuffmanNode> left; ///< Pointer to the left child.
    std::shared_ptr<HuffmanNode> right; ///< Pointer to the right child.

    /**
     * @brief Constructor for HuffmanNode.
     * @param val The value to be stored in the node.
     * @param freq The frequency of the value.
     */
    HuffmanNode(int16_t val, int freq) : value(val), frequency(freq), left(nullptr), right(nullptr) {}

    /**
     * @brief Comparator for priority queue.
     * @param other The other HuffmanNode to compare with.
     * @return True if this node's frequency is greater than the other node's frequency.
     */
    bool operator>(const HuffmanNode& other) const {
        return frequency > other.frequency;
    }
};

/**
 * @class HuffmanCodes
 * @brief Responsible for building the Huffman codes from the Huffman Tree.
 */
class HuffmanCodes {
public:
    /**
     * @brief Recursively builds the Huffman codes.
     * @param node The current node in the Huffman Tree.
     * @param prefix The current prefix string representing the path in the tree.
     * @param codebook The map storing the Huffman codes for each value.
     */
    void buildCodes(HuffmanNode* node, const std::string& prefix, std::unordered_map<int16_t, std::string>& codebook) {
        if (!node) return;

        if (!node->left && !node->right) { // Leaf node check
            codebook[node->value] = prefix;
        } else {
            if (node->left) {
                buildCodes(node->left.get(), prefix + "0", codebook);
            }
            if (node->right) {
                buildCodes(node->right.get(), prefix + "1", codebook);
            }
        }
    }
};

/**
 * @class HuffmanTree
 * @brief Responsible for building the Huffman Tree from a frequency table.
 */
class HuffmanTree {
public:
    /**
     * @brief Builds the Huffman Tree from a frequency table.
     * @param frequencies The frequency table mapping values to their frequencies.
     * @return The root node of the constructed Huffman Tree.
     */
    std::shared_ptr<HuffmanNode> buildTree(const std::unordered_map<int16_t, int>& frequencies) {
        if (frequencies.empty()) {
            return nullptr;
        }

        auto cmp = [](std::shared_ptr<HuffmanNode> left, std::shared_ptr<HuffmanNode> right) { return *left > *right; };
        std::priority_queue<std::shared_ptr<HuffmanNode>, std::vector<std::shared_ptr<HuffmanNode>>, decltype(cmp)> minHeap(cmp);

        for (const auto& pair : frequencies) {
            minHeap.push(std::make_shared<HuffmanNode>(pair.first, pair.second));
        }

        while (minHeap.size() > 1) {
            auto left = minHeap.top(); minHeap.pop();
            auto right = minHeap.top(); minHeap.pop();
            auto merged = std::make_shared<HuffmanNode>('\0', left->frequency + right->frequency);
            merged->left = left;
            merged->right = right;
            minHeap.push(merged);
        }

        return minHeap.top();
    }
};

/**
 * @class HuffmanCompressor
 * @brief Provides methods to compress and decompress data using Huffman coding.
 */
class HuffmanCompressor {
public:
    /**
     * @brief Compresses the input data using Huffman coding.
     * @param data The input data to be compressed.
     * @return A pair containing the compressed data as a vector of uint8_t and the root node of the Huffman Tree.
     */
    std::pair<std::vector<uint8_t>, std::shared_ptr<HuffmanNode>> compress(const std::vector<int16_t>& data) {
        std::unordered_map<int16_t, int> frequencies;
        for (int16_t value : data) {
            frequencies[value]++;
        }

        HuffmanTree treeBuilder;
        std::shared_ptr<HuffmanNode> root = treeBuilder.buildTree(frequencies);

        HuffmanCodes codeBuilder;
        std::unordered_map<int16_t, std::string> codebook;
        codeBuilder.buildCodes(root.get(), "", codebook);

        std::string encodedString;
        for (int16_t value : data) {
            encodedString += codebook[value];
        }

        std::vector<uint8_t> compressedData;
        compressedData.push_back(encodedString.size() & 0xFF); // Store lower byte of size
        compressedData.push_back((encodedString.size() >> 8) & 0xFF); // Store upper byte of size

        uint8_t currentByte = 0;
        int bitCount = 0;

        for (char bit : encodedString) {
            currentByte = (currentByte << 1) | (bit - '0');
            bitCount++;

            if (bitCount == 8) {
                compressedData.push_back(currentByte);
                currentByte = 0;
                bitCount = 0;
            }
        }

        if (bitCount > 0) {
            currentByte <<= (8 - bitCount); // Shift remaining bits to the left
            compressedData.push_back(currentByte);
        }

        return {compressedData, root};
    }

    /**
     * @brief Decompresses the input data using Huffman coding.
     * @param compressedData The compressed data to be decompressed.
     * @param root The root node of the Huffman Tree used for decompression.
     * @param originalSize The original size of the decompressed data.
     * @return The decompressed data as a vector of int16_t.
     */
    std::vector<int16_t> decompress(const std::vector<uint8_t>& compressedData, HuffmanNode* root, size_t originalSize) {
        size_t encodedSize = (compressedData[1] << 8) | compressedData[0]; // Read size
        std::string bitString;
        for (size_t i = 2; i < compressedData.size(); ++i) {
            std::bitset<8> bits(compressedData[i]);
            for (int j = 7; j >= 0; --j) {
                bitString.push_back(bits.test(j) ? '1' : '0');
            }
        }
        bitString = bitString.substr(0, encodedSize); // Use the stored size to trim the bitString

        std::vector<int16_t> decompressedData;
        HuffmanNode* currentNode = root;
        std::cout << "root value: " << root->value << std::endl;
        std::cout << "bitString: " << bitString << std::endl;
        for (char bit : bitString) {
            currentNode = (bit == '0') ? currentNode->left.get() : currentNode->right.get();
            // print the value of the current node
            std::cout << "Current node value: " << currentNode->value << std::endl;
            if (!currentNode->left && !currentNode->right) {
                std::cout << "Leaf node reached, value: " << currentNode->value << std::endl;
                decompressedData.push_back(currentNode->value);
                currentNode = root;
                if (decompressedData.size() == originalSize) break;
            }
        }

        return decompressedData;
    }
};


#endif // HUFFMANCOMPRESSOR_H
