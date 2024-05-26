#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include "WavReadWrite.h"

// Node structure for the Huffman tree
struct HuffmanNode {
    int16_t value;
    int frequency;
    HuffmanNode* left;
    HuffmanNode* right;

    HuffmanNode(int16_t val, int freq);
};

class DiskHuffmanTree {
public:
    // Default constructor
    DiskHuffmanTree();

    // Method to build the Huffman tree from WAV files
    void buildTree(const std::vector<std::string>& wavFiles);

    // Method to build the Huffman tree from raw data
    void buildTreeFromData(const std::vector<int16_t>& data);

    // Method to write the Huffman tree to disk
    void writeTreeToDisk(const std::string& filename) const;

    // Method to read the Huffman tree from disk
    void readTreeFromDisk(const std::string& filename);

    // Method to encode data using the Huffman tree
    void encode(const std::vector<int16_t>& data, std::string& encodedData) const;

    // Method to decode data using the Huffman tree
    void decode(const std::string& encodedData, std::vector<int16_t>& decodedData) const;

    // Main function to build and save Huffman tree
    static void buildAndSaveTree(const std::string& directory, const std::string& outputFile);

private:
    std::unordered_map<int16_t, int> frequencyMap;
    std::unordered_map<int16_t, std::string> huffmanCodes;
    HuffmanNode* root;

    // Helper function to serialize the tree
    void serializeTree(std::ofstream& outFile, HuffmanNode* node) const;

    // Helper function to deserialize the tree
    HuffmanNode* deserializeTree(std::ifstream& inFile);

    // Helper function to generate Huffman codes
    void generateCodes(HuffmanNode* node, const std::string& code);
};
