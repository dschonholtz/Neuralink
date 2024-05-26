#include "DiskHuffmanTree.h"
#include <queue>
#include <iostream>
#include <fstream>
#include <limits>
#include <stack>
#include <filesystem>

namespace fs = std::__fs::filesystem;

// Node structure for the Huffman tree
HuffmanNode::HuffmanNode(int16_t val, int freq) : value(val), frequency(freq), left(nullptr), right(nullptr) {}


// Comparator for the priority queue
struct Compare {
    bool operator()(HuffmanNode* l, HuffmanNode* r) {
        return l->frequency > r->frequency;
    }
};

void DiskHuffmanTree::serializeTree(std::ofstream& outFile, HuffmanNode* node) const {
    if (!node) {
        outFile << "# "; // Use '#' to denote null nodes
        return;
    }
    outFile << node->value << " " << node->frequency << " ";
    serializeTree(outFile, node->left);
    serializeTree(outFile, node->right);
}

DiskHuffmanTree::DiskHuffmanTree() : root(nullptr) {
    // Initialize frequencyMap with all possible int16_t values set to 1
    for (int i = std::numeric_limits<int16_t>::min(); i <= std::numeric_limits<int16_t>::max(); ++i) {
        frequencyMap[static_cast<int16_t>(i)] = 1;
    }
}

void DiskHuffmanTree::buildTreeFromData(const std::vector<int16_t>& data) {
    // Reset frequencyMap
    for (auto& entry : frequencyMap) {
        entry.second = 1;
    }

    // Update frequencyMap with the provided data
    for (const auto& sample : data) {
        frequencyMap[sample]++;
    }

    // Create a priority queue to build the Huffman tree
    std::priority_queue<HuffmanNode*, std::vector<HuffmanNode*>, Compare> pq;

    // Create a leaf node for each unique character and add it to the priority queue
    for (const auto& entry : frequencyMap) {
        pq.push(new HuffmanNode(entry.first, entry.second));
    }

    // Iterate until the size of the queue becomes 1
    while (pq.size() != 1) {
        // Extract the two nodes with the highest priority (lowest frequency)
        HuffmanNode* left = pq.top();
        pq.pop();
        HuffmanNode* right = pq.top();
        pq.pop();

        // Create a new internal node with these two nodes as children and with frequency equal to the sum of the two nodes' frequencies
        HuffmanNode* top = new HuffmanNode(-1, left->frequency + right->frequency);
        top->left = left;
        top->right = right;

        // Add the new node to the priority queue
        pq.push(top);
    }

    // The remaining node is the root node and the tree is complete
    root = pq.top();

    // Generate Huffman codes
    generateCodes(root, "");
}

void DiskHuffmanTree::buildTree(const std::vector<std::string>& wavFiles) {
    WavHeader header;
    std::vector<int16_t> allData;
    for (const auto& file : wavFiles) {
        std::vector<int16_t> data = WavReadWrite::readWavFile(file, header);
        allData.insert(allData.end(), data.begin(), data.end());
    }

    buildTreeFromData(allData);
}

void DiskHuffmanTree::generateCodes(HuffmanNode* node, const std::string& code) {
    if (!node) return;

    // If this is a leaf node, store the code
    if (!node->left && !node->right) {
        huffmanCodes[node->value] = code;
    }

    // Traverse the left and right children
    generateCodes(node->left, code + "0");
    generateCodes(node->right, code + "1");
}

void DiskHuffmanTree::writeTreeToDisk(const std::string& filename) const {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile.is_open()) {
        throw std::runtime_error("Could not open file for writing");
    }

    // Serialize the tree
    serializeTree(outFile, root);

    // Serialize the codes
    for (const auto& entry : huffmanCodes) {
        outFile << entry.first << " " << entry.second << " ";
    }

    outFile.close();
}

HuffmanNode* DiskHuffmanTree::deserializeTree(std::ifstream& inFile) {
    std::stack<HuffmanNode**> nodeStack;
    HuffmanNode* root = nullptr;
    nodeStack.push(&root);

    std::string value;
    while (!nodeStack.empty()) {
        HuffmanNode** nodePtr = nodeStack.top();
        nodeStack.pop();

        inFile >> value;
        if (value == "#") {
            *nodePtr = nullptr;
        } else {
            int16_t nodeValue = std::stoi(value);
            int frequency;
            inFile >> frequency;

            *nodePtr = new HuffmanNode(nodeValue, frequency);
            nodeStack.push(&((*nodePtr)->right));
            nodeStack.push(&((*nodePtr)->left));
        }
    }

    return root;
}

void DiskHuffmanTree::readTreeFromDisk(const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile.is_open()) {
        throw std::runtime_error("Could not open file for reading");
    }

    // Deserialize the tree
    root = deserializeTree(inFile);

    // Deserialize the codes
    int16_t key;
    std::string code;
    while (inFile >> key >> code) {
        huffmanCodes[key] = code;
    }

    inFile.close();
}

void DiskHuffmanTree::encode(const std::vector<int16_t>& data, std::string& encodedData) const {
    encodedData.clear();
    for (const auto& sample : data) {
        auto it = huffmanCodes.find(sample);
        if (it != huffmanCodes.end()) {
            encodedData += it->second;
        } else {
            throw std::runtime_error("Sample not found in Huffman codes");
        }
    }
}

void DiskHuffmanTree::decode(const std::string& encodedData, std::vector<int16_t>& decodedData) const {
    decodedData.clear();
    HuffmanNode* currentNode = root;

    for (char bit : encodedData) {
        if (bit == '0') {
            currentNode = currentNode->left;
        } else if (bit == '1') {
            currentNode = currentNode->right;
        } else {
            throw std::runtime_error("Invalid bit in encoded data");
        }

        // If a leaf node is reached
        if (!currentNode->left && !currentNode->right) {
            decodedData.push_back(currentNode->value);
            currentNode = root; // Reset to root for the next symbol
        }
    }

    // Ensure the last node is a leaf node
    if (currentNode != root) {
        throw std::runtime_error("Encoded data does not represent a valid Huffman encoding");
    }
}

void DiskHuffmanTree::buildAndSaveTree(const std::string& directory, const std::string& outputFile) {
    std::vector<std::string> wavFiles;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.path().extension() == ".wav") {
            wavFiles.push_back(entry.path().string());
            if (wavFiles.size() == 5) break;
        }
    }

    if (wavFiles.empty()) {
        std::cerr << "No WAV files found in the directory." << std::endl;
        return;
    }

    DiskHuffmanTree tree;
    tree.buildTree(wavFiles);
    tree.writeTreeToDisk(outputFile);
}
