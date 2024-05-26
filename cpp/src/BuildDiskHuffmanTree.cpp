#include "DiskHuffmanTree.h"
#include <iostream>


int main(int argc, char* argv[]) {
    std::string directory = (argc > 1) ? argv[1] : ".";
    std::string outputFile = (argc > 2) ? argv[2] : "huffman_tree.dat";

    DiskHuffmanTree::buildAndSaveTree(directory, outputFile);

    return 0;
}

