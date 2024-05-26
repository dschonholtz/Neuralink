#include <iostream>
#include <fstream>
#include <bitset>
#include "DiskHuffmanTree.h"
#include "WavReadWrite.h"

std::string readBinaryFile(const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Error opening input file: " << filename << std::endl;
        return "";
    }

    std::string encodedData;
    char byte;
    while (inFile.get(byte)) {
        std::bitset<8> bits(static_cast<unsigned char>(byte));
        encodedData += bits.to_string();
    }

    inFile.close();
    return encodedData;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_encoded_file> <output_wav_file>" << std::endl;
        return 1;
    }

    std::string inputEncodedFile = argv[1];
    std::string outputWavFile = argv[2];

    // Read the encoded data from the input file
    std::string encodedData = readBinaryFile(inputEncodedFile);
    if (encodedData.empty()) {
        return 1;
    }

    // Load the Huffman tree from disk
    DiskHuffmanTree huffmanTree;
    huffmanTree.readTreeFromDisk("huffman_tree.dat");

    // Decode the encoded data
    std::vector<int16_t> decodedData;
    huffmanTree.decode(encodedData, decodedData);

    // Write the decoded data to the output WAV file
    WavHeader header;
    // Assuming the header is stored or can be reconstructed, otherwise it needs to be passed or stored separately
    WavReadWrite::writeWavFile(outputWavFile, header, decodedData);

    std::cout << "Successfully decoded " << inputEncodedFile << " to " << outputWavFile << std::endl;
    return 0;
}
