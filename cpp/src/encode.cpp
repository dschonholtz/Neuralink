#include <iostream>
#include <fstream>
#include "DiskHuffmanTree.h"
#include "WavReadWrite.h"

void writeBinaryFile(const std::string& filename, const std::string& encodedData) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening output file: " << filename << std::endl;
        return;
    }

    // Convert the string of '0' and '1' to actual bytes
    size_t dataSize = encodedData.size();
    for (size_t i = 0; i < dataSize; i += 8) {
        std::bitset<8> byte(encodedData.substr(i, 8));
        outFile.put(static_cast<unsigned char>(byte.to_ulong()));
    }

    outFile.close();
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_wav_file> <output_encoded_file>" << std::endl;
        return 1;
    }

    std::string inputWavFile = argv[1];
    std::string outputEncodedFile = argv[2];

    // Read the WAV file
    WavHeader header;
    std::vector<int16_t> wavData = WavReadWrite::readWavFile(inputWavFile, header);

    // Load the Huffman tree from disk
    DiskHuffmanTree huffmanTree;
    // huffmanTree.readTreeFromDisk("huffman_tree.dat");
    huffmanTree.buildTreeFromData(wavData);
    huffmanTree.writeTreeToDisk("huffman_tree.dat");

    // Encode the WAV data
    std::string encodedData;
    huffmanTree.encode(wavData, encodedData);

    // Write the encoded data to the output file as binary
    writeBinaryFile(outputEncodedFile, encodedData);

    std::cout << "Successfully encoded " << inputWavFile << " to " << outputEncodedFile << std::endl;
    return 0;
}
