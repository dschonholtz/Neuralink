#include "HuffmanCompressor.h"
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <cstdint>

struct WavHeader {
    char riff[4];                // "RIFF"
    uint32_t overall_size;       // overall size of file in bytes
    char wave[4];                // "WAVE"
    char fmt_chunk_marker[4];    // "fmt " string with trailing null char
    uint32_t length_of_fmt;      // length of the format data
    uint16_t format_type;        // format type
    uint16_t channels;           // number of channels
    uint32_t sample_rate;        // sampling rate (blocks per second)
    uint32_t byterate;           // SampleRate * NumChannels * BitsPerSample/8
    uint16_t block_align;        // NumChannels * BitsPerSample/8
    uint16_t bits_per_sample;    // bits per sample
    char data_chunk_header[4];   // "data" string
    uint32_t data_size;          // size of the data section
};

size_t getWavDataSize(const std::string& filePath) {
    std::ifstream inFile(filePath, std::ios::binary);
    if (!inFile) {
        std::cerr << "Error opening WAV file: " << filePath << std::endl;
        exit(1);
    }

    WavHeader header;
    inFile.read(reinterpret_cast<char*>(&header), sizeof(WavHeader));
    inFile.close();

    return header.data_size;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << std::endl;
        return 1;
    }

    std::ifstream inFile(argv[1], std::ios::binary);
    if (!inFile) {
        std::cerr << "Error opening input file: " << argv[1] << std::endl;
        return 1;
    }

    std::vector<uint8_t> loadedData((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());
    inFile.close();

    size_t originalDataSize = getWavDataSize(argv[1]);

    HuffmanCompressor compressor;
    HuffmanTree treeBuilder;
    std::unordered_map<int16_t, int> frequencies;
    for (uint8_t byte : loadedData) {
        frequencies[static_cast<int16_t>(byte)]++;
    }

    std::shared_ptr<HuffmanNode> root = treeBuilder.buildTree(frequencies);
    std::vector<int16_t> decompressedData = compressor.decompress(loadedData, root.get(), originalDataSize);

    std::ofstream outFile(argv[2], std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening output file: " << argv[2] << std::endl;
        return 1;
    }

    outFile.write(reinterpret_cast<char*>(decompressedData.data()), decompressedData.size() * sizeof(int16_t));
    outFile.close();

    return 0;
}
