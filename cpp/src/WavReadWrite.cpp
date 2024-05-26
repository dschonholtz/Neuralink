#include "WavReadWrite.h"
#include <fstream>
#include <iostream>

std::vector<int16_t> WavReadWrite::readWavFile(const std::string& filename, WavHeader& header) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file for reading");
    }

    file.read(reinterpret_cast<char*>(&header), sizeof(WavHeader));
    if (std::string(header.riff, 4) != "RIFF" || std::string(header.wave, 4) != "WAVE") {
        throw std::runtime_error("Invalid WAV file");
    }

    std::vector<int16_t> data(header.dataSize / sizeof(int16_t));
    file.read(reinterpret_cast<char*>(data.data()), header.dataSize);

    return data;
}

void WavReadWrite::writeWavFile(const std::string& filename, const WavHeader& header, const std::vector<int16_t>& data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file for writing");
    }

    file.write(reinterpret_cast<const char*>(&header), sizeof(WavHeader));
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(int16_t));
}