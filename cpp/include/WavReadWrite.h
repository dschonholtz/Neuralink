#ifndef WAVREADWRITE_H
#define WAVREADWRITE_H

#include <vector>
#include <string>

struct WavHeader {
    char riff[4];                // "RIFF"
    uint32_t chunkSize;          // Size of the entire file in bytes minus 8 bytes
    char wave[4];                // "WAVE"
    char fmt[4];                 // "fmt "
    uint32_t subchunk1Size;      // Size of the fmt chunk
    uint16_t audioFormat;        // Audio format (1 for PCM)
    uint16_t numChannels;        // Number of channels
    uint32_t sampleRate;         // Sampling rate
    uint32_t byteRate;           // Byte rate
    uint16_t blockAlign;         // Block align
    uint16_t bitsPerSample;      // Bits per sample
    char data[4];                // "data"
    uint32_t dataSize;           // Size of the data section
};

class WavReadWrite {
public:
    static std::vector<int16_t> readWavFile(const std::string& filename, WavHeader& header);
    static void writeWavFile(const std::string& filename, const WavHeader& header, const std::vector<int16_t>& data);
};

#endif // WAVREADWRITE_H