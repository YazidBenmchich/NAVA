#ifndef DATALOADER_H
#define DATALOADER_H

#include "mathtools.h"
#include <vector>
#include <string>

namespace NAVA {

class DataLoader {
public:
    // MNIST dataset loading
    static std::vector<MathTools::Vector> loadMNISTImages(const std::string& filename);
    static std::vector<MathTools::Vector> loadMNISTLabels(const std::string& filename);
    
    // CSV data loading
    static std::vector<MathTools::Vector> loadCSV(const std::string& filename, bool has_header = false);
    
    // Data preprocessing utilities
    static MathTools::Vector normalizeVector(const MathTools::Vector& input, double min_val = 0.0, double max_val = 1.0);
    static std::vector<MathTools::Vector> normalizeDataset(const std::vector<MathTools::Vector>& dataset);
    
    // Label encoding
    static MathTools::Vector oneHotEncode(int label, int num_classes);
    static std::vector<MathTools::Vector> encodeLabels(const std::vector<int>& labels, int num_classes);
    
private:
    static uint32_t reverseBytes(uint32_t value);
    static std::vector<std::string> splitString(const std::string& str, char delimiter);
};

}

#endif
