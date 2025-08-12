#include "dataloader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace NAVA {

std::vector<MathTools::Vector> DataLoader::loadMNISTImages(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open MNIST image file: " + filename);
    }
    
    uint32_t magic_number, num_images, rows, cols;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    
    // Convert from big-endian to little-endian
    magic_number = reverseBytes(magic_number);
    num_images = reverseBytes(num_images);
    rows = reverseBytes(rows);
    cols = reverseBytes(cols);
    
    if (magic_number != 2051) {
        throw std::runtime_error("Invalid MNIST image file format");
    }
    
    std::vector<MathTools::Vector> images;
    images.reserve(num_images);
    
    for (uint32_t i = 0; i < num_images; ++i) {
        MathTools::Vector image;
        image.data = Eigen::VectorXd(rows * cols);
        
        for (uint32_t j = 0; j < rows * cols; ++j) {
            unsigned char pixel;
            file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
            image.data[j] = static_cast<double>(pixel) / 255.0; // Normalize to [0,1]
        }
        
        images.push_back(image);
    }
    
    file.close();
    return images;
}

std::vector<MathTools::Vector> DataLoader::loadMNISTLabels(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open MNIST label file: " + filename);
    }
    
    uint32_t magic_number, num_labels;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));
    
    magic_number = reverseBytes(magic_number);
    num_labels = reverseBytes(num_labels);
    
    if (magic_number != 2049) {
        throw std::runtime_error("Invalid MNIST label file format");
    }
    
    std::vector<int> raw_labels;
    raw_labels.reserve(num_labels);
    
    for (uint32_t i = 0; i < num_labels; ++i) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        raw_labels.push_back(static_cast<int>(label));
    }
    
    file.close();
    return encodeLabels(raw_labels, 10); // MNIST has 10 classes (0-9)
}

std::vector<MathTools::Vector> DataLoader::loadCSV(const std::string& filename, bool has_header) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open CSV file: " + filename);
    }
    
    std::vector<MathTools::Vector> data;
    std::string line;
    
    // Skip header if present
    if (has_header && std::getline(file, line)) {
        // Header skipped
    }
    
    while (std::getline(file, line)) {
        std::vector<std::string> values = splitString(line, ',');
        
        MathTools::Vector row;
        row.data = Eigen::VectorXd(values.size());
        
        for (size_t i = 0; i < values.size(); ++i) {
            try {
                row.data[i] = std::stod(values[i]);
            } catch (const std::exception&) {
                row.data[i] = 0.0; // Default value for invalid entries
            }
        }
        
        data.push_back(row);
    }
    
    file.close();
    return data;
}

MathTools::Vector DataLoader::normalizeVector(const MathTools::Vector& input, double min_val, double max_val) {
    MathTools::Vector normalized;
    
    double input_min = input.data.minCoeff();
    double input_max = input.data.maxCoeff();
    double range = input_max - input_min;
    
    if (range == 0) {
        normalized.data = Eigen::VectorXd::Constant(input.data.size(), min_val);
    } else {
        normalized.data = (input.data.array() - input_min) / range;
        normalized.data = normalized.data * (max_val - min_val) + min_val;
    }
    
    return normalized;
}

std::vector<MathTools::Vector> DataLoader::normalizeDataset(const std::vector<MathTools::Vector>& dataset) {
    if (dataset.empty()) {
        return dataset;
    }
    
    std::vector<MathTools::Vector> normalized;
    normalized.reserve(dataset.size());
    
    for (const auto& sample : dataset) {
        normalized.push_back(normalizeVector(sample));
    }
    
    return normalized;
}

MathTools::Vector DataLoader::oneHotEncode(int label, int num_classes) {
    MathTools::Vector encoded;
    encoded.data = Eigen::VectorXd::Zero(num_classes);
    
    if (label >= 0 && label < num_classes) {
        encoded.data[label] = 1.0;
    }
    
    return encoded;
}

std::vector<MathTools::Vector> DataLoader::encodeLabels(const std::vector<int>& labels, int num_classes) {
    std::vector<MathTools::Vector> encoded;
    encoded.reserve(labels.size());
    
    for (int label : labels) {
        encoded.push_back(oneHotEncode(label, num_classes));
    }
    
    return encoded;
}

uint32_t DataLoader::reverseBytes(uint32_t value) {
    return ((value & 0xFF000000) >> 24) |
           ((value & 0x00FF0000) >> 8)  |
           ((value & 0x0000FF00) << 8)  |
           ((value & 0x000000FF) << 24);
}

std::vector<std::string> DataLoader::splitString(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    
    return tokens;
}

}
