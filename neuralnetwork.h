#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "mathtools.h"
#include <vector>
#include <string>
#include <fstream>

namespace NAVA {

class NeuralNetwork {
private:
    std::vector<MathTools::Matrix> weights;
    std::vector<MathTools::Vector> biases;
    std::vector<int> layer_sizes;
    std::vector<std::string> activation_functions;
    std::vector<std::string> output_labels;

public:
    // Constructor
    NeuralNetwork(const std::vector<int>& layers, 
                  const std::vector<std::string>& activations);
    
    // Core functions
    MathTools::Vector predict(const MathTools::Vector& input);
    double accuracy(const std::vector<MathTools::Vector>& test_data, 
                   const std::vector<MathTools::Vector>& test_labels);
    
    // Weight management
    void saveWeights(const std::string& filename);
    void loadWeights(const std::string& filename);
    
    // Utility
    int maxProbabilityLabel(MathTools::Vector& output_values);
    void initializeWeights();
    int getOutputSize() const { return layer_sizes.back(); }
    const std::vector<std::string>& getActivations() const { return activation_functions; }
    
    // Get total number of parameters (weights + biases) in the network
    size_t getParameterCount() const;
};

}

#endif