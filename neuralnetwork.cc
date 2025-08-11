#include "neuralnetwork.h"
#include <random>
#include <algorithm>
#include <stdexcept>

namespace NAVA {

NeuralNetwork::NeuralNetwork(const std::vector<int>& layers, 
                           const std::vector<std::string>& activations)
    : layer_sizes(layers), activation_functions(activations) {
    
    if (layers.size() < 2) {
        throw std::invalid_argument("Neural network must have at least 2 layers");
    }
    
    // Check activation functions count
    // We need (layers.size() - 2) activations for hidden layers
    size_t expected_activations = layers.size() - 1;
    if (activations.size() != expected_activations) {
        throw std::invalid_argument("Number of activation functions must equal number of hidden layers + 1 (" 
                                  + std::to_string(expected_activations) + ")");
    }
    
    initializeWeights();
}


MathTools::Vector NeuralNetwork::predict(const MathTools::Vector& input) {
    MathTools::Vector current = input;
    for (size_t i = 0; i < weights.size(); ++i) {
        
        MathTools::Vector weighted;
        weighted.data = current.data.transpose() * weights[i].data;
        
        // Add bias
        MathTools::Vector layer_output;
        layer_output.data = weighted.data + biases[i].data;
        
        // Apply activation function
        if (i == weights.size() - 1) {
            // Output layer - always use softmax for classification
            current = MathTools::softmax(layer_output);
        } else {
            // Hidden layers - use specified activation for this layer
            current = MathTools::Activation(layer_output, activation_functions[i]);
        }
    }
    
    return current;
}

// ...existing accuracy() code stays the same...

void NeuralNetwork::saveWeights(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    // Save layer sizes
    size_t num_layers = layer_sizes.size();
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
    file.write(reinterpret_cast<const char*>(layer_sizes.data()), 
               num_layers * sizeof(int));
    
    // Save activation functions
    size_t num_activations = activation_functions.size();
    file.write(reinterpret_cast<const char*>(&num_activations), sizeof(num_activations));
    
    for (const auto& activation : activation_functions) {
        size_t activation_length = activation.length();
        file.write(reinterpret_cast<const char*>(&activation_length), sizeof(activation_length));
        file.write(activation.c_str(), activation_length);
    }
    
    // Save weights and biases (same as before)
    for (size_t i = 0; i < weights.size(); ++i) {
        // Save weight matrix dimensions
        int rows = weights[i].data.rows();
        int cols = weights[i].data.cols();
        file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        
        // Save weight data
        file.write(reinterpret_cast<const char*>(weights[i].data.data()),
                   rows * cols * sizeof(double));
        
        // Save bias vector size and data
        int bias_size = biases[i].data.size();
        file.write(reinterpret_cast<const char*>(&bias_size), sizeof(bias_size));
        file.write(reinterpret_cast<const char*>(biases[i].data.data()),
                   bias_size * sizeof(double));
    }
    
    file.close();
}

void NeuralNetwork::loadWeights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    
    // Load layer sizes
    size_t num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    layer_sizes.resize(num_layers);
    file.read(reinterpret_cast<char*>(layer_sizes.data()), 
              num_layers * sizeof(int));
    
    // Load activation functions
    size_t num_activations;
    file.read(reinterpret_cast<char*>(&num_activations), sizeof(num_activations));
    activation_functions.clear();
    activation_functions.resize(num_activations);
    
    for (size_t i = 0; i < num_activations; ++i) {
        size_t activation_length;
        file.read(reinterpret_cast<char*>(&activation_length), sizeof(activation_length));
        activation_functions[i].resize(activation_length);
        file.read(&activation_functions[i][0], activation_length);
    }
    
    // Load weights and biases (same as before)
    weights.clear();
    biases.clear();
    
    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
        // Load weight matrix
        int rows, cols;
        file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        
        MathTools::Matrix weight;
        weight.data = Eigen::MatrixXd(rows, cols);
        file.read(reinterpret_cast<char*>(weight.data.data()),
                  rows * cols * sizeof(double));
        weights.push_back(weight);
        
        // Load bias vector
        int bias_size;
        file.read(reinterpret_cast<char*>(&bias_size), sizeof(bias_size));
        
        MathTools::Vector bias;
        bias.data = Eigen::VectorXd(bias_size);
        file.read(reinterpret_cast<char*>(bias.data.data()),
                  bias_size * sizeof(double));
        biases.push_back(bias);
    }
    
    file.close();
}

size_t NeuralNetwork::getParameterCount() const {
    size_t total_params = 0;
    
    for (size_t i = 0; i < weights.size(); ++i) {
        total_params += weights[i].data.rows() * weights[i].data.cols();
        total_params += biases[i].data.size();
    }
    
    return total_params;
}

}