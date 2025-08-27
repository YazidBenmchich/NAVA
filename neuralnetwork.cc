#include "neuralnetwork.h"
#include <random>
#include <algorithm>
#include <stdexcept>
#include <fstream>

namespace NAVA {

NeuralNetwork::NeuralNetwork(const std::vector<int>& layers, 
                           const std::vector<std::string>& activations)
    : layer_sizes(layers), activation_functions(activations) {
    
    if (layers.size() < 1) {
        throw std::invalid_argument("Neural network must have at least 1 layers");
    }
    
    // Check activation functions count
    // We need (layers.size() - 2) activations for hidden layers + 1 for output layer
    size_t expected_activations = layers.size();
    if (activations.size() != expected_activations) {
        throw std::invalid_argument("Number of activation functions must equal number of hidden layers + 1 (" 
                                  + std::to_string(expected_activations) + ")");
    }
    
    initializeWeights();
}


MathTools::Vector NeuralNetwork::predict(const MathTools::Vector& input) {
    MathTools::Vector current = input;
    for (size_t i = 0; i < weights.size(); ++i) {
        MathTools::Vector layer_output;
        layer_output = weights[i] * current + biases[i];
        current = MathTools::Activation(layer_output, activation_functions[i]);
    }
    return current;
}
int NeuralNetwork::maxProbabilityLabel(const MathTools::Vector& output_values) const {
    if (output_values.data.size() == 0) {
        throw std::invalid_argument("Empty vector passed to maxProbabilityLabel");
    }
    int maxIndex = 0;
    double maxValue = output_values.data[0];
    for (int i = 1; i < output_values.data.size(); ++i) {
        if (output_values.data[i] > maxValue) {
            maxValue = output_values.data[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}
double NeuralNetwork::accuracy(const std::vector<MathTools::Vector>& test_data, 
                   const std::vector<MathTools::Vector>& test_labels){
                    int accurateDate(0);
                    int dataSize(test_data.size());
                    for (int i(0); i<dataSize;i++){
                        if (maxProbabilityLabel(predict(test_data[i])) == maxProbabilityLabel(test_labels[i])){
                            accurateDate++;
                        }
                    }
                    return (double)accurateDate/dataSize;
}
void NeuralNetwork::saveWeights(const std::string& filename) {
    std::ofstream file("test.txt", std::ios::trunc);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    // Save layer sizes
    size_t num_layers = layer_sizes.size();
    file << num_layers << "\n";
    for (int size : layer_sizes) {
        file << size << " ";
    }
    file << "\n";
    // Save activation functions
    for (int i(0); i<activation_functions.size();i++){
        file << activation_functions[i]<<";";
    }
    file << "\n";
    // Save weights and biases
    //Upcoming changes
    
}

void NeuralNetwork::loadWeights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    

    size_t num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    layer_sizes.resize(num_layers);
    file.read(reinterpret_cast<char*>(layer_sizes.data()), 
              num_layers * sizeof(int));
    
   
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