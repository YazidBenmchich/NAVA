#ifndef MATHTOOLS_H
#define MATHTOOLS_H

#include <cmath>
#include <vector>
#include <iostream>
#include <Eigen/Dense>

namespace MathTools {
struct Matrix
{
    Eigen::MatrixXd data;
    std::vector<std::vector<double>> toCppVectorForm();
};
struct Vector
{
    Eigen::VectorXd data;
    std::vector<double> toCppVectorForm();
};

// Activation functions
Vector softmax(const Vector& input);
Vector ReLu(const Vector& input);
Vector tanh(const Vector& input);
Vector sigmoid(const Vector& input);
Vector Activation(const Vector& input, const std::string& type);

// Matrix operations
inline Vector operator+(const Vector& a, const Vector& b);
inline Vector operator*(const Matrix& a, const Vector& b);

// Utility functions
double vectorNorm(const Vector& v);
double vectorMean(const Vector& v);
Vector vectorAbs(const Vector& v);
double cosineSimilarity(const Vector& a, const Vector& b);

}

#endif