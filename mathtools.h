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
};
struct Vector
{
    Eigen::VectorXd data;
};

// Activation functions
Vector softmax(const Vector& input);
Vector ReLu(const Vector& input);
Vector tanh(const Vector& input);
Vector sigmoid(const Vector& input);
Vector Activation(const Vector& input, const std::string& type);

// Matrix operations
inline Vector operator+(const Vector& a, const Vector& b);
inline Vector operator*(const Vector& a, const Matrix& b);



}

#endif