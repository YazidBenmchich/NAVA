#include "mathtools.h"

namespace MathTools {
Vector softmax(const Vector& input)
{
    Vector output;
    output.data = (input.data.array()-input.data.maxCoeff()).exp();
    output.data = output.data.array() / output.data.array().sum();
    return output;
}

Vector ReLu(const Vector& input)
{
    Vector output;
    output.data = input.data.array().max(0);
    return output;
}
Vector Activation(const Vector& input, const std::string& type){
    switch(type[0]){
        case 'softmax': return softmax(input);
        case 'ReLu': return ReLu(input);
        case 'tanh' : return tanh(input);
        case 'sigmoid': return sigmoid(input);
        default: throw std::invalid_argument("Unknown activation type");
    }
}
inline Vector operator+(const Vector& a, const Vector& b){
    Vector result;
    result.data = a.data + b.data;
    return result;
}
inline Vector operator*(const Matrix& a, const Vector& b){
    Vector result;
    result.data = a.data.transpose() * b.data;
    return result;
}

double vectorNorm(const Vector& v) {
    return v.data.norm();
}

double vectorMean(const Vector& v) {
    return v.data.mean();
}

Vector vectorAbs(const Vector& v) {
    Vector result;
    result.data = v.data.cwiseAbs();
    return result;
}

std::vector<std::vector<double>> Matrix::toCppVectorForm(){
    std::vector<std::vector<double>> result(data.rows(), std::vector<double>(data.cols()));
    for(int i = 0; i < data.rows(); ++i)
        for(int j = 0; j < data.cols(); ++j)
            result[i][j] = data(i,j);
    return result;
}
std::vector<double> Vector::toCppVectorForm(){
    std::vector<double> result(data.size());
    for(int i = 0; i < data.size(); ++i)
        result[i] = data(i);
    return result;

}