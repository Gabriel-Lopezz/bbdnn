#include "bbdnn/DenseLayer.hpp"

namespace bbdnn {

    DenseLayer::DenseLayer(int neuronCount, ActivationPtr acFunc) : neuronCount(neuronCount), activation(std::move(acFunc)), activatedValues(neuronCount, 0.0f), unactivatedValues(neuronCount, 0.0f) {
        if (activation == nullptr)
            throw std::invalid_argument("Activation function pointer cannot be null.");
    }

    DenseLayer::DenseLayer(const DenseLayer& other) : neuronCount(other.neuronCount), activatedValues(other.activatedValues), unactivatedValues(other.unactivatedValues) {
        activation = std::move(other.activation->clone());
    }

    DenseLayer& DenseLayer::operator=(DenseLayer& other) {
        if (this == &other)
            return *this;

        neuronCount = other.neuronCount;
        activatedValues = other.activatedValues;
        unactivatedValues = other.unactivatedValues;
        activation = std::move(other.activation->clone());

        return *this;
    }

    DenseLayer::~DenseLayer() {}

    const ActivationPtr& DenseLayer::getActivationFunction() const {
        return activation;
    }

    Vector DenseLayer::getActivatedVector() const {
        return activatedValues;
    }

    Vector DenseLayer::getUnactivatedVector() const {
        return unactivatedValues;
    }

    float DenseLayer::getActivatedValue(int i) const {
        return activatedValues[i];
    }

    float DenseLayer::getUnactivatedValue(int i) const {
        return unactivatedValues[i];
    }

    int DenseLayer::size() const {
        return neuronCount;
    }

    void DenseLayer::setActivationFunction(ActivationPtr& func) {
        activation = std::move(func);
    }

    void DenseLayer::setActivatedValues(const Vector& newVals) {
        if (newVals.size() != this->neuronCount)
            throw std::invalid_argument("Input vector size must be equal to layer size");

        for (int i = 0; i < newVals.size(); i++) {
            activatedValues[i] = newVals[i];
        }
    }

    void DenseLayer::setUnactivatedValues(const Vector& newVals) {
        if (newVals.size() != this->neuronCount)
            throw std::invalid_argument("Input vector size must be equal to layer size");

        for (int i = 0; i < newVals.size(); i++) {
            unactivatedValues[i] = newVals[i];
        }
    }

}
