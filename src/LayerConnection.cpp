#include "bbdnn/LayerConnection.hpp"

namespace bbdnn {

    LayerConnection::LayerConnection(DenseLayer& InLayer, DenseLayer& OutLayer, bool autoInitializeWeights, 
        uint_fast32_t randomSeed) : inLayer(InLayer), outLayer(OutLayer), biases(outLayer.size(), 0.0f) {
        // Create based on activation function
        if (autoInitializeWeights)
            initializeWeights(randomSeed);
        else
            weights = Matrix(inLayer.size(), outLayer.size());
    }

    LayerConnection::LayerConnection(DenseLayer& InLayer, DenseLayer& OutLayer, Matrix& Weights, float Biases[]) : inLayer(InLayer), outLayer(OutLayer), weights(Weights), biases(Biases, outLayer.size()) {
    }

    LayerConnection::LayerConnection(const LayerConnection& other) : inLayer(other.inLayer), outLayer(other.outLayer), weights(other.weights), biases(other.biases) {
    }

    LayerConnection& LayerConnection::operator=(const LayerConnection& other) {
        inLayer = other.inLayer;
        outLayer = other.outLayer;
        weights = other.weights;
        biases = other.biases;

        return *this;
    }

    LayerConnection::~LayerConnection() {}

    const Matrix& LayerConnection::getWeights() const {
        return weights;
    }

    void LayerConnection::setWeights(const Matrix& newMatrix) {
        if (newMatrix.Rows() != weights.Rows() || newMatrix.Cols() != weights.Cols())
            throw std::invalid_argument("The given Matrix's dimensions do not the specifications for the layer connection.");

        weights = newMatrix;
    }

    Vector LayerConnection::getBiases() const {
        return biases;
    }

    void LayerConnection::setBiases(float newBiases[], int newBiasesSize) {
        if (newBiasesSize != outLayer.size())
            throw std::invalid_argument("The given biases array is not the appropriate size for the layer connection.");

        for (int i = 0; i < outLayer.size(); i++)
            biases[i] = newBiases[i];
    }

    void LayerConnection::setBiases(const Vector& newBiases) {
        if (newBiases.size() != outLayer.size())
            throw std::invalid_argument("The given biases array is not the appropriate size for the layer connection.");

        biases = newBiases;
    }

    void LayerConnection::initializeWeights(const uint_fast32_t& randomSeed) {
        // Use Kaiming initialization for ReLU and LeakyReLU
        const auto& activationFunc = outLayer.getActivationFunction();

        if (dynamic_cast<ReLUActivation*>(activationFunc.get()) != nullptr ||
            dynamic_cast<LeakyReLUActivation*>(activationFunc.get()) != nullptr) {
            weights = Matrix::kaimingMatrix(inLayer.size(), outLayer.size(), randomSeed);
            return;
        }

        // Xavier initialization for other activations (Linear, Sigmoid, Logistic, Tanh)
        else {
            weights = Matrix::xavierMatrix(inLayer.size(), outLayer.size(), randomSeed);
        }
    }

    float LayerConnection::weightAt(int j, int i) const {
        return weights.at(i,j);
    }

    float LayerConnection::biasAt(int i) const {
        return biases[i];
    }

    void LayerConnection::forwardPropogate() {
        int outSize = outLayer.size();

        // Get the dot product of the layers; gets z_i at layer l
        Vector products = weights.applyMatrix(inLayer.getActivatedVector());
    
        Vector activated(outSize);
        Vector unactivated(outSize);
    
        // Calculate value of a_i at layer l
        for (int i = 0; i < outSize; i++)
        {
            // Calculate Z = W.P + b ; where P is outut of prev. layer, or A^(l-1)
            double nueronVal = products[i] + biases[i];

            unactivated[i] = nueronVal;

            // Get A = σ(Z)
            nueronVal = (*outLayer.getActivationFunction())(nueronVal);

            activated[i] = nueronVal;
        }
    
        outLayer.setActivatedValues(activated);
        outLayer.setUnactivatedValues(unactivated);
    }

    Vector LayerConnection::getOutput() const {
        return outLayer.getActivatedVector();
    }

}
