#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <vector>
#include <cstdint>
#include "bbdnn/DenseLayer.hpp"
#include "bbdnn/LayerConnection.hpp"

namespace bbdnn {

    /// Feed-forward neural network composed of dense layers.
    class NeuralNetwork {
        std::vector<DenseLayer> layers;
        std::vector<LayerConnection> connections;

        int layerCount;

        uint_fast32_t rngSeed;

        Vector getLayerErrorSensitivity(int layerIndex, const Vector& nextLayerSensitivity);

    public:
        /// Construct a network from a list of layers.
        NeuralNetwork(uint_fast32_t RngSeed, std::vector<DenseLayer> Layers);

        /// Destroy the network.
        ~NeuralNetwork();

        /// Get a vector of LayerConnections objects.
        const std::vector<LayerConnection>& getConnections() const;

        /// Get a DenseLayer object by index.
        const DenseLayer& getLayer(int l) const;

        /// Get an activated neuron value.
        float getNeuronValue(int l, int i) const;

        /// Set the input layer values.
        void setInput(Vector input);

        /// Get the output layer activations.
        Vector output() const;
    
        /// Run forward propagation through all layers. Output is stored in output layer.
        void forwardPropogate();

        /// Backpropagate and return weight/bias deltas and SSR.
        std::tuple<std::vector<Matrix>, std::vector<Vector>, float> backPropagate(Vector expected, float learningRate);

        /// Compute new parameters from delta weights and biases.
        std::pair<std::vector<Matrix>, std::vector<Vector>> takeStep(const std::vector<Matrix>& deltaWeights, const std::vector<Vector>& deltaBiases);

        /// Update network parameters.
        void updateParameters(std::vector<Matrix> newWeights, std::vector<Vector> newBiases);
        
        /// Train the network and return collected metrics.
        std::vector<float> train(const std::vector<Vector>& trainingFeatures, const std::vector<Vector>& trainingLabels, float learningRate, int epochs, bool isStochastic = false);
        
        /// Evaluate the network and return metrics for each example.
        std::vector<float> evaluate(const std::vector<Vector>& testFeatures, const std::vector<Vector>& testLabels);

        /// Predict output for a single input.
        Vector predict(const Vector& input);

        /// Clear cached activations.
        void clear();

        /// Number of layers.
        int size() const;

        /// Input layer size.
        int inputSize() const;

        /// Output layer size.
        int outputSize() const;
    };

}

#endif
