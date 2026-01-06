#include "bbdnn/NeuralNetwork.hpp"
#include <algorithm>

namespace bbdnn {

    NeuralNetwork::NeuralNetwork(uint_fast32_t RngSeed, std::vector<DenseLayer> Layers): layers(std::move(Layers)), layerCount(layers.size()), rngSeed(RngSeed) {
        if (layerCount < 2)
            throw std::invalid_argument("Neural Network input vector must contain at least 2 layers");

        connections.reserve(layerCount - 1);

        for (int i = 0; i < layerCount - 1; i++) {
            DenseLayer& inLayer = layers[i];
            DenseLayer& outLayer = layers[i+1];

            // Add connection to connections list
            connections.push_back(LayerConnection(inLayer, outLayer, true, rngSeed));
        }
    }

    const DenseLayer& NeuralNetwork::getLayer(int l) const {
        return layers[l];
    }

    float NeuralNetwork::getNeuronValue(int l, int i) const {
        return layers[l].getActivatedValue(i);
    }

    NeuralNetwork::~NeuralNetwork() {
        // std::cerr << "Deleting Neural Network" << std::endl;
    }

    void NeuralNetwork::setInput(Vector input)
    {   
        DenseLayer& inputLayer = layers[0];

        inputLayer.setActivatedValues(input);
    }

    Vector NeuralNetwork::output() const {
        int last = layerCount - 1;

        return layers[last].getActivatedVector();
    }

    void NeuralNetwork::forwardPropogate() {
        for (LayerConnection& connection : connections)
            connection.forwardPropogate();
    }

    Vector NeuralNetwork::getLayerErrorSensitivity(int layerIndex, const Vector& nextLayerSensitivity) {
        if (layerIndex < 0 || layerIndex >= layerCount - 1)
            throw std::out_of_range("Layer index out of range for getting layer error sensitivity.");

        LayerConnection& connection = connections[layerIndex];

        DenseLayer& currentLayer = layers[layerIndex];
        DenseLayer& nextLayer = layers[layerIndex + 1];

        if (nextLayerSensitivity.size() != nextLayer.size())
            throw std::invalid_argument("Next layer sensitivity vector size does not match next layer size.");

        Vector curDerivatives(currentLayer.size());

        // Look at every neuron
        for (int i = 0; i < currentLayer.size(); i++)
        {
            float sensOfLayer = 0; // The sensitivity of the next layer wrt the current activation
        
            float activationDerivative = 1;
        
            float preactivationVal = currentLayer.getUnactivatedValue(i);
            activationDerivative = currentLayer.getActivationFunction()->derive(preactivationVal);

            for (int j = 0; j < nextLayer.size(); j++)
            {
                float deltaUnactivatedNeuron = nextLayerSensitivity[j] * connection.weightAt(j, i);   
                sensOfLayer += deltaUnactivatedNeuron;
            }

            sensOfLayer *= activationDerivative;
        
            curDerivatives[i] = sensOfLayer;
        }

        return curDerivatives;
    }

    std::tuple<std::vector<Matrix>, std::vector<Vector>, float> NeuralNetwork::backPropagate(Vector expected, 
        float learningRate) {
        DenseLayer& last = layers.back();
        int lastSize = last.size();

        if (lastSize != expected.size())
            throw std::invalid_argument("Expected values must be the sme size as output layer.");

        Vector predicted = last.getActivatedVector();

        // Evaluate accuracy
        float residualSquared = 0;
        for (int i = 0; i < outputSize(); i++)
            residualSquared += (expected[i] - predicted[i]) * (expected[i] - predicted[i]);
        

        std::vector<Matrix> weightsDiff;
        std::vector<Vector> biasesDiff;

        // Sensitivity of next layers' unactivated neurons
        Vector forwardDerivatives(lastSize);

        // Derivative of loss function
        for (int i = 0; i < lastSize; i++)
        {
            float dE_dC = -2 * (expected[i] - predicted[i]);
            float activationDerivative = 1;

        
            float preactivationVal = last.getUnactivatedValue(i);
            activationDerivative = last.getActivationFunction()->derive(preactivationVal);

            float errorSens = dE_dC * activationDerivative;
            forwardDerivatives[i] = errorSens;
        }

        Matrix weightsGradient = layers[layerCount-2].getActivatedVector() * forwardDerivatives.transposed();
        Vector biasGradient = forwardDerivatives;

        Matrix costweightsDiff = weightsGradient * learningRate;
        Vector costBiasesDiff = biasGradient * learningRate;

        weightsDiff.push_back(costweightsDiff);
        biasesDiff.push_back(costBiasesDiff);

        // Derivatives of hidden layers
        for (int l = layerCount - 2; l >= 1; l--)
        {
            LayerConnection& backwardConnection = connections[l-1];

            DenseLayer& currentLayer = layers[l];
            DenseLayer& prevLayer = layers[l-1];
            Vector curDerivatives(currentLayer.size(), 0.0f);

            curDerivatives = getLayerErrorSensitivity(l, forwardDerivatives);

            Matrix weightsGradient =  prevLayer.getActivatedVector() * curDerivatives.transposed();
            Vector& biasGradient = curDerivatives;

            Matrix layerWeightsDiff = (weightsGradient * learningRate);
            Vector layerBiasDiff = biasGradient * learningRate;

            weightsDiff.push_back(layerWeightsDiff);
            biasesDiff.push_back(layerBiasDiff);

            forwardDerivatives = curDerivatives;
        }

        // reverse to match layer order
        std::reverse(weightsDiff.begin(), weightsDiff.end());
        std::reverse(biasesDiff.begin(), biasesDiff.end());

        return { weightsDiff, biasesDiff, residualSquared };
    }

    std::pair<std::vector<Matrix>, std::vector<Vector>> NeuralNetwork::takeStep(const std::vector<Matrix>& deltaWeights, const std::vector<Vector>& deltaBiases) {
        int connectionCount = connections.size();
        
        if (deltaWeights.size() != connectionCount || deltaBiases.size() != connectionCount)
            throw std::invalid_argument("Delta weights/biases size must match number of connections.");

        std::vector<Matrix> newWeights(connectionCount);
        std::vector<Vector> newBiases(connectionCount);

        for (int l = 0; l < connectionCount; l++) {
            LayerConnection& connection = connections[l];

            Matrix currentWeights = connection.getWeights();
            Vector currentBiases = connection.getBiases();

            newWeights[l] = currentWeights - deltaWeights[l];
            newBiases[l] = currentBiases - deltaBiases[l];
        }

        return { newWeights, newBiases };
    }

    void NeuralNetwork::updateParameters(std::vector<Matrix> newWeights, std::vector<Vector> newBiases) {
        if (newWeights.size() != connections.size() || newBiases.size() != connections.size())
            throw std::invalid_argument("New weights/biases size must match number of connections.");

        for (size_t l = 0; l < connections.size(); l++) {
            connections[l].setWeights(newWeights[l]);
            connections[l].setBiases(newBiases[l]);
        }
    }

    std::vector<float> NeuralNetwork::train(const std::vector<Vector>& trainingFeatures, const std::vector<Vector>& trainingLabels, float learningRate, int epochs, bool isStochastic) {
        if (epochs <= 0)
            throw std::invalid_argument("Must have 1+ epochs to train model");
        
        if (trainingFeatures.size() != trainingLabels.size())
            throw std::invalid_argument("Training features and training labels must be of same count.");
        
        if (trainingFeatures.empty())
            throw std::invalid_argument("Training dataset must not be empty.");
        
            
        size_t connectionCount = connections.size();
        size_t exampleCount = trainingFeatures.size();
        float exampleWeight = 1.0f / static_cast<float>(exampleCount);
        
        std::vector<float> metrics;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            std::vector<Matrix> cumDeltaWeights(connectionCount);
            std::vector<Vector> cumDeltaBiases(connectionCount);

            if (!isStochastic)
            {
                // Set shape and zero-out 'new<Parameter>' matrices
                std::transform(connections.begin(), connections.end(), cumDeltaWeights.begin(), 
                    [](const LayerConnection& lc){ return Matrix(lc.getWeights().Rows(), lc.getWeights().Cols(), 0.0f); });

                std::transform(connections.begin(), connections.end(), cumDeltaBiases.begin(), 
                    [](const LayerConnection& lc){ return Vector(lc.getBiases().size(), 0.0f); });
            }

            for (size_t exampleInd = 0; exampleInd < exampleCount; exampleInd++) {
                const Vector& expectedOut = trainingLabels[exampleInd];

                setInput(trainingFeatures[exampleInd]);
                forwardPropogate();
                auto [deltaWeights, deltaBiases, outMetric] = backPropagate(expectedOut, learningRate);
                metrics.push_back(outMetric);

                // Update params based on method
                if (isStochastic) { // SGD
                    auto [newWeights, newBiases] = takeStep(deltaWeights, deltaBiases);
                    updateParameters(newWeights, newBiases);
                }
                else { // Full-batch Accumulation
                    for (size_t l = 0; l < connectionCount; l++) {
                        cumDeltaWeights[l] += deltaWeights[l] * exampleWeight;
                        cumDeltaBiases[l] += deltaBiases[l] * exampleWeight;
                    }
                }
            }

            if (!isStochastic) { // Full-batch Update
                auto [newWeights, newBiases] = takeStep(cumDeltaWeights, cumDeltaBiases);
                updateParameters(newWeights, newBiases);
            }
        }
        
        return metrics;
    }

    std::vector<float> NeuralNetwork::evaluate(const std::vector<Vector>& testFeatures, const std::vector<Vector>& testLabels) {
        if (testFeatures.size() != testLabels.size())
            throw std::invalid_argument("Test features and test labels must be of same count.");
        
        if (testFeatures.empty())
            throw std::invalid_argument("Test dataset must not be empty.");

        size_t exampleCount = testFeatures.size();
        std::vector<float> metrics;
        metrics.reserve(exampleCount);

        for (size_t exampleInd = 0; exampleInd < exampleCount; exampleInd++) {
            const Vector& expectedOut = testLabels[exampleInd];

            setInput(testFeatures[exampleInd]);
            forwardPropogate();
            DenseLayer& last = layers.back();
            Vector predicted = last.getActivatedVector();

            float residulSquared = 0;
            for (int i = 0; i < outputSize(); i++)
                residulSquared += (expectedOut[i] - predicted[i]) * (expectedOut[i] - predicted[i]);
            
            metrics.push_back(residulSquared);
        }

        return metrics;
    }

    Vector NeuralNetwork::predict(const Vector& input) {
        setInput(input);
        forwardPropogate();
        Vector prediction = output();
        clear();

        return prediction;
    }

    void NeuralNetwork::clear() {
        for (auto& layer : layers)
            layer.setActivatedValues(Vector(layer.size(), 0.0f));
    }

    const std::vector<LayerConnection>& NeuralNetwork::getConnections() const {
        return connections;
    }

    int NeuralNetwork::size() const {
        return layerCount;
    }

    int NeuralNetwork::inputSize() const
    {
        return layers[0].size();
    }

    int NeuralNetwork::outputSize() const
    {
        int last = layerCount - 1;

        return layers[last].size();
    }

}
