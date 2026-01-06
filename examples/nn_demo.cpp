#include <iostream>
#include <utility>
#include <vector>

#include "bbdnn/NeuralNetwork.hpp"

using namespace bbdnn;

int main() {
    int seed = 42;

    NeuralNetwork nn(seed, {
        DenseLayer(2, Activation::Linear()),
        DenseLayer(8, Activation::Tanh()),
        DenseLayer(8, Activation::Tanh()),
        DenseLayer(1, Activation::Sigmoid()),
    });

    std::vector<Vector> features {
        Vector{0.0f, 0.0f},
        Vector{0.0f, 1.0f},
        Vector{1.0f, 0.0f},
        Vector{1.0f, 1.0f},
    };

    std::vector<Vector> labels {
        Vector{0.0f},
        Vector{1.0f},
        Vector{1.0f},
        Vector{0.0f},
    };

    const float learningRate = 0.05f;
    const int epochs = 20000;

    // Performs full-batch training `epochs` times
    nn.train(features, labels, learningRate, epochs, false);

    for (const Vector& feature : features) {
        Vector prediction = nn.predict(feature);
        std::cout << "Input: ";
        
        for (int i = 0; i < feature.size(); i++)
            std::cout << feature[i] << " ";

        std::cout << "=> Prediction: " << prediction[0] << std::endl;
    }

    return 0;
}
