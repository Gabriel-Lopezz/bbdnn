#ifndef LAYERCONNECTION_HPP
#define LAYERCONNECTION_HPP

#include <iostream>
#include <cstdint>
#include "bbdnn/Matrix.hpp"
#include "bbdnn/DenseLayer.hpp"

namespace bbdnn {

    /// Connection between two dense layers with weights and biases.
    class LayerConnection {
        DenseLayer& inLayer;
        DenseLayer& outLayer;
    
        Matrix weights;
        Vector biases;

        // Automatically initializes based on activation Function of outLayer and seed
        void initializeWeights(const uint_fast32_t& randomSeed);
    public:
        /// Construct a connection with optional auto-initialization.
        LayerConnection(DenseLayer& InLayer, DenseLayer& OutLayer, bool autoInitWeights = false, uint_fast32_t randomSeed = 0);
        /// Construct a connection with explicit weights and biases.
        LayerConnection(DenseLayer& InLayer, DenseLayer& OutLayer, Matrix& Weights, float Biases[]);
        /// Copy-construct a connection.
        LayerConnection(const LayerConnection& other);
        /// Destroy the connection.
        ~LayerConnection();

        /// Assign from another connection.
        LayerConnection& operator=(const LayerConnection& other);

        /// Get the output of the out layer.
        Vector getOutput() const;

        /// Get the weight matrix.
        const Matrix& getWeights() const;
        
        /// Set the weight matrix.
        void setWeights(const Matrix& newMatrix);

        /// Get the bias vector.
        Vector getBiases() const;
        /// Set biases from a raw array.
        void setBiases(float newBiases[], int newBiasesSize);
        /// Set biases from a vector.
        void setBiases(const Vector& newBiases);

        /// Get weight at (j, i).
        float weightAt(int j, int i) const;
        /// Get bias at index.
        float biasAt(int i) const;

        /// Forward propagate through this connection.
        void forwardPropogate();
    };

}

#endif
