#ifndef DENSELAYER_HPP
#define DENSELAYER_HPP

#include <functional>
#include "bbdnn/Matrix.hpp"
#include "bbdnn/Activations.hpp"

namespace bbdnn {

    /// Dense layer with an activation function, hold activated and unactivated values.
    class DenseLayer
    {
    private:
        int neuronCount;
        ActivationPtr activation;

        Vector activatedValues;
        Vector unactivatedValues;

    public:
        /// Construct a layer with neuron count and activation.
        DenseLayer(int neuronCount, ActivationPtr acFunc);
        /// Copy-construct a layer.
        DenseLayer(const DenseLayer& other);
        /// Destroy the layer.
        ~DenseLayer();

        /// Assign from another layer.
        DenseLayer& operator=(DenseLayer& other);

        /// Get the activation function.
        const ActivationPtr& getActivationFunction() const;
        /// Set the activation function.
        void setActivationFunction(ActivationPtr& newActivation);

        /// Get activated output values.
        Vector getActivatedVector() const;
        /// Get pre-activation values.
        Vector getUnactivatedVector() const;
        /// Get activated value at index.
        float getActivatedValue(int i) const;
        /// Get pre-activation value at index.
        float getUnactivatedValue(int i) const;

        /// Set activated values.
        void setActivatedValues(const Vector& newVals);
        /// Set pre-activation values.
        void setUnactivatedValues(const Vector& newVals);
    
        /// Number of neurons in the layer.
        int size() const;
    };

}

#endif
