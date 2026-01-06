#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include <cmath>
#include <memory>
#include <concepts>
#include <type_traits>

namespace bbdnn {

    /// Activation function interface.
    struct IActivation {
        /// Construct a base activation.
        IActivation() = default;
        /// Virtual destructor for interface.
        virtual ~IActivation() = default;
    
        /// Apply activation to a value.
        virtual float operator()(float x) const = 0;
        /// Derivative of activation at a value.
        virtual float derive(float x) const = 0;
    
        /// Clone this activation.
        virtual std::unique_ptr<IActivation> clone() const =  0;
    };

    /// Owning pointer to an activation implementation
    typedef std::unique_ptr<IActivation> ActivationPtr;

    /// Linear activation: f(x) = x.
    struct LinearActivation : public IActivation {
        /// Construct a linear activation.
        LinearActivation() = default;

        /// Apply activation to a value.
        float operator()(float x) const override;

        /// Derivative of activation at a value.
        float derive(float x) const override;
        
        /// Clone this activation.
        ActivationPtr clone() const override;
    };

    /// ReLU activation: f(x) = max(0, x).
    struct ReLUActivation : public IActivation {
        /// Construct a ReLU activation.
        ReLUActivation() = default;

        /// Apply activation to a value.
        float operator()(float x) const override;
        /// Derivative of activation at a value.
        float derive(float x) const override;
        /// Clone this activation.
        ActivationPtr clone() const override;
    };

    /// Leaky ReLU activation with configurable negative slope.
    struct LeakyReLUActivation : public IActivation {
        /// Slope for negative inputs.
        float alpha;
    
        /// Construct a Leaky ReLU activation.
        LeakyReLUActivation(float a);

        /// Apply activation to a value.
        float operator()(float x) const override;
        /// Derivative of activation at a value.
        float derive(float x) const override;
        /// Clone this activation.
        ActivationPtr clone() const override;
    };

    /// Sigmoid activation.
    struct SigmoidActivation : public IActivation {
        /// Construct a sigmoid activation.
        SigmoidActivation() = default;

        /// Apply activation to a value.
        float operator()(float x) const override;
        /// Derivative of activation at a value.
        float derive(float x) const override;
        /// Clone this activation.
        ActivationPtr clone() const override;
    };

    /// Logistic activation with configurable L and K.
    struct LogisticActivation : public IActivation {
        /// Maximum value of the curve.
        float l;
        /// Steepness of the curve.
        float k;
    
        /// Construct a logistic activation.
        LogisticActivation(float L, float K);

        /// Apply activation to a value.
        float operator()(float x) const override;
        /// Derivative of activation at a value.
        float derive(float x) const override;
        /// Clone this activation.
        ActivationPtr clone() const override;
    };

    /// Hyperbolic tangent activation.
    struct TanhActivation : public IActivation {
        /// Construct a tanh activation.
        TanhActivation() = default;

        /// Apply activation to a value.
        float operator()(float x) const override;
        /// Derivative of activation at a value.
        float derive(float x) const override;
        /// Clone this activation.
        ActivationPtr clone() const override;
    };

    /// Concept for activation implementations.
    template<typename T>
    concept ActivationDerived = std::derived_from<T, IActivation> ;

    /// make_unique wrapper for cleaner activation construction.
    template <ActivationDerived ActivationType, typename... Args>
    ActivationPtr make_activation(Args&&... args) {
        return std::make_unique<ActivationType>(std::forward<Args>(args)...);
    }

    /// Activation factory helpers.
    namespace Activation {
        /// Create a linear activation.
        ActivationPtr Linear();
        /// Create a ReLU activation.
        ActivationPtr ReLU();
        /// Create a Leaky ReLU activation.
        ActivationPtr LeakyReLU(float alpha);
        /// Create a sigmoid activation.
        ActivationPtr Sigmoid();
        /// Create a logistic activation.
        ActivationPtr Logistic(float L, float K);
        /// Create a tanh activation.
        ActivationPtr Tanh();
    }

}

#endif
