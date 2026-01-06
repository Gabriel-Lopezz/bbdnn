#include "bbdnn/Activations.hpp"

namespace bbdnn {

    float LinearActivation::operator()(float x) const {
        return x;
    }

    float LinearActivation::derive(float x) const {
        (void)x;
        return 1;
    }

    ActivationPtr LinearActivation::clone() const {
        return std::make_unique<LinearActivation>(*this);
    }

    float ReLUActivation::operator()(float x) const {
        return x > 0 ? x : 0;
    }

    float ReLUActivation::derive(float x) const {
        return x > 0 ? 1 : 0;
    }

    ActivationPtr ReLUActivation::clone() const {
        return std::make_unique<ReLUActivation>(*this);
    }

    LeakyReLUActivation::LeakyReLUActivation(float a) : alpha(a) {}

    float LeakyReLUActivation::operator()(float x) const {
        return x > 0 ? x : alpha * x;
    }

    float LeakyReLUActivation::derive(float x) const {
        return x > 0 ? 1 : alpha;
    }

    ActivationPtr LeakyReLUActivation::clone() const {
        return std::make_unique<LeakyReLUActivation>(*this);
    }

    float SigmoidActivation::operator()(float x) const {
        return 1.0 / (1.0 + std::exp(-x));
    }

    float SigmoidActivation::derive(float x) const {
        float sig = (*this)(x);
        return sig * (1 - sig);
    }

    ActivationPtr SigmoidActivation::clone() const {
        return std::make_unique<SigmoidActivation>(*this);
    }

    LogisticActivation::LogisticActivation(float L, float K) : l(L), k(K) {}

    float LogisticActivation::operator()(float x) const {
        float denom = 1 + std::exp(-k * x);
        return l / denom;
    }

    float LogisticActivation::derive(float x) const {
        float logVal = (*this)(x);
        return logVal * (1 - logVal);
    }

    ActivationPtr LogisticActivation::clone() const {
        return std::make_unique<LogisticActivation>(*this);
    }

    float TanhActivation::operator()(float x) const {
        return std::tanh(x);
    }

    float TanhActivation::derive(float x) const {
        float t = (*this)(x);
        return 1 - t * t;
    }

    ActivationPtr TanhActivation::clone() const {
        return std::make_unique<TanhActivation>(*this);
    }

    namespace Activation {
        ActivationPtr Linear() { return make_activation<LinearActivation>(); }

        ActivationPtr ReLU() { return make_activation<ReLUActivation>(); }

        ActivationPtr LeakyReLU(float alpha) { return make_activation<LeakyReLUActivation>(alpha); }

        ActivationPtr Sigmoid() { return make_activation<SigmoidActivation>(); }
    
        ActivationPtr Logistic(float L, float K) { return make_activation<LogisticActivation>(L, K); }
    
        ActivationPtr Tanh() { return make_activation<TanhActivation>(); }
    }

}
