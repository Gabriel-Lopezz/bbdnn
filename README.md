# bbdnn

A small, from-scratch C++ neural network library for demonstrative use. The goal is to keep the API easy to use while staying flexible enough for experimentation and learning. It focuses on simple feed-forward dense networks that you can assemble by hand, train with backpropagation, and inspect at each step.

## Purpose

- Demonstrate how feed-forward neural networks work without heavy dependencies.
- Provide a compact, approachable API for building and training small networks.
- Stay flexible so you can customize activations, layer sizes, and training loops.

## Capabilities

- Dense (fully connected) layers with configurable activations.
- Common activations: Linear, ReLU, LeakyReLU, Sigmoid, Logistic, Tanh.
- Xavier and Kaiming weight initialization in `LayerConnection`.
- Forward propagation and backpropagation for gradient-based learning.
- Lightweight `Matrix` and `Vector` types for basic linear algebra.
- Optional helper utilities on `NeuralNetwork` such as `train`, `evaluate`, and `predict`.

## Quick Start

Use the umbrella header:

```cpp
#include "bbdnn/bbdnn.hpp"
```

Or include individual headers such as:

```cpp
#include "bbdnn/Activations.hpp"
```

All library types live in the `bbdnn` namespace.

## Demo: `examples/nn_demo.cpp`

The demo trains a tiny network on XOR and prints predictions:

1. Build a 2-3-1 network: input layer (2), hidden layer (3), output layer (1).
2. Use `Linear`, `Tanh`, and `Sigmoid` activations.
3. Train on XOR samples with `train(features, labels, learningRate, epochs, false)`.
4. Use `predict` to get outputs for each input.

Key snippet from the demo:

```cpp
NeuralNetwork nn(42, {
    DenseLayer(2, Activation::Linear()),
    DenseLayer(3, Activation::Tanh()),
    DenseLayer(1, Activation::Sigmoid())
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

nn.train(features, labels, 0.05f, 10000, false);
```

## Build

This project uses CMake:

```sh
cmake -S . -B build
cmake --build build
```

The `nn_demo` executable will be built from `examples/nn_demo.cpp`.

## Build with Makefile

There is also a simple Makefile in the project root:

```sh
make
./nn_demo
```

To clean build artifacts:

```sh
make clean
```
