cmake_minimum_required(VERSION 3.16)
project(bbdnn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(bbdnn
  src/Activations.cpp
  src/DenseLayer.cpp
  src/LayerConnection.cpp
  src/Matrix.cpp
  src/NeuralNetwork.cpp
)

target_include_directories(bbdnn PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ---- Example ----
add_executable(nn_demo
  examples/nn_demo.cpp
)

target_link_libraries(nn_demo PRIVATE bbdnn)