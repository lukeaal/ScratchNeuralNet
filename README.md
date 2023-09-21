# ScratchNeuralNet
A from scratch multi-layer perceptron using only Numpy, trained on UCI Wine

# Multi-Layer Perceptron (MLP) Neural Network in Python

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.x-green.svg)

A simple implementation of a Multi-Layer Perceptron (MLP) neural network in Python using the NumPy library.

## Table of Contents

- [ScratchNeuralNet](#scratchneuralnet)
- [Multi-Layer Perceptron (MLP) Neural Network in Python](#multi-layer-perceptron-mlp-neural-network-in-python)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Convergence](#convergence)

## Introduction

This MLP neural network implementation is designed to provide a basic framework for building and training neural networks. It uses the Python programming language and the NumPy library for numerical computations. The code is organized into a Python class (`MLP`) that allows you to create, train, and evaluate MLP models.

    All algorithims for this model were taken from Frank Rosenblatt, the developer of the MLP back in 1958 when he published his paper about the "Perceptron".
    
<img href= "" alt="results from training">

## Features

- Implementation of a feedforward MLP neural network.
- Support for customizing the number of hidden nodes, learning rate, and input/target data.
- Sigmoid activation function for hidden layers and softmax for output layer.
- Training method for iteratively updating weights using forward and backward passes.
- Evaluation methods to calculate the sum of square errors and accuracy.
- Easy-to-use interface for training and testing neural networks.

## Installation

0. Clone or download the repository:

   ```bash
   git clone https://github.com/yourusername/mlp-neural-network.git
   ```
1. Navigate to project 
    ```bash
    cd mlp-neural-network
    ```
2. Install numpy 
    ```bash
    pip install numpy
    ```
    
## Usage 
You can use the MLP class in your Python projects to create, train, and evaluate MLP neural networks. Here's a simple example:

```python
# Import the MLP class
from mlp import MLP

# Define your input and target data
inputs = [...]  # Your input data
targets = [...]  # Your target data

# Create an MLP instance
mlp = MLP(inputs, targets, num_hidden_nodes=64, learning_rate=0.01)

# Train the MLP
mlp.train()

# Evaluate the model
sse = mlp.sum_of_square_errors()
accuracy = mlp.accuracy(val_inputs, val_targets)

print(f"Sum of Square Errors: {sse}")
print(f"Accuracy: {accuracy}")
```
## Convergence
Thanks to the beauty of mathematics, this stochastic process is still proven to 
converge every single time! A quantitative proof, while not symbolic, aligns with the 
previous statment and displays the convergence of my implementation on UCI Wine. Bellow 
is a graph of the sum of square errors (SSE) over the epoch the model is trained on. I 
hope this enstils a sense of aw like it does me!

<img src="https://github.com/lukeaal/ScratchNeuralNet/blob/main/media/SSE_Epoch.jpg" alt="sse over epoch" width="600" height="400">
