# ScratchNeuralNet
A from scratch multi-layer perceptron using only Numpy, trained on UCI Wine

# Multi-Layer Perceptron (MLP) Neural Network in Python

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.x-green.svg)

A simple implementation of a Multi-Layer Perceptron (MLP) neural network in Python using the NumPy library.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Documentation](#documentation)
6. [License](#license)

## Introduction

This MLP neural network implementation is designed to provide a basic framework for building and training neural networks. It uses the Python programming language and the NumPy library for numerical computations. The code is organized into a Python class (`MLP`) that allows you to create, train, and evaluate MLP models.

    > all algorithims for this model were taken from Frank Rosenblatt, the origonal developer of the MLP back in 1958 when he published his paper and the "Perceptron"

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
