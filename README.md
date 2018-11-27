# DeepLearning Projects
DeepLearning projects for the deep learning course at EPFL

## Mini-project 1: Prediction of finger movements from EEG recordings.

The goal of this project is to implement a neural network to predict the laterality of finger movement
(left or right) from the EEG recording.
Several neural networks architectures have been tested

- Multilayer Perceptron : 26 %
- Convolutional Neural Network : 
- Recurent Neural Network (basic version) : 23 % error
- Long Short Term Memory : 17% error

## Mini-project 2: Implementing from scratch a mini deep-learning framework.

This framework import only torch.FloatTensor and torch.LongTensor from pytorch, and
use no pre-existing neural-network python toolbox.

It provides the necessary tools to:
- build networks combining fully connected layers, Tanh, and ReLU,
- run the forward and backward passes,
- optimize parameters with SGD for MSE.
