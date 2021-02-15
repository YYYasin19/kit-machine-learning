#!/usr/bin/env python
# coding: utf-8

# # Exercise 6

# ## 1) Neural Network Classifier from Scratch (10p.)
# 
# In this exercise we will implement a small neural network from scratch, i.e., only using numpy. This is nothing you would do "in real life" but it is a good exercise to deepen understanding. 
# 
# The network will consist of an arbitrary number of hidden layers with ReLU activation, a sigmoid output layer (as we are doing binary classification) and we will train it using the binary cross entropy (negative bernoulli likelihood). Ok, so lets start by importing and loading what we need. 

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Load our two moons (I promise we will get a new dataset in the next exercise)
train_data = dict(np.load("two_moons.npz", allow_pickle=True)) 
test_data = dict(np.load("two_moons_test.npz", allow_pickle=True))
# we need to reshape our labels so that they are [N, 1] and not [N] anymore
train_samples, train_labels = train_data["samples"], train_data["labels"][:, None]
test_samples, test_labels = test_data["samples"], test_data["labels"][:, None]


# 
# ### 1.1.) Auxillary Functions (3 p.)
# We start with implementing some auxillary functions we are going to need later. The sigmoid and relu activation functions, the binary cross entropy loss as well as their derviatives. 
# 
# The binary cross entropy loss is given as 
# $ - \dfrac{1}{N} \sum_{i=1}^N (y_i \log (p_i) + (1 - y_i) \log (1 - p_i)) $ where $y_i$ denotes the ground truth label and $p_i$ the network prediction for sample $i$.
# 
# **Hint** all derivatives where derived/implemented during the lecture or previous exercise - so feel free to borrow them from there. 

# In[3]:


def relu(x: np.ndarray) -> np.ndarray:
    """
    elementwise relu activation function
    :param x: input to function [shape: arbitrary]
    :return : relu(x) [shape: same as x]
    """
    ### DONE #########################
    return np.max(0,x)
    ##################################


def d_relu(x: np.ndarray) -> np.ndarray:
    """
    elementwise gradient of relu activation function
    :param x: input to function [shape: arbitrary]
    :return : d relu(x) / dx [shape: same as x]
    """
    ### DONE #########################
    x[x<=0] = 0
    x[x>0] = 1
    return x
    ##################################


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    elementwise sigmoid activation function
    :param x: input to function [shape: arbitrary]
    :return : d sigmoid(x) /dx [shape: same as x]
    """
    ### DONE #########################
    return (1/(1+np.exp(-x)))
    ##################################


def d_sigmoid(x: np.ndarray) -> np.ndarray:
    """
    elementwise sigmoid activation function
    :param x: input to function [shape: arbitrary]
    :return : sigmoid(x) [shape: same as x]
    """
    # --- Is this correct?
    # --- Yes, seems so (https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e)
    ### DONE #########################
    return sigmoid(x) * (1-sigmoid(x))
    ##################################


def binary_cross_entropy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    binary cross entropy loss (negative bernoulli ll)
    :param predictions: predictions by model (shape [N])
    :param labels: class labels corresponding to train samples, (shape: [N])
    :return binary cross entropy
    """
    ### DONE #########################
    N = labels.shape[0]
    loss = np.zeros((N,1))
    loss[labels == 1] = - np.log(predictions[labels == 1])
    loss[labels == 0] = - np.log(1-predictions[labels == 0])
    
    # optional: divide by number of predictions/ labels
    return 1/N * np.sum(loss)
    ##################################


def d_binary_cross_entropy(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    gradient of the binary cross entropy loss
    :param predictions: predictions by model (shape [N])
    :param labels: class labels corresponding to train samples, (shape [N])
    :return gradient of binary cross entropy, w.r.t. the predictions (shape [N])
    """
    ### DONE #########################
    N = predictions.shape[0]
    gradient = np.zeros(N)
    gradient = (labels / predictions - ((1-labels) / (1-predictions)))
    
    # optional: divide by number of predictions/ labels
    return -1/N * gradient
    ##################################


# ## General Setup & Intialization
# 
# Next we are going to set up the Neural Network. We will represent it as a list of weight matrices and a list of bias vectors. Each list has one entry for each layer.
# 

# In[4]:


def init_weights(neurons_per_hidden_layer: List[int], input_dim: int, output_dim: int, seed: int = 0)         -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    :param neurons_per_hidden_layer: list of numbers, indicating the number of neurons of each hidden layer
    :param input_dim: input dimension of the network
    :param output_dim: output dimension of the network
    :param seed: seed for random number generator
    :return list of weights and biases as specified by dimensions and hidden layer specification
    """
    # seed random number generator
    rng = np.random.RandomState(seed)
    scale_factor = 1.0
    prev_n = input_dim
    weights = []
    biases = []

    # hidden layers
    for n in neurons_per_hidden_layer:
        # initialize weights with gaussian noise
        weights.append(scale_factor * rng.normal(size=[prev_n, n]))
        # initialize bias with zeros
        biases.append(np.zeros([1, n]))
        prev_n = n

    # output layer
    weights.append(scale_factor * rng.normal(size=[prev_n, output_dim]))
    biases.append(np.zeros([1, output_dim]))

    return weights, biases


# **NOTE** As NNs are non-convex, initialization plays a very important role in NN training and there is a lot of work into how to initialize them properly - this here is not a very good initialization, but sufficient for our small example.

# ## 1.2) Forward Pass (3 p.)
# 
# Next step is the forward pass, i.e., propagate a batch of samples through the network to get the final prediciton.
# But that's not all - to compute the gradietns later we also need to store all necessary quantities, here those are:
# - The input to every layer (here called h's)
# - The "pre-activation" of every layer, i.e., the qantity that is fed into the non-linearity (here called z's)
# 

# In[5]:


def forward_pass(x: np.ndarray, weights: List[np.ndarray], biases: List[np.ndarray])        -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    propagate input through network
    :param x: input: (shape, [N x input_dim])
    :param weights: weight parameters of the layers
    :param biases: bias parameters of the layers
    :return: - Predictions of the network (shape, [N x out_put_dim])
             - hs: output of each layer (input + all hidden layers) (length: len(weights))
             - zs: preactivation of each layer (all hidden layers + output) (length: len(weights))
    """

    hs = []  # list to store all inputs
    zs = []  # list to store all pre-activations
    
    # input to first hidden layer is just the input to the network 
    h = x
    hs.append(h)
  
    ### DONE #########################
    # pass "h" to all hidden layers
    # record all inputs and pre-activations in the lists  
    for layer, w in enumerate(weights):
        b = biases[layer]
        
        # weight shape: [2, 64], [64,64], [64, 1]
        # data shape:   [100,2], [64,64], [64,64]
        # bias shape:   [1, 64], [1, 64], [1, 1]
        print("Layer", layer, "Weights", w.shape, "Data", h.shape)
        s = np.sum(w.T @ h.T,axis=0)
        z = (s + b.T).T
        print(s.shape, z.shape, b.shape)
        zs.append(z)

        h = sigmoid(z)
        hs.append(h)        
        
    ##################################
    # has to have same shape as labels, i.e. [N,1]
    y = sigmoid(z)   # z denotes the pre-activation of the output layer here. Feel free to rename it

    return y, hs[:-1], zs


# ## 1.3) Backward Pass (4 p.)
# 
# For training by gradient descent we need - well - gradients. Those are computed using backpropagation during the so called "backward pass". We will use the chain rule to propagate the gradient back through the network and at every layer, compute the gradients for the weights and biases at that layer. The initial gradient is given by the gradient of the loss function w.r.t. the network output. 

# In[25]:


def backward_pass(loss_grad: np.ndarray, 
                  hs: List[np.ndarray], zs: List[np.ndarray], 
                  weights: List[np.ndarray], biases: List[np.ndarray]) -> \
    Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    propagate gradient backwards through network
    :param loss_grad: gradient of the loss function w.r.t. the network output (shape: [N, 1])
    :param hs: values of all hidden layers during forward pass
    :param zs: values of all preactivations during forward pass
    :param weights: weight paramameters of the layers
    :param biases: bias parameters of the layers
    :return: d_weights: List of weight gradients - one entry with same shape for each entry of "weights"
             d_biases: List of bias gradients - one entry with same shape for each entry of "biases"
    """

    # return gradients as lists - we pre-initialize the lists as we iterate backwards
    d_weights = [None] * len(weights)
    d_biases = [None] * len(biases)

    ### TODO #########################
    layers = len(hs)
    
    # backwards trough network -- [2, 1, 0] for 3 layers
    
    hs_grad = [None] * (len(weights)+1)
    print(f'Weights: {len(weights)} with shapes: {[w.shape for w in weights]}')
    hs_grad[len(weights)] = np.sum(loss_grad).reshape(-1,1)
    zs_grad = [None] * len(weights)


    for layer in range(len(weights)-1, -1, -1):
        
        zs_grad[layer] = hs_grad[layer+1] * d_sigmoid(zs[layer])
        
        d_weights[layer] = np.outer(zs_grad[layer],hs[layer])
        d_biases[layer] = zs_grad[layer]

        # update hs_grad for next round -- like giving the loss to the subnetwork
        hs_grad[layer] = weights[layer] @ zs_grad[layer]
        
    # print("Weights", len(weights), len(weights[0]), len(weights[1]), len(weights[2]), 
    #      "Biases", len(biases), len(biases[0]), len(biases[1]), len(biases[2]))
        
    #print("Weight Grad.", len(d_weights), len(d_weights[0]), len(d_weights[1]), len(d_weights[2]), 
    #      "Bias Grad.", len(d_biases), len(d_biases[0]), len(d_biases[1]), len(d_biases[2]))
        
    ##################################

    return d_weights, d_biases


# ## Tying Everything Together 
# 
# Finally we can tie everything together and train our network. 

# In[26]:


N = train_samples.shape[0]

# hyper parameters 
layers = [64, 64]
learning_rate = 1e-2

# init model
weights, biases = init_weights(layers, input_dim=2, output_dim=1, seed=42)


#book keeping
train_losses = []
test_losses = []

# Here we work with a simple gradient descent implementation, using the whole dataset at each iteration,
# You can modify it to stochastic gradient descent or a batch gradient descent procedure as an exercise
for i in range(1000):
    
    # predict network outputs and record intermediate quantities using the forward pass
    prediction, hs, zs = forward_pass(train_samples, weights, biases)
    # print("Labels:", train_labels.shape, "vs. predictions", prediction.shape)
    train_losses.append(binary_cross_entropy(prediction, train_labels))

    # compute gradients
    loss_grad = d_binary_cross_entropy(prediction, train_labels)
    w_grads, b_grads = backward_pass(loss_grad, hs, zs, weights, biases)

    # apply gradients
    for i in range(len(w_grads)):
        weights[i] -= learning_rate * w_grads[i]
        biases[i] -= learning_rate * b_grads[i]

    test_losses.append(binary_cross_entropy(forward_pass(test_samples, weights, biases)[0], test_labels))

# plotting
plt.title("Loss")
plt.semilogy(train_losses)
plt.semilogy(test_losses)
plt.legend(["Train Loss", "Test Loss"])
