""" Activation functions for plotting in network nodes.
    (the blue curves)
"""
import numpy as np


def relu(x, y, radius):
    if x <= 0:
        return 0 - radius / 4
    return x - radius / 4


def sigmoid(x, y, radius):
    return radius / (1 + np.exp(-radius * 100 * x)) - radius / 2


def linear(x, y, radius):
    return x


def make_node_data(x, y, radius, activation):
    """ Create data for a network node's activation
        function using the node center, it's radius 
        and an activation function.

        :param x: x-coord of node center.
        :type x: float
        
        :param y: y-coord of node center.
        :type y: float

        :param radius: radius of node.
        :type radius: node

        :param activation: activation function.
    """
    X = np.linspace(x - radius, x + radius)
    Y = [y + activation(xi - x, y, radius) for xi in X]
    return X, Y


def dispacth_activation(s):
    """ Get a callable corresponding to the
        activation string. In essence this translates
        from a string, e.g. 'sigmoid', to the sigmoid
        function. If the input is already callable it 
        is returned as is.
    """
    if callable(s):
        return s
    elif s == "relu":
        return relu
    elif s == "sigmoid":
        return sigmoid
    elif s == "linear":
        return linear
    return None
