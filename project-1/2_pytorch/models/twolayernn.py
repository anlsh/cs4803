import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TwoLayerNN(nn.Module):
    def __init__(self, im_size, hidden_dim, n_classes):
        '''
        Create components of a two layer neural net classifier (often
        referred to as an MLP) and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            n_classes (int): Number of classes to score
        '''
        super(TwoLayerNN, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        self.hidden_dim = hidden_dim
        self.datum_size = im_size[0] * im_size[1] * im_size[2]
        self.layer1_linear = nn.Linear(self.datum_size, hidden_dim)
        self.layer1_relu = nn.ReLU()
        self.layer2_linear = nn.Linear(self.hidden_dim, n_classes)
        nn.init.xavier_uniform(self.layer1_linear.weight)
        nn.init.xavier_uniform(self.layer2_linear.weight)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the NN to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = None
        #############################################################################
        # TODO: Implement the forward pass. This should take very few lines of code.
        #############################################################################
        flat_images = torch.reshape(images, (images.shape[0], self.datum_size))
        scores = self.layer2_linear(self.layer1_relu(self.layer1_linear(flat_images)))
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores
