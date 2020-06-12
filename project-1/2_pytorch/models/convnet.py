from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(CNN, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        ############################################################################$
        self.conv_in_channels = im_size[0]
        self.conv_num_filters = hidden_dim
        self.conv_kernel_size = kernel_size
        self.conv_padding = int((self.conv_kernel_size - 1) / 2)

        self.pool_width = 2
        self.pool_height = self.pool_width
        self.pool_stride = self.pool_width

        self.image_size_after_pool = int(im_size[1] / self.pool_stride)
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim

        self.conv_layer = nn.Conv2d(in_channels=self.conv_in_channels,
                                    out_channels=self.conv_num_filters,
                                    kernel_size=self.conv_kernel_size,
                                    padding=self.conv_padding)
        self.conv_relu_pool = nn.Sequential(
            self.conv_layer,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.pool_width, stride=self.pool_stride),
        )
        self.fully_connected = nn.Sequential(
            nn.Linear((self.image_size_after_pool**2) * self.conv_num_filters, self.n_classes),
        )
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the CNN to
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
        # TODO: Implement the forward pass. This should take few lines of code.
        #############################################################################
        # scores = self.model(images)
        tmp = self.conv_relu_pool(images)
        scores = self.fully_connected(tmp.reshape(tmp.size(0), -1))
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores
