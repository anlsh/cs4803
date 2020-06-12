from __future__ import division

import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
    We multiply this against a weight matrix of shape (D, M) where
    D = \prod_i d_i

    Inputs:
    x - Input data, of shape (N, d_1, ..., d_k)
    w - Weights, of shape (D, M)
    b - Biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    N = x.shape[0]
    flat_length = np.prod(x.shape[1:])
    #############################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You     #
    # will need to reshape the input into rows.                                 #
    #############################################################################
    out = np.dot(x.reshape(N, flat_length), w) + b
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the affine backward pass.                                 #
    #############################################################################
    N = x.shape[0]
    flat_length = np.prod(x.shape[1:])
    flat_x = x.reshape(N, flat_length)

    gradloss_w = np.dot(flat_x.T, dout)
    gradloss_x = np.dot(dout, w.T).reshape(x.shape)
    gradloss_b = np.sum(dout, axis=0)

    dw = gradloss_w
    dx = gradloss_x
    db = gradloss_b

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    #############################################################################
    # TODO: Implement the ReLU forward pass.                                    #
    #############################################################################
    y = x.copy()
    y[y < 0] = 0
    out = y
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    #############################################################################
    # TODO: Implement the ReLU backward pass.                                   #
    #############################################################################
    dx = dout.copy()
    dx[x < 0] = 0

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the convolutional forward pass.                           #
    # Hint: you can use the function np.pad for padding.                        #
    #############################################################################
    stride = conv_param['stride']
    pad_size = int(conv_param['pad'])

    H_prime = int(1 + (x.shape[2] + 2 * pad_size - w.shape[2]) / stride)
    W_prime = int(1 + (x.shape[3] + 2 * pad_size - w.shape[3]) / stride)
    out = np.zeros((x.shape[0], w.shape[0], H_prime, W_prime))

    x_padded = np.pad(x, ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)),
                      mode='constant')

    for n in range(x.shape[0]):
        for f in range(w.shape[0]):
            for h_p in range(H_prime):
                for w_p in range(W_prime):
                    x_window = x_padded[n][range(w.shape[1]),
                                           stride*h_p: (stride)*h_p + w.shape[2],
                                           stride*w_p: (stride)*w_p + w.shape[3]]

                    out[n, f, h_p, w_p] = np.sum(np.multiply(x_window, w[f])) + b[f]
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    x, w, b, conv_param = cache
    stride = conv_param['stride']
    pad_size = int(conv_param['pad'])

    H_prime = int(1 + (x.shape[2] + 2 * pad_size - w.shape[2]) / stride)
    W_prime = int(1 + (x.shape[3] + 2 * pad_size - w.shape[3]) / stride)

    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################

    """
    Input:
    - (d)x: Input data of shape (N, C, H, W)
    - (d)w: Filter weights of shape (F, C, HH, WW)
    - (d)b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    - dout: Output derivatives of shape (N, F, H', W')
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    """
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)),
                       mode='constant')
    dx_padded = np.zeros(x_padded.shape)
    db = np.zeros(b.shape)

    ##########################################
    # Calculate the gradient wrt everything! #
    ##########################################

    dw = np.zeros(w.shape)

    # I think this should probably work...
    for n in range(dout.shape[0]):
        for f in range(w.shape[0]):
            db[f] += np.sum(dout[n, f])
            for h_p in range(H_prime):
                for w_p in range(W_prime):
                    dx_padded[n, range(w.shape[1]),
                              stride*h_p: (stride)*h_p + w.shape[2],
                              stride*w_p: (stride)*w_p + w.shape[3]] += dout[n, f, h_p, w_p] * w[f]

                    dw[f] += dout[n, f, h_p, w_p] * x_padded[n, range(w.shape[1]),
                                                             stride*h_p: (stride)*h_p + w.shape[2],
                                                             stride*w_p: (stride)*w_p + w.shape[3]]

    # Should un-pad the input, which is needed!
    dx = dx_padded[:, :,
                   pad_size:x_padded.shape[2] - pad_size,
                   pad_size:x_padded.shape[3] - pad_size]

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the max pooling forward pass                              #
    #############################################################################
    stride = pool_param['stride']
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']

    H_prime = int(1 + (x.shape[2] - pool_height) / stride)
    W_prime = int(1 + (x.shape[3] - pool_width) / stride)
    out = np.zeros((x.shape[0], x.shape[1], H_prime, W_prime))

    for n in range(x.shape[0]):
        for c in range(x.shape[1]):
            for h_p in range(H_prime):
                for w_p in range(W_prime):
                    x_window = x[n][c,
                                    stride*h_p: (stride)*h_p + pool_height,
                                    stride*w_p: (stride)*w_p + pool_width]

                    out[n, c, h_p, w_p] = np.amax(x_window)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    #############################################################################
    # TODO: Implement the max pooling backward pass                             #
    #############################################################################

    x = cache[0]
    dx = np.zeros(x.shape)

    pool_param = cache[1]
    stride = pool_param['stride']
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']

    for n in range(dx.shape[0]):
        for c in range(dout.shape[1]):
            for h in range(dout.shape[2]):
                for w in range(dout.shape[3]):
                    x_window = x[n][c,
                                    stride*h: (stride)*h + pool_height,
                                    stride*w: (stride)*w + pool_width]
                    max_pos = np.unravel_index(x_window.argmax(), x_window.shape)
                    orig_h = stride*h + max_pos[0]
                    orig_w = stride*w + max_pos[1]

                    dx[n, c, orig_h, orig_w] = dout[n, c, h, w]
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
