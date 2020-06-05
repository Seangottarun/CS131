"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    delta_y = Hk // 2
    delta_x = Wk // 2

    vert_edge_padding = delta_y
    horiz_edge_padding = delta_x

    for m in range(Hi):
        for n in range(Wi):
            sum = 0
            for i in range(-delta_y, delta_y + 1):
                for j in range(-delta_x, delta_x + 1):
                    if m - i < 0 or n - j < 0 or m - i >= Hi or n - j >= Wi:
                        sum += 0
                    else:
                        sum += image[m - i, n - j] * kernel[delta_y + i, delta_x + j]
            out[m, n] = sum
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.zeros((H + 2 * pad_height, W + 2 * pad_width), dtype=image.dtype)
    out[pad_height:H + pad_height, pad_width: W + pad_width] = image
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    delta_y = Hk // 2
    delta_x = Wk // 2

    # flip kernel along both axes
    filter = np.copy(kernel)
    filter = np.flip(np.flip(filter, 0), 1)

    padded_img = zero_pad(image, delta_y, delta_x)

    for m in range(Hi):
        for n in range(Wi):
            out[m, n] = np.sum(padded_img[m : m + Hk, n : n + Wk] * filter)
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    delta_y = int((Hk - 1) / 2)
    delta_x = int((Wk - 1) / 2)

    # flip kernel along both axes
    filter = np.copy(kernel)
    np.flip(filter, 0)
    np.flip(filter, 1)

    padded_img = zero_pad(image, delta_y, delta_x)

    # use Fast Fourier Transforms
    out = np.real(np.fft.ifft2(np.fft.fft2(image)*np.fft.fft2(filter, s=image.shape)))
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g_new = np.flip(np.flip(g, 0), 1)
    out = conv_fast(f, g_new)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g_avg = np.mean(g)
    g_new = g - g_avg
    g_new = np.flip(np.flip(g_new, 0),1)
    out = conv_fast(f, g_new)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    out = np.zeros(f.shape, dtype=f.dtype)
    Hi, Wi = f.shape
    # force odd dimensions for template
    if g.shape[0] % 2 == 0:
        g = g[0:-1]
    if g.shape[1] % 2 == 0:
        g = g[:,0:-1]

    Ht, Wt = g.shape
    Hp, Wp = g.shape

    delta_y = Hp // 2
    delta_x = Wp // 2

    g_avg = np.mean(g)
    g_std = np.std(g)

    g_norm = (g - g_avg) / g_std

    for m in range(delta_y, Hi - delta_y):
        for n in range(delta_x, Wi - delta_x):
            f_patch = f[m - delta_y: m + delta_y + 1, n - delta_x: n + delta_x + 1]
            f_avg = np.mean(f_patch)
            f_std = np.std(f_patch)
            f_norm = (f_patch - f_avg) / f_std

            out[m, n] =np.sum( f_norm * g_norm )
    ### END YOUR CODE

    return out
