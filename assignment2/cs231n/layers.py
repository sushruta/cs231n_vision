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
  x_shape = x.shape
  x_reshaped = x.reshape(x_shape[0], np.prod(x_shape[1:]))
  out = np.dot(x_reshaped, w) + b
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
  
  x_shape = x.shape
  x_mutated = x.reshape(x_shape[0], np.prod(x_shape[1:]))

  dx = np.dot(dout, w.T)
  dw = np.dot(x_mutated.T, dout)
  db = np.sum(dout, axis=0)

  dx = dx.reshape(x_shape)
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
  out = np.maximum(0.0, x)
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
  dx = dout
  dout[x <= 0] = 0.0
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
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape

  stride = conv_param['stride']
  pad = conv_param['pad']

  Hc = 1 + (H + 2 * pad - HH) / stride
  Wc = 1 + (H + 2 * pad - WW) / stride

  ## pad all the images
  xp = np.pad(x,
        ((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode='constant', constant_values=0)

  out = np.random.randn(N, F, Hc, Wc)

  hc, wc = (0, 0)
  for i in xrange(N):
    for j in xrange(F):
      for hc in xrange(Hc):
        for wc in xrange(Wc):
          xs = xp[i, :, hc*stride:hc*stride+HH, wc*stride:wc*stride+WW]
          out[i, j, hc, wc] = np.sum(xs * w[j,:,:,:]) + b[j]
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
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x, w, b, conv_param = cache
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  N, F, Hc, Wc = dout.shape
  stride = conv_param['stride']

  print(dout.shape)
  print(x.shape)
  print(w.shape)

  #dout = np.pad(dout, ((0,0),(0,0),(1,1),(1,1)), mode='constant', constant_values=0)
  xp = np.pad(x, ((0,0),(0,0),(1,1),(1,1)), mode='constant', constant_values=0)

  db = np.array([np.sum(dout[:,i,:,:]) for i in xrange(F)])
  dw = np.random.randn(F, C, HH, WW)
  for f in xrange(F):
    for c in xrange(C):
      for hh in xrange(HH):
        for ww in xrange(WW):
          dw[f, c, hh, ww] = np.sum(dout[:, f, :, :] * xp[:, c, hh:H+hh:stride, ww:W+ww:stride])

  dx = np.zeros(x.shape)
  dx = np.pad(dx, ((0,0), (0,0), (1,1), (1,1)), mode='constant', constant_values=0)
  for i in xrange(N):
    for hh in xrange(HH):
      for ww in xrange(WW):
        whw = w[:, :, hh, ww].T
        for hc in xrange(Hc):
          for wc in xrange(Wc):
            he = hc * stride + hh
            wi = wc * stride + ww
            dx[i, :, he, wi] += np.sum(whw * dout[i, :, hc, wc], axis=1)
  
  dx = dx[:, :, 1:-1, 1:-1]
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
  N, C, H, W = x.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']

  Hc = (H - pool_height) / stride + 1
  Wc = (W - pool_width) / stride + 1
  out = np.random.randn(N, C, Hc, Wc)
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  for i in xrange(N):
    for c in xrange(C):
      for hc in xrange(Hc):
        for wc in xrange(Wc):
          out[i, c, hc, wc] = np.max(x[i, c, hc:stride*hc+pool_height, wc:stride*wc+pool_width])
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
  x, pool_params = cache
  N, C, H, W = x.shape

  pool_height = pool_params['pool_height']
  pool_width = pool_params['pool_width']
  stride = pool_params['stride']

  Hc = (H - pool_height) / stride + 1
  Wc = (W - pool_width) / stride + 1

  dx = np.zeros(x.shape)
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  for i in xrange(N):
    for c in xrange(C):
      for hc in xrange(Hc):
        for wc in xrange(Wc):
          subx = x[i, c, hc:stride*hc+pool_height, wc:stride*wc+pool_width]
          subdx = dx[i, c, hc:stride*hc+pool_height, wc:stride*wc+pool_width]
          max_value = np.max(subx)
          
          subdx += (subx == max_value) * dout[i, c, hc, wc]
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

