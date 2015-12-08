import numpy as np
import matplotlib.pyplot as plt

def init_two_layer_model(input_size, hidden_size, output_size):
  """
  Initialize the weights and biases for a two-layer fully connected neural
  network. The net has an input dimension of D, a hidden layer dimension of H,
  and performs classification over C classes. Weights are initialized to small
  random values and biases are initialized to zero.

  Inputs:
  - input_size: The dimension D of the input data
  - hidden_size: The number of neurons H in the hidden layer
  - ouput_size: The number of classes C

  Returns:
  A dictionary mapping parameter names to arrays of parameter values. It has
  the following keys:
  - W1: First layer weights; has shape (D, H)
  - b1: First layer biases; has shape (H,)
  - W2: Second layer weights; has shape (H, C)
  - b2: Second layer biases; has shape (C,)
  """
  # initialize a model
  model = {}
  #model['W1'] = 0.00001 * np.random.randn(input_size, hidden_size)
  model['W1'] = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / (input_size * hidden_size))
  model['b1'] = 0.0001 * np.ones(hidden_size)
  #model['W2'] = 0.00001 * np.random.randn(hidden_size, output_size)
  model['W2'] = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / (hidden_size * output_size))
  model['b2'] = 0.0001 * np.ones(output_size)
  return model

def two_layer_net(X, model, y=None, reg=0.0):
  """
  Compute the loss and gradients for a two layer fully connected neural network.
  The net has an input dimension of D, a hidden layer dimension of H, and
  performs classification over C classes. We use a softmax loss function and L2
  regularization the the weight matrices. The two layer net should use a ReLU
  nonlinearity after the first affine layer.

  The two layer net has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each
  class.

  Inputs:
  - X: Input data of shape (N, D). Each X[i] is a training sample.
  - model: Dictionary mapping parameter names to arrays of parameter values.
    It should contain the following:
    - W1: First layer weights; has shape (D, H)
    - b1: First layer biases; has shape (H,)
    - W2: Second layer weights; has shape (H, C)
    - b2: Second layer biases; has shape (C,)
  - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
    an integer in the range 0 <= y[i] < C. This parameter is optional; if it
    is not passed then we only return scores, and if it is passed then we
    instead return the loss and gradients.
  - reg: Regularization strength.

  Returns:
  If y not is passed, return a matrix scores of shape (N, C) where scores[i, c]
  is the score for class c on input X[i].

  If y is not passed, instead return a tuple of:
  - loss: Loss (data loss and regularization loss) for this batch of training
    samples.
  - grads: Dictionary mapping parameter names to gradients of those parameters
    with respect to the loss function. This should have the same keys as model.
  """

  # unpack variables from the model dictionary
  W1,b1,W2,b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, D = X.shape
  ## p = 0.95

  # compute the forward pass
  scores = None
  H1 = np.maximum(0.0, np.dot(X, W1) + b1)

  H2 = np.dot(H1, W2) + b2

  scores = H2

  if y is None:
    return scores

  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
  log_probs = -np.log(probs[range(N), y])
  
  loss = np.sum(log_probs)/N
  reg_loss = 0.5 * reg * (np.sum(W1*W1) + np.sum(W2*W2))
  loss += reg_loss

  # compute the gradients
  grads = {}
  dscores = probs
  dscores[range(N), y] -= 1
  dscores /= N

  dW2 = np.dot(H1.T, dscores) + reg * W2
  db2 = np.sum(dscores, axis=0)
  dH1 = np.dot(dscores, W2.T)
  dH1[H1 <= 0] = 0
  dW1 = np.dot(X.T, dH1) + reg * W1
  db1 = np.sum(dH1, axis=0)

  #print("shape of b2 :: {0} db2 :: {1} W1 :: {2} dW1 :: {3}".format(b2.shape, db2.shape, W1.shape, dW1.shape))

  grads['W1'] = dW1
  grads['W2'] = dW2
  grads['b1'] = db1
  grads['b2'] = db2

  return loss, grads
