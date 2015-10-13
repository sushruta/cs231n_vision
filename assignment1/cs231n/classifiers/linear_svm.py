import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    dW_i = np.zeros(W.shape)
    dW_i_yi = np.zeros(W[0].shape)
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW_i[j] = X[:, i]
        dW_i_yi += dW_i[j]
    dW_i[y[i]] = -dW_i_yi
    dW += dW_i

  loss /= num_train
  dW /= num_train

  loss += 0.5 * reg * np.sum(W * W)

  dW += reg * W
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  return loss, dW
