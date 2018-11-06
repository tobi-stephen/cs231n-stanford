import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)	#for numerical stability
    correct_class_score = scores[y[i]]
    grad_scores = np.sum(np.exp(scores))
    top = np.exp(correct_class_score)
    bottom = 0
    for j in range(num_classes):
      bottom += np.exp(scores[j])
      if j == y[i]:
        dW[:, j] += -X[i] + ((1/grad_scores) * (X[i] * np.exp(scores[j])))
      else:
        dW[:, j] += (1/grad_scores) * (X[i] * np.exp(scores[j]))
    prob = top / bottom
    loss += (-1 * np.log(prob))

  loss /= num_train
  dW /= num_train
  

  #regularization
  loss += 0.5 * reg * np.sum(W*W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores -= np.max(scores, axis=1).reshape(-1, 1) #numerical stability
  correct_scores = scores[range(num_train), list(y)].reshape(-1, 1)

  top = np.exp(correct_scores)
  bottom = np.sum(np.exp(scores), axis=1).reshape(-1,1)
  prob = top/bottom
  loss = -np.sum(np.log(prob)) / num_train

  temp = np.exp(scores)/bottom
  temp[range(num_train), list(y)] += -1

  dW = X.T.dot(temp)
  dW /= num_train

  #regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

