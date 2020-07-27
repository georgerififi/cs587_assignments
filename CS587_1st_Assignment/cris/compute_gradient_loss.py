import numpy as np
from random import shuffle

def compute_gradient_and_loss1(W, X, y, reg, reg_type, opt):
  """
  loss and gradient function.

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
  - reg_type: (int) regularization type (1: l1, 2: l2)
  - opt: (int) 0 for computing both loss and gradient, 1 for computing loss only
  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  if opt == 0:                          # compute gradient only if opt == 0
      dW = np.zeros(W.shape)            # initialize the gradient as zero
  
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  #############################################################################
  # TODO:                                                                     #
  # Implement the routine to compute the loss, storing the result in loss     #
  #############################################################################  
  for i in xrange(num_train):           # for every augmended image data (3072+1 vector)
      s = X[i].dot(W)                   # compute s (scores)
      s_y = s[y[i]]                     # keep the correct ground truth class score
      for j in xrange(num_classes):     # for every class
          if j != y[i]:                 # don't take the correct ground truth index
              term = s[j] - s_y + 1     # max term with Delta = 1, according to Hinge loss formula
              if term > 0:              # trick: take only the term > 0, equal to max(0,...) formula
                  loss += term          # add the possitive term 
                  if opt == 0:          # compute gradient only if opt == 0
                      dW[:, j] += X[i]  # this is a analytically with Calculus gradient, case j<>y[i]
                      dW[:, y[i]] -= X[i] #  case j==y[i]

#  loss /= num_train                    # num_train = M, according to given formula                 

  if reg_type == 1:                     # loss + regularization , l2 or l1
      loss += reg * np.sum(np.abs(W))   # l1, reg is actually lambda regularization strength
  else:
      loss += reg * np.sum(W * W)       # l2
      
  if opt == 0:                          # compute gradient only if opt == 0
      dW /= num_train                   # we have to divide by num_train in order to have the 'mean' gradient
      if reg_type == 1:                 # we use deriv_abs function for l1 derivative
          dW += reg * deriv_abs(W)
      else:
          dW += 2 * reg * W             # l2 derivative formula
          
      return loss, dW
  else:
      return loss, None
  
  print 'CSFAK INSIDE compute_gradient_and_loss'
  #############################################################################
  # TODO:                                                                     #
  # Implement the gradient for the required loss, storing the result in dW.	  #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  #pass

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
def compute_gradient_and_loss2(W, X, y, reg, reg_type, opt):
  """
  loss and gradient function.

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
  - reg_type: (int) regularization type (1: l1, 2: l2)
  - opt: (int) 0 for computing both loss and gradient, 1 for computing loss only
  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  if opt == 0:                          # compute gradient only if opt == 0
      dW = np.zeros(W.shape)            # initialize the gradient as zero
  
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  #############################################################################
  # TODO:                                                                     #
  # Implement the routine to compute the loss, storing the result in loss     #
  #############################################################################  
  for i in xrange(num_train):           # for every augmended image data (3072+1 vector)
      s = X[i].dot(W)                   # compute s (scores)
      s_y = s[y[i]]                     # keep the correct ground truth class score
      max_sj = -999
      argmax_sj = -1
      local_loss = 0.0
      for j in xrange(num_classes):     # for every class        
          if j == y[i]:                 # don't take the correct ground truth index
              continue
          if s[j] > max_sj:
              max_sj = s[j]
              argmax_sj = j

      term = 1 + max_sj - s_y     # max term with Delta = 1, according to Hinge loss formula            

      for j in xrange(num_classes):     # for every class        
          if j == y[i]:                 # don't take the correct ground truth index
              continue
          if term > 0:              # trick: take only the term > 0, equal to max(0,...) formula
              local_loss = term          # add the possitive term 
              if opt == 0:          # compute gradient only if opt == 0
                  if j == argmax_sj:
                      dW[:, j] += X[i]  # this is a analytically with Calculus gradient, case j<>y[i]
                      dW[:, y[i]] -= X[i] #  case j==y[i]
      
      loss += local_loss             

#  loss /= num_train                    # num_train = M, according to given formula                 

  if reg_type == 1:                     # loss + regularization , l2 or l1
      loss += reg * np.sum(np.abs(W))   # l1, reg is actually lambda regularization strength
  else:
      loss += reg * np.sum(W * W)       # l2
      
  if opt == 0:                          # compute gradient only if opt == 0
      dW /= num_train                   # we have to divide by num_train in order to have the 'mean' gradient
      if reg_type == 1:                 # we use deriv_abs function for l1 derivative
          dW[:,-1] += reg * deriv_abs(W[:,-1]) #dW[:,-1]
      else:
          dW[:,-1] += 2 * reg * W[:,-1]             # l2 derivative formula
          
      return loss, dW
  else:
      return loss, None
  
  print 'CSFAK INSIDE compute_gradient_and_loss'
  #############################################################################
  # TODO:                                                                     #
  # Implement the gradient for the required loss, storing the result in dW.	  #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  #pass

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################  
def compute_gradient_and_loss(W, X, y, reg, reg_type, opt):
  """
  loss and gradient function.

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
  - reg_type: (int) regularization type (1: l1, 2: l2)
  - opt: (int) 0 for computing both loss and gradient, 1 for computing loss only
  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  if opt == 0:                          # compute gradient only if opt == 0
      dW = np.zeros(W.shape)            # initialize the gradient as zero
  
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  #############################################################################
  # TODO:                                                                     #
  # Implement the routine to compute the loss, storing the result in loss     #
  #############################################################################  
  for i in xrange(num_train):           # for every augmended image data (3072+1 vector)
      s = X[i].dot(W)                   # compute s (scores)
      s_y = s[y[i]]                     # keep the correct ground truth class score
      max_sj = -999
      argmax_sj = -1
      local_loss = 0.0
      for j in xrange(num_classes):     # for every class        
          if j != y[i]:                 # don't take the correct ground truth index
              if s[j] > max_sj:
                  max_sj = s[j]
                  argmax_sj = j

      term = 1 + max_sj - s_y     # max term with Delta = 1, according to Hinge loss formula 
      
      if term > 0:
          local_loss = term
      
      loss += local_loss
      
      for j in xrange(num_classes):     # for every class        
          if j != y[i]:                 # don't take the correct ground truth index
              if opt == 0:          # compute gradient only if opt == 0
                  if j == argmax_sj:
                      dW[:, j] += X[i]  # this is a analytically with Calculus gradient, case j<>y[i]
                      dW[:, y[i]] -= X[i] #  case j==y[i]
          
                

#  loss /= num_train                    # num_train = M, according to given formula                 

  if reg_type == 1:                     # loss + regularization , l2 or l1
      loss += reg * np.sum(np.abs(W))   # l1, reg is actually lambda regularization strength
  else:
      loss += reg * np.sum(W * W)       # l2
      
  if opt == 0:                          # compute gradient only if opt == 0
      dW /= num_train                   # we have to divide by num_train in order to have the 'mean' gradient
      if reg_type == 1:                 # we use deriv_abs function for l1 derivative
#          dW += reg * deriv_abs(W) #dW[:,-1]
#      else:
#          dW += 2 * reg * W             # l2 derivative formula       
          dW[:-1,:] += reg * np.sign((W[:-1,:])) #dW[:,-1]
      else:
          dW[:-1,:] += 2 * reg * W[:-1,:]             # l2 derivative formula              
      return loss, dW
  else:
      return loss, None
  
  print 'CSFAK INSIDE compute_gradient_and_loss'
  #############################################################################
  # TODO:                                                                     #
  # Implement the gradient for the required loss, storing the result in dW.	  #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  #pass

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################    
def deriv_abs(x):
    mask_pos = (x >= 0) * 1.0       # positive weight branch
    mask_neg = (x < 0) * -1.0       # negative weight branch
    return mask_pos + mask_neg      # add ones and minus ones...

#  return loss, dW