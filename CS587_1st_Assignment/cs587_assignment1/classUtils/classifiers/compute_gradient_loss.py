import numpy as np
from random import shuffle

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
    - opt: (int) 0ny element of an iterable is True. If not, any() returns False. for computing both loss and gradient, 1 for computing loss only
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    num_dim = X.shape[1]
    loss = 0.0
    delta = 1.0

    #############################################################################
    # TODO:                                                                     #
    # Implement the routine to compute the loss, storing the result in loss     #
    #############################################################################
    
    #############################################################################
    # TODO:                                                                     #
    # Implement the gradient for the required loss, storing the result in dW.	  #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    
    scores = X.dot(W) # (n,d)(d,c) -> (n,c)
    
    # margin term 
    for i in range(num_train):
        score = scores[i,:] # row (1,10) of scores, 1 for each class
        score_y = score[y[i]] # score yi : the score of the correct class 
        score_j = score[np.arange(num_classes) != y[i]] # scores j : the other 9 scores
        score_j_max = np.max(score_j) # val of max score j
        margin = (score_j_max + delta) - score_y 
        if margin > 0:
            indcs = np.where(score == score_j_max) # indexes of scores = score_j_max
            indx_j_max = indcs[0][indcs[0]!=y[i]] # if, unluckily, y[i] was also equal to score_j_max, then ignore it.
                                                  # if there are two equally max dims among classes 'j', 
                                                  # then update weights for both. 
            loss += margin
            if opt==0:
                #print indcs[0][0], indx_j_max[0]
                #print dW[:, indcs[0][0]].shape, X[i].shape
                #print dW[:, y[i]].shape, X[i].shape
                dW[:, indx_j_max[0]] += X[i] # (3073,)
                dW[:, y[i]] -= X[i] # (3073,)
    #loss /= num_train # normalize loss
            
    # regularization term        
    if reg_type == 1:
        reg_term = np.sum(np.abs(W))
        reg_der = np.sign(W)
    elif reg_type == 2:
        reg_term = np.sum(W*W)
        reg_der = 2*W
    else:
        print('reg_type should be 1 or 2')
    
    loss += reg * reg_term # add regularization term to loss function 
    
    if opt==0:
        dW /= num_train
        dW[:-1,:] += reg * reg_der[:-1,:] # do not apply regularization to the last line of bias 
    
    # output
    if opt == 0:
        return loss, dW
    else:
        return loss