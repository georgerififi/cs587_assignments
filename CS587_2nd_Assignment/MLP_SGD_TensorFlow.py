
# coding: utf-8

# # Build, Train & Test a Multilayer Neural Networks using TensorFlow
# 
# 
# ### Goals: 
# - Intro: build and train a feed forward neural network using the `TensorFlow` framework.
# - The SGD method will be used for training to apply automatic differentiation based on TensorFlow.
# - Tune the hyperparameters and modify the structure of your NN to achieve the highest accuracy.
# - Use Tensorboard to visualize the graph and results.
# 
# ### Dataset:
# - Digits: 10 class handwritten digits
# - It will automatically be downloaded once you run the provided code using the scikit-learn library.
# - Check for info in the following websites:
# - http://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html
# - http://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits
# - http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits

# In[ ]:

# get_ipython().magic('matplotlib inline')
# display figures in the notebook
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder #csfak
digits = load_digits()


# In[ ]:

sample_index = 45
plt.figure(figsize=(3, 3))
plt.imshow(digits.images[sample_index], cmap=plt.cm.gray_r,
           interpolation='nearest')
plt.title("image label: %d" % digits.target[sample_index]);


# ### Preprocessing
# 
# - normalization of your input data
# - train/test split

# In[ ]:

from sklearn import preprocessing
# import numpy #CSFAK COMMEND
# numpy.set_printoptions(threshold=numpy.nan) #CSFAK COMMEND

data = np.asarray(digits.data, dtype='float32')
target = np.asarray(digits.target, dtype='int32')

X_train = data[0:1500,:]
y_train = target[0:1500]

X_test = data[1500:,:]
y_test = target[1500:]

# mean = 0 ; standard deviation = 1.0
scaler = preprocessing.StandardScaler()

# print(scaler.mean_)
# print(scaler.scale_)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Check that the train and test targets/labels are balanced within each set
plt.hist(y_train)  # plt.hist passes it's arguments to np.histogram
plt.title("Histogram of train labels/targets")
plt.show()

plt.hist(y_test)  # plt.hist passes it's arguments to np.histogram
plt.title("Histogram of test labels/targets")
plt.show()


# Let's display the one of the transformed sample (after feature standardization):

# In[ ]:

sample_index = 150
plt.figure(figsize=(3, 3))
plt.imshow(X_train[sample_index].reshape(8, 8),
           cmap=plt.cm.gray_r, interpolation='nearest')
plt.title("transformed sample\n(standardization)");


# The scaler objects makes it possible to recover the original sample:

# In[ ]:

plt.figure(figsize=(3, 3))
plt.imshow(scaler.inverse_transform(X_train[sample_index]).reshape(8, 8),
           cmap=plt.cm.gray_r, interpolation='nearest')
plt.title("original sample");


# In[ ]:

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# ### TensorFlow is a symbolic graph computation engine, that allows automatic differentiation of each node
# - https://www.tensorflow.org 
# - https://www.tensorflow.org/tutorials/mnist/tf/
# 
# TensorFlow builds where nodes may be:
# - **constant:** constants tensors, such as a learning rate
# - **Variables:** any tensor, such as parameters of the models
# - **Placeholders:** placeholders for inputs and outputs of your models
# - many other types of nodes (functions, loss, ...)
# 
# The graph is symbolic, no computation is performed until a `Session` is defined and the command `run` or `eval` is invoked. TensorFlow may run this computation on (multiple) CPUs or GPUs

# In[ ]:

import tensorflow as tf

a = tf.constant(3)
b = tf.constant(2)
c = tf.Variable(0)
c = a + b
with tf.Session() as sess:
    print(sess.run(c))


# In[ ]:

X = tf.placeholder("float32", name="input")
Y = X + tf.constant(3.0)
with tf.Session() as sess:
    print(sess.run(Y, feed_dict={X:2}))


# **Note: batches in inputs**
# - the first dimension of the input is usually kept for the batch dimension. A typical way to define an input placeholder with a 1D tensor of 128 dimensions, is:
# ```
# X = tf.placeholder("float32", shape=[None, 128])
# ```

# ## 1) Build a model using TensorFlow
# 
# - Using TensorFlow, build a simple model (one hidden layer)

# In[ ]:

# helper functions

def init_weights(shape, node_name):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name = node_name)

def accuracy(y_pred, y=y_test):
    return np.mean(np.argmax(y_pred, axis=1) == y)


# In[ ]:

# BUILD YOUR MODEL, LOSS, PREDICT & TRAIN OPERATORS & INIT ALL VARIABLES IN THE GRAPH


# hyperparams
batch_size = 32
hid_size = 15
learning_rate = 0.01
num_epochs = 10
input_size = X_train.shape[1]
output_size = 10

# input and output
X = tf.placeholder("float32", shape=[None, input_size])
# y = tf.placeholder("int32", shape=[None])
y = tf.placeholder("float32", shape=[None,output_size]) #CSFAK
#todo: build the model and weights
#W_h = []#CSFAK COMMEN D
W_h = init_weights([input_size, hid_size], "W_h") #CSFAK
b_h = init_weights([hid_size], "b_h") #CSFAK
W_o = init_weights([hid_size, output_size], "W_o") #CSFAK
b_o = init_weights([output_size], "b_o") #CSFAK

#define/set your activation function and the output of your net
# h = []
h = tf.add(tf.matmul(X, W_h), b_h) #CSFAK h = W_h*x + b_h
h = tf.nn.sigmoid(h) #CSFAK
#h = tf.nn.relu(h) #CSFAK
# out_act = []

out_act = tf.nn.softmax(tf.add(tf.matmul(h, W_o), b_o)) #CSFAK 
#todo: build the loss using softmax cross entropy error function, 
# mock loss and b, you need to change
# b = init_weights([output_size], "b") #CSFAK TEMP
b = -np.sum(tf.reduce_sum(y * tf.log(out_act))) #CSFAK
loss = b
# loss = -tf.reduce_mean(tf.reduce_sum(y * tf.log(out_act))) #CSFAK
#todo: define/set your optimizer (Stochastic Gradient Descent) using the given learning rate value
#and the train operator
# optimizer = []
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss) #CSFAK
# train_op = [] 
train_op = optimizer #CSFAK
#todo: build predict node using the softmax error function
# predict = X #CSFAK
predict = tf.equal(tf.argmax(y, 1), tf.argmax(out_act, 1))#CSFAK

my_accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

# add a summary to store the accuracy
tf.summary.scalar('accuracy', my_accuracy)
# Initialization of all variables in the graph
init = tf.global_variables_initializer()

# ### 2) Train your model using SGD algorithm and check the generalization on the test set of your dataset.

# In[ ]:

#Init your session, run training
#Render your graph and monitor your training procedure using TensorBoard

with tf.Session() as sess: 
    sess.run(init)
    
    losses = []
    one_hot_all_targets_y_train = np.eye(output_size)[y_train] #CSFAK
    one_hot_all_targets_y_test = np.eye(output_size)[y_test] #CSFAK
    for e in range(num_epochs):
        for i in range(X_train.shape[0] // batch_size):
            # Build batches of batch_size            
            idx, idxn = i * batch_size, min(X_train.shape[0]-1, (i+1) * batch_size)
            batch_xs, batch_ys = X_train[idx: idxn], y_train[idx: idxn]            
            one_hot_targets = np.eye(output_size)[batch_ys] #CSFAK
            # Run train operator for the current batch and save/monitor loss.
            _, l = sess.run([train_op, loss], feed_dict={X: batch_xs, y: one_hot_targets})
            losses.append(l)
        
        #to-do: For each epoch, run accuracy on train and test.
        
        predicts_train = sess.run(predict, feed_dict={X: X_train, y: one_hot_all_targets_y_train})
        
        
        predicts_test = sess.run(predict, feed_dict={X: X_test, y: one_hot_all_targets_y_test})
 
        #to-do: call the accuracy function to compute accuracy scores and print them
        # acc_train = 0
        acc_train = np.count_nonzero(predicts_train)/predicts_train.shape[0]
        acc_test = np.count_nonzero(predicts_test)/predicts_test.shape[0]
        print("epoch: %d train accuracy: %0.3f test accuracy: %0.3f"
              % (e, acc_train, acc_test))
    
    # For monitoring purposes
    file_writer = tf.summary.FileWriter('./tensorflow_summaries', sess.graph)

plt.figure(figsize=(5, 5)) #CSFAK   
plt.plot(losses);
plt.title("Loss during iteration");

#to-do Do not forget to USE TENSORBOARD for graph rendereing + monitoring your training process 

# 
# ### 3) In order to maximize the accuracy on the given dataset try different settings for your model
# 
# Play around with the structure of your NN model and fine-tune its hyperparameters.
# 
# - A. Experiment with di fferent hyperparameters (learning rate = 0.001,..,0.1, batch size = 8,..,128, size of hidden layers = 5,..,25, number of epochs).
# - B. Try di fferent activation functions (e.g., ReLU, TanH).
# - C. Try to add more hidden layers and increase their size.
# - D. Add L2 regularization (e.g., with regularization strength 10^(-4))
# 
# ### Bonus: A + 15% will be distributed to the top-performing models based on the accuracy on the test set (e.g if there are K submissions with equal top performance, each one will get a bonus 15%/K)

# In[ ]:

#to-do- MAXimize the accuracy on the given dataset try different settings for your model

