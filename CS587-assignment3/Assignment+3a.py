# coding: utf-8

# # CS - 587 : Exercise 3a
# ## Scope:
# The goal of this assignment is to get familiar with fine-tunning in a new dataset a Convolutional Neural Network (CNN) that has been trained in another dataset, taking advantage of transfer learning.
# 
# In your assignment you will be fine-tunning AlexNet a popular CNN architecture, that has been pretrained on ImageNet dataset. Your network will be finetuned for the task of recognizing art painting categories in a large dataset of art painting images, known as Wikiart.
# 
# The WikiArt dataset, which consists of 4000 images of paintings of arbitrary sizes from 10 different styles - Baroque, Realism, Expressionism, etc.
# 
# 

# In[4]:


import os
os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'
import tensorflow as tf
import numpy as np
from models.AlexNet import AlexNet
from Utilities.datagenerator import ImageDataGenerator
import sys
from datetime import datetime
from tqdm import tqdm
import urllib

"""
Configuration settings
"""
weight_path= os.path.join('weights','bvlc_alexnet.npy')
general_path_weights = os.path.join('weights')

# Create parent path if it doesn't exist
if not os.path.isdir(general_path_weights): 
    os.mkdir(general_path_weights)
 
def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize: # near the end
            sys.stderr.write("\n")
    else: # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))   
    
if os.path.isfile(weight_path) == False:
    print('Went to download weights for AlexNet, 230Mb from your disk will be put to good use! ')
    print('Relax...')
    weight_file = urllib.request.urlretrieve('http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy',os.path.join('weights/bvlc_alexnet.npy'),reporthook)
    print('Done with weights!')
else:
    print('GOOD TO GO! Weights already downloaded and stored!')
    

tf.logging.set_verbosity(tf.logging.ERROR)

# Path to the textfiles for the trainings and validation set
training_dirname = os.path.join('Utilities','data', 'train.txt')

val_dirname = os.path.join('Utilities','data', 'test.txt')

# Path for tf.summary.FileWriter and to store model checkpoints
general_path = os.path.join('finetune_alexnet')
filewriter_path = os.path.join('finetune_alexnet','wikiart')
checkpoint_path = os.path.join('finetune_alexnet','CheckPoint')

# Create parent path if it doesn't exist
if not os.path.isdir(general_path): 
    os.mkdir(general_path)
# Create parent path if it doesn't exist
if not os.path.isdir(filewriter_path): os.mkdir(filewriter_path)
# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)


# In[ ]:


# Learning params
learning_rate = 0.01
num_epochs = 30
batch_size = 32

# Network params
dropout_rate = 0.5
num_classes = 10

#---------------LOOK AT ME----------------------------------------------------

train_layers = ['fc8', 'fc7'] # Change me if you want to try stuff

#------------------------------------------------------------------------------

# # Pretrained Model
# For all of our image generation experiments, we will start with a convolutional neural network which was pretrained to perform image classification on ImageNet. We can use any model here, but for the purposes of this assignment we will use AlexNet

# In[ ]:


# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Link variable to model output
score = model.fc8

################################################################################################
# TODO: Implement the (a) losss function (Soft-max Cross Entropy), (b) the optimization        #
# process using Gradient Descent, (c) accuracy (using argmax). Create summaries in tensorboard #
# for the loss, the gradients of trainable variables (histogram form) and the accuracy.        #
#                                                                                              # 
# Hint: in order to take the gradients per variable use tf.gradient(,)                         #
# https://www.tensorflow.org/api_docs/python/tf/gradients  -> based on the loss                #
################################################################################################    
 

pass



  
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Initalize the data generator seperately for the training and validation set
train_generator = ImageDataGenerator(training_dirname, 
                                     horizontal_flip = True, shuffle = True)
val_generator = ImageDataGenerator(val_dirname, shuffle = False) 

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)



# In[ ]:


# Start Tensorflow session
with tf.Session() as sess:
 
  # Initialize all variables
  sess.run(tf.global_variables_initializer())
  
  # Add the model graph to TensorBoard
  writer.add_graph(sess.graph)
  
  # Load the pretrained weights into the non-trainable layer
  model.load_initial_weights(sess)
  
  print("{} Start training...".format(datetime.now()))
  print("{} Open Tensorboard at --logdir {}".format(datetime.now(), 
                                                    filewriter_path))
  
  # Loop over number of epochs
  for epoch in range(num_epochs):
    
        print("{} Epoch number: {}".format(datetime.now(), epoch+1))
        
        step = 1
        
        for step in tqdm(range(train_batches_per_epoch)):
            
            # Get a batch of images and labels
            batch_xs, batch_ys = train_generator.next_batch(batch_size)
            
            ######################################################################################
            #TODO: Run the training operation, print the current loss value write the symmaries.  #
            #      The summarries must be written every 3 batches                                #
            ######################################################################################
            
           
            pass
            
            #End of this task
            
        ############################################################
        #TODO: Validate the model on the ENTIRE validation set     #
        #      Print the final validation accuracy of your model   #
        ############################################################   
        
        
        
        pass
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        
        # Reset the file pointer of the image data generator
        val_generator.reset_pointer()
        train_generator.reset_pointer()
        
        print("{} Saving checkpoint of model...".format(datetime.now()))  
        
        #save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)  
        
        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
        

