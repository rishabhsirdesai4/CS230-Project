import os
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import h5py
import math
import scipy
from scipy import ndimage
from tensorflow.python.framework import ops
from numpy import zeros, newaxis

import tensorflow as tf
# print ("TensorFlow version: " + tf.__version__)

home = os.path.expanduser("~")
dataRoot = os.path.join(home, "CS230", "CheXpert-v1.0-small")
os.listdir(dataRoot)

# Train dataframe preprocessing
train_df = pd.read_csv(os.path.join(dataRoot, "train.csv"), encoding = "ISO-8859-1")
print(train_df.columns)

# Filtering for Frontal imaging
sub_train = train_df[train_df["Frontal/Lateral"] == "Frontal"]
# Drop AP/PA
sub_train = sub_train.drop(["AP/PA"], axis = 1)
# Convert Unmenttioned to Negatives  i.e. Nah to 0
sub_train = sub_train.fillna(0)
# Convert Uncertain to Positive i.e. -1 to 1
sub_train = sub_train.replace(-1, 1)

# Pull cols to form y_train
cols_train = ["No Finding","Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion","Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax","Pleural Effusion","Pleural Other","Fracture","Support Devices"] #...
Y_train = sub_train[cols_train]
print("the shape of the Y_train set before transpose is:",Y_train.shape)
#Y_train = np.transpose(Y_train)
# Using 1/10th of the data for basline model
Y_train = Y_train[:sub_train.shape[0]/100]
#Y_train = np.transpose(Y_train)
#print(y_train)
print("The shape of the Y_train set after transpose and reduction is:", Y_train.shape)
print("The number of training examples is:",Y_train.shape[1])

print(sub_train.shape[0])

from tqdm import tqdm #to track progress


# importing images from the path mentioned in the.csv file and converting them into numpy arrays
def load_data(img_rows = 224,img_cols = 224): 
    X_train = []   
     
    for image in tqdm(sub_train["Path"][:sub_train.shape[0]/100]):   
        
        img = Image.open("/Users/rishabhsirdesai/CS230/" + image)         
        img.load()         
        img = img.resize((img_rows, img_cols), PIL.Image.ANTIALIAS)         
        data = np.asarray(img, dtype="float64")         
        X_train.append(data)         
    
    return np.array(X_train)
    
X_train = load_data()
print(len(X_train)) #verifying the match between y_train and x_train


print(X_train.shape)

#converint each image in X to a consolidated tensor
tensor_X_train = [tf.convert_to_tensor(image_data, np.float32) for image_data in X_train]

# Valid data preprocessing

valid_df = pd.read_csv(os.path.join(dataRoot, "valid.csv"), encoding = "ISO-8859-1")

# Filtering for Frontal imaging
sub_valid = valid_df[valid_df["Frontal/Lateral"] == "Frontal"]
# Drop AP/PA
sub_valid = sub_valid.drop(["AP/PA"], axis = 1)
# Convert Unmenttioned to Negatives  i.e. Nah to 0
sub_valid = sub_valid.fillna(0)
# Convert Uncertain to Positive i.e. -1 to 1
sub_valid = sub_valid.replace(-1, 1)

# Pull cols to form y_valid
cols_valid = ["No Finding","Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion","Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax","Pleural Effusion","Pleural Other","Fracture","Support Devices"] #...
Y_test = sub_valid[cols_valid]
print("the shape of the y_valid set before transpose is:",
      Y_test.shape)
#Y_test = np.transpose(Y_test)
# Using 1/10th of the data for basline model
Y_test = Y_test[:sub_valid.shape[0]]
#print(y_train)
print("The shape of the Y_valid set is:", Y_test.shape)
print("The number of valid examples is:",Y_test.shape[1])

from tqdm import tqdm #to track progress


# importing images from the path mentioned in the.csv file and converting them into numpy arrays
def load_data_valid(img_rows = 224,img_cols = 224): 
    X_test = []   
     
    for image_valid in tqdm(sub_valid["Path"][:sub_valid.shape[0]]):   
        
        img_valid = Image.open("/Users/rishabhsirdesai/CS230/" + image_valid)         
        img_valid.load()         
        img_valid = img_valid.resize((img_rows, img_cols), PIL.Image.ANTIALIAS)         
        data_valid = np.asarray(img_valid, dtype="float64")         
        X_test.append(data_valid)         
    
    return np.array(X_valid)
    
X_test = load_data_valid()

#converint each image in X to a consolidated tensor
tensor_X_test = [tf.convert_to_tensor(image_data, np.float32) for image_data in X_test]
X_train = np.reshape(X_train , (1910,224,224,1))
X_test = np.reshape(X_test , (202,224,224,1))
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
conv_layers = {}
Y_test = np.array(Y_test)
Y_train = np.array(Y_train)

# Create Placeholders

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0), name='X')
    Y = tf.placeholder(tf.float32, shape=(None, n_y), name='Y') 
    
    return X, Y

X, Y = create_placeholders(224, 224, 1, 14)
print ("X = " + str(X))
print ("Y = " + str(Y))

def initialize_parameters():
    tf.set_random_seed(1)                   

    W1 = tf.get_variable("W1", shape = (4,4,1,8), initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2", shape = (2,2,8,16), initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    parameters = {"W1": W1,
                  "W2": W2}
    
    return parameters

def forward_propagation(X, parameters):
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 
    Z3 = tf.contrib.layers.fully_connected(P2, 14, activation_fn=tf.nn.sigmoid)
    return Z3

tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(1)
    X, Y = create_placeholders(224, 224, 1, 14)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(Z3, {X: np.random.randn(2,224,224,1), Y: np.random.randn(2,14)})
    print("Z3 = " + str(a))

def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))

    return cost

tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(1)
    X, Y = create_placeholders(224,224, 1, 14)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(cost, {X: np.random.randn(4,224,224,1), Y: np.random.randn(4,14)})
    print("cost = " + str(a))

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):

    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
# GRADED FUNCTION: model

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 64, print_cost = True):
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    seed = 3                                          # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]                            
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    # Initialize parameters
    parameters = initialize_parameters()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
     
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                _ , temp_cost = sess.run([optimizer, cost], {X: minibatch_X, Y: minibatch_Y})
                
                minibatch_cost += temp_cost / num_minibatches
                

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        
        
        # plot the cost
        # plt.plot(np.squeeze(costs))
        # plt.ylabel('cost')
        # plt.xlabel('iterations (per tens)')
        # plt.title("Learning rate =" + str(learning_rate))
        # plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
                
        return train_accuracy, test_accuracy, parameters
_, _, parameters = model(X_train, Y_train, X_test, Y_test)