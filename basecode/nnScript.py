import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W









####################################################################################################################################

def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return 1.0 / (1.0 + np.exp(-z))  #sigmoid formula

####################################################################################################################################








####################################################################################################################################

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    
    #init empty lists
    all_train_data = []
    all_train_labels = []
    test_data = []
    test_label = []
    
    #loop thru digits 0-9
    for digit in range(10):
        #grab data for this digit
        digit_examples = mat['train' + str(digit)]
        #save training examples
        all_train_data.append(digit_examples)
        #create same-value labels
        all_train_labels.append(np.ones(len(digit_examples)) * digit)
        
        #same for test data
        test_examples = mat['test' + str(digit)]
        test_data.append(test_examples)
        test_label.append(np.ones(len(test_examples)) * digit)
    
    #stack everything into arrays
    all_train_data = np.vstack(all_train_data)
    all_train_labels = np.hstack(all_train_labels)
    test_data = np.vstack(test_data)
    test_label = np.hstack(test_label)
    
    #shuffle training data
    shuffle_idx = np.random.permutation(len(all_train_data))
    
    #split into train/valid sets
    train_data = all_train_data[shuffle_idx[:50000]]
    train_label = all_train_labels[shuffle_idx[:50000]]
    validation_data = all_train_data[shuffle_idx[50000:60000]]
    validation_label = all_train_labels[shuffle_idx[50000:60000]]

    # Feature selection
    # Your code here.
    
    #find non-constant features
    feature_variance = np.var(train_data, axis=0)
    selected_features = np.where(feature_variance > 0)[0]
    
    #keep only useful features
    train_data = train_data[:, selected_features]
    validation_data = validation_data[:, selected_features]
    test_data = test_data[:, selected_features]
    
    #normalize to 0-1
    train_data = train_data/255.0
    validation_data = validation_data/255.0
    test_data = test_data/255.0

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label

####################################################################################################################################











####################################################################################################################################

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here

    #get number of samples
    n = training_data.shape[0]
    
    #create bias column (fill w 1s)
    bias = np.ones(n)
    
    #add bias to training data
    input_with_bias = np.column_stack([training_data, bias])
    

    #feedforward to hidden layer
    #####################################################
    #multiply inputs with weights
    hidden_inputs = np.dot(input_with_bias, w1.T)
    #apply activation function
    hidden_outputs = sigmoid(hidden_inputs)
    
    #add bias to hidden layer outputs
    hidden_with_bias = np.column_stack([hidden_outputs, bias])
    #####################################################


    #feedforward to output layer
    #####################################################
    #multiply hidden outputs with weights
    output_inputs = np.dot(hidden_with_bias, w2.T)
    #apply activation function
    output_final = sigmoid(output_inputs)
    #####################################################


    #prepare true labels matrix
    #####################################################
    true_labels = np.zeros((n, n_class))
    for i in range(n):
        #set 1 at position of correct class
        true_labels[i, int(training_label[i])] = 1
    #####################################################


    #calculate error
    #####################################################
    #for each example -[y*log(o) + (1-y)*log(1-o)]
    pos_term = true_labels * np.log(output_final)
    neg_term = (1 - true_labels) * np.log(1 - output_final)
    error_matrix = pos_term + neg_term
    error = -np.sum(error_matrix) / n
    #####################################################


    #add regularization
    #####################################################
    #reg_term = (lambda/2n) * (sum of squared weights)
    reg_term = (lambdaval/(2*n)) * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    obj_val = error + reg_term
    #####################################################
    

    #backpropagation
    #####################################################
    #output layer error
    delta_output = output_final - true_labels
    
    #backpropagate to hidden layer
    #error * weights * derivative of sigmoid
    delta_hidden_with_bias = np.dot(delta_output, w2) * (hidden_with_bias * (1 - hidden_with_bias))
    #remove bias error
    delta_hidden = delta_hidden_with_bias[:, :-1]
    #####################################################
    

    #calculate gradients
    #####################################################
    #gradient = (input.T * delta)/n + regularization
    grad_w2 = (np.dot(delta_output.T, hidden_with_bias) + lambdaval * w2) / n
    grad_w1 = (np.dot(delta_hidden.T, input_with_bias) + lambdaval * w1) / n
    #####################################################
    

    #combine gradients into single vector
    #####################################################
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)
    #####################################################


    return (obj_val, obj_grad)

####################################################################################################################################



















####################################################################################################################################

def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here

    #add bias to input
    n = data.shape[0]
    input_with_bias = np.hstack((data, np.ones((n, 1))))
    
    #feedforward - hidden layer
    hidden_layer_input = np.dot(input_with_bias, w1.T)
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    #add bias to hidden layer
    hidden_with_bias = np.hstack((hidden_layer_output, np.ones((n, 1))))
    
    #feedforward - output layer
    output_layer_input = np.dot(hidden_with_bias, w2.T)
    output_layer_output = sigmoid(output_layer_input)
    
    #get predictions - class with highest prob
    labels = np.argmax(output_layer_output, axis=1)

    return labels

####################################################################################################################################











"""**************Neural Network Script Starts here********************************"""
if __name__ == "__main__":
    
        
    train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

    #  Train Neural Network

    # set the number of nodes in input unit (not including bias unit)
    n_input = train_data.shape[1]

    # set the number of nodes in hidden unit (not including bias unit)
    n_hidden = 50

    # set the number of nodes in output unit
    n_class = 10

    # initialize the weights into some random matrices
    initial_w1 = initializeWeights(n_input, n_hidden)
    initial_w2 = initializeWeights(n_hidden, n_class)

    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

    # set the regularization hyper-parameter
    lambdaval = 0

    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

    # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

    opts = {'maxiter': 50}  # Preferred value.

    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

    # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
    # and nnObjGradient. Check documentation for this function before you proceed.
    # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


    # Reshape nnParams from 1D vector into w1 and w2 matrices
    w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # Test the computed parameters

    predicted_label = nnPredict(w1, w2, train_data)

    # find the accuracy on Training Dataset

    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, validation_data)

    # find the accuracy on Validation Dataset

    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, test_data)

    # find the accuracy on Validation Dataset

    print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')


