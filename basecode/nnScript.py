import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle
import matplotlib.pyplot as plt
import time


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


    #add regularization (excluding bias weights)
    #####################################################
    #reg_term = (lambda/2n) * (sum of squared weights excluding bias)
    w1_no_bias = w1[:, :-1]  
    w2_no_bias = w2[:, :-1] 
    
    reg_term = (lambdaval/(2*n)) * (np.sum(np.square(w1_no_bias)) + np.sum(np.square(w2_no_bias)))
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
    
    #w2 gradient
    grad_w2 = np.dot(delta_output.T, hidden_with_bias) / n
    
    # regularization term excluding bias weights
    grad_w2_reg = grad_w2.copy()
    grad_w2_reg[:, :-1] = grad_w2[:, :-1] + (lambdaval/n) * w2[:, :-1]
    
    # for w1 gradient
    grad_w1 = np.dot(delta_hidden.T, input_with_bias) / n
    
    # add regularization term excluding bias 
    grad_w1_reg = grad_w1.copy()
    grad_w1_reg[:, :-1] = grad_w1[:, :-1] + (lambdaval/n) * w1[:, :-1]
    #####################################################
    

    #combine gradients into single vector
    #####################################################
    obj_grad = np.concatenate((grad_w1_reg.flatten(), grad_w2_reg.flatten()), 0)
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











# Hyperparam tuning
####################################################################################################################################
#test diff hidden units & lambda vals
hidden_units = [4, 8, 12, 16, 20] 
lambda_values = [0, 10, 20, 30, 40, 50] 

#store results
val_acc = np.zeros((len(hidden_units), len(lambda_values)))
train_times = np.zeros((len(hidden_units), len(lambda_values)))

print("Starting hyperparameter tuning...")
#try all combos
for i, n_hidden in enumerate(hidden_units):
    for j, lambdaval in enumerate(lambda_values):
        print(f"Testing h={n_hidden}, lambda={lambdaval}")
        
        #init weights
        initial_w1 = initializeWeights(n_input, n_hidden)
        initial_w2 = initializeWeights(n_hidden, n_class)
        initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)
        
        #setup args
        args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)
        
        #time the training
        import time
        start = time.time()
        opts = {'maxiter': 50}
        nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
        end = time.time()
        train_times[i, j] = end - start
        
        #get weights
        w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
        w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
        
        #calc validation acc
        pred = nnPredict(w1, w2, validation_data)
        acc = 100 * np.mean((pred == validation_label).astype(float))
        val_acc[i, j] = acc
        
        print(f"Val acc: {acc:.2f}%, Time: {train_times[i, j]:.2f}s")

#find best params
best_idx = np.unravel_index(np.argmax(val_acc), val_acc.shape)
best_hidden = hidden_units[best_idx[0]]
best_lambda = lambda_values[best_idx[1]]

print(f"Best params: h={best_hidden}, lambda={best_lambda}, acc={val_acc[best_idx]:.2f}%")

#plot results for video
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))

#plot1: lambda vs acc
plt.subplot(1, 2, 1)
for i, h in enumerate(hidden_units):
    plt.plot(lambda_values, val_acc[i], 'o-', label=f'{h} units')
plt.xlabel('Lambda')
plt.ylabel('Accuracy (%)')
plt.title('Lambda effect on accuracy')
plt.legend()
plt.grid(True)

#plot2: hidden units vs time
plt.subplot(1, 2, 2)
avg_time = np.mean(train_times, axis=1)
plt.plot(hidden_units, avg_time, 'o-')
plt.xlabel('Hidden Units')
plt.ylabel('Time (s)')
plt.title('Training time vs hidden units')
plt.grid(True)

plt.tight_layout()
plt.savefig('tuning_results.png')

#train final model w best params
print("Training final model...")
initial_w1 = initializeWeights(n_input, best_hidden)
initial_w2 = initializeWeights(best_hidden, n_class)
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

args = (n_input, best_hidden, n_class, train_data, train_label, best_lambda)
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

#extract final weights
w1 = nn_params.x[0:best_hidden * (n_input + 1)].reshape((best_hidden, (n_input + 1)))
w2 = nn_params.x[(best_hidden * (n_input + 1)):].reshape((n_class, (best_hidden + 1)))

#test final model
train_acc = 100 * np.mean((nnPredict(w1, w2, train_data) == train_label).astype(float))
valid_acc = 100 * np.mean((nnPredict(w1, w2, validation_data) == validation_label).astype(float))
test_acc = 100 * np.mean((nnPredict(w1, w2, test_data) == test_label).astype(float))

print(f"Train acc: {train_acc:.2f}%")
print(f"Valid acc: {valid_acc:.2f}%")
print(f"Test acc: {test_acc:.2f}%")

#save params for submission
selected_features = np.where(np.var(train_data, axis=0) > 0)[0]
params = {
    'selected_features': selected_features,
    'optimal_n_hidden': best_hidden,
    'w1': w1,
    'w2': w2,
    'optimal_lambda': best_lambda
}

with open('params.pickle', 'wb') as f:
    pickle.dump(params, f)