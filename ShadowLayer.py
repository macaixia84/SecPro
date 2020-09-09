import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model

from planar_utils import plot_decision_boundary,sigmoid,load_planar_dataset,load_extra_datasets
from testCases import *
np.random.seed(1)

X,Y = load_planar_dataset()
plt.scatter(X[0,:],X[1,:],c = Y.ravel(),s=40,cmap=plt.cm.Spectral)

shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1]
print('X的维度是：'+str(shape_X))
print('Y的维度是：'+str(shape_Y))
print('训练样本的个数是：'+str(m))

clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T,Y.T.ravel())
LR_predictions = clf.predict(X.T)
print('预测准确度是：%d'%float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) + '%')
plot_decision_boundary(lambda x:clf.predict(x),X,Y.ravel())




def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros(shape = (n_h,1))

    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros(shape = (n_y,1))

    parameters = {'W1':W1,'b1':b1,'W2':W2,'b2':b2}
    return parameters
def forward_propagation(X,parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1":Z1,"A1":A1,"Z2":Z2,"A2":A2}
    return A2,cache

def compute_cost(A2,Y,parameters):
    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2),Y) + np.multiply((1-Y),np.log(1-A2))
    cost = -np.sum(logprobs) / m
    return cost
def backward_propagation(parameters,cache,X,Y):
    m = X.shape[1]

    W1 = parameters['W1']
    W2 = parameters['W2']

    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y
    dW2 = (1/m)*np.dot(dZ2,A1.T)
    db2 = (1/m)*np.sum(dZ2,axis = 1,keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T,dZ2),1-np.power(A1,2))
    dW1 = (1/m)*np.dot(dZ1,X.T)
    db1 = (1/m)*np.sum(dZ1,axis=1,keepdims=True)

    print("W1 dim:"+str(W1.shape))
    print('W2 dim:'+str(W2.shape))
    print("---------------------------")

    print("dZ2 dim:"+str(dZ2.shape))
    print("dZ1 dim:"+str(dZ1.shape))
    print("-----------------------------")

    grads = {'dW1':dW1,'db1':db1,'dW2':dW2,'db2':db2}
    return grads

def update_parameters(parameters,grads,learning_rate=1.2):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {'W1':W1,'b1':b1,'W2':W2,'b2':b2}
    return parameters

def nn_model(X,Y,n_h,num_iterations = 10000,print_cost = False):
    np.random.seed(3)
    n_x = X.shape[0]
    n_y = Y.shape[0]

    parameters = initialize_parameters(n_x,n_y,n_h)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    for i in range(0,num_iterations):
        A2,cache = forward_propagation(X,parameters)
        cost = compute_cost(A2,Y,parameters)
        grads = backward_propagation(parameters,cache,X,Y)
        parameters = update_parameters(parameters,grads)
        if print_cost and i%1000 == 0:
            print("在训练%i次后，成本是：%f"%(i,cost))
    return parameters

def predict(parameters,X):
    A2,cache = forward_propagation(X,parameters)
    predictions = np.round(A2)
    return predictions

parameters, X_assess = predict_test_case()

predictions = predict(parameters, X_assess)
print("predictions mean = " + str(np.mean(predictions)))

parameters = nn_model(X, Y, n_h = 4, num_iterations=10000, print_cost=True)
predictions = predict(parameters,X)
# print('预测准确率是: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')
# plot_decision_boundary(lambda x:predict(parameters,X.T),X,Y.ravel())

# 展示不同神经元个数的不同准确度。这段代码可以要运行几分钟。

plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50] # 不同的神经元个数
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i + 1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations=5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y.ravel())
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    print ("{}个隐藏层神经元时的准确度是: {} %".format(n_h, accuracy))









