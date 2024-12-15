import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


class Layer_Dense:
    def __init__(self, n_inputs , n_neurons):
        #Generate Matrix of random values. Size: inputs x neurons
        #Shape is transposed already
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) 

        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        #Apply weights, add biases, produce output
        self.output = np.dot(inputs,self.weights) + self.biases


class Activation_ReLU: #Rectify linear unit
    def forward(self,inputs):
        self.output = np.maximum(0,inputs) #Rectify function (0,x)


#==Softmax Activation, Occurs Before output layer==
class Activation_Softmax:
    def forward(self, inputs): 
        exp_values = np.exp(inputs) - np.max(inputs, axis=1, keepdims=True)
        probabilities =  exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
#==





X , y= spiral_data(samples=100, classes=3)
dense1= Layer_Dense(2,3) #2 coordinates in x,y data

activation1 = Activation_ReLU()

dense2= Layer_Dense(3,3)

activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])
