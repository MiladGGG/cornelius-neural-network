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

class Loss:
    def calculate(self,output,y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CatergoricalCrossentropy(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1: #Scalar values "[0,1]"
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2: #One hot encoded vector "[[0,1],[1,0]]"
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1) #sum turns matrix into vector
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return correct_confidences



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
loss_function = Loss_CatergoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)
print("Loss: ",loss)
