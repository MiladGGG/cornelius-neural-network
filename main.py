import numpy as np



class Layer_Dense: #Creates Neuron layer with random weights
    def __init__(self, n_inputs , n_neurons):
        #Generate Matrix of random values. Size: inputs x neurons
        #Shape is transposed already
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) 

        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        #Apply weights, add biases, produce output
        self.output = np.dot(inputs,self.weights) + self.biases

    def backward(self, dvalues, learning_rate): #BACKPROPAGATE WOO
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        self.dinputs = np.dot(dvalues,self.weights.T)

        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases


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


class Loss:
    def calculate(self,output,y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CatergoricalCrossentropy(Loss): #Calculate output layer loss
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1: #Scalar values "[0,1]"
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2: #One hot encoded vector "[[0,1],[1,0]]"
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1) #sum turns matrix into vector
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return correct_confidences





network_initialised = False

def initialise_network(inputs):
    pass    


def run_network(inputs):
    
    #Hidden Layer #1
    Layer1 = Layer_Dense(len(inputs), 64)
    Activation1 = Activation_ReLU()

    Layer1.forward(inputs)
    Activation1.forward(Layer1.output)

    #Hidden Layer #2
    Layer2 = Layer_Dense(64,64)
    Activation2 = Activation_ReLU()

    Layer2.forward(Activation1.output)
    Activation2.forward(Layer2.output)
    
    #Output Layer
    OutputLayer = Layer_Dense(64, 2)
    ActivationOutput = Activation_Softmax()

    OutputLayer.forward(Activation2.output)
    ActivationOutput.forward(OutputLayer.output)

    #Probabilities
    probabilities = ActivationOutput.output 
    print("Probabilities: ", probabilities)
    
    #Loss
    true_values = np.array([1,0])
    Loss = Loss_CatergoricalCrossentropy()
    loss_output = Loss.forward(probabilities, true_values)
    print("Loss: ", loss_output)
    



#example data
"""
X , y= spiral_data(samples=100, classes=3)
dense1= Layer_Dense(2,3) #2 coordinates in x,y data

activation1 = Activation_ReLU()

dense2= Layer_Dense(3,3)

acti    vation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])
loss_function = Loss_CatergoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)
print("Loss: ",loss)
"""
