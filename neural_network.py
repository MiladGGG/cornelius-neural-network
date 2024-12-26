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

        self.inputs = np.array(inputs)  #Store to be used in backward pass

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
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities =  exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

        self.dinputs[self.inputs <= 0] = 0
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

    def backward(self, y_pred, y_true):
        samples = len(y_pred)

        self.dvalues = y_pred - y_true
        self.dvalues = self.dvalues / samples

        return self.dvalues










class Neural_Network:

    def initialise_network(self):
        #5 outputs, 64 neurons in each hidden layers, 900 px input
        self.learning_rate = 0.01
        self.hasRun = False
        self.cumalative_int = 0
 
        #Hidden Layer #1, 30 by 30px input
        self.Layer1 = Layer_Dense(900, 64)
        self.Activation1 = Activation_ReLU()
    
        #Hidden Layer #2
        self.Layer2 = Layer_Dense(64,64)
        self.Activation2 = Activation_ReLU()

        #Output Layer
        self.OutputLayer = Layer_Dense(64, 5)
        self.ActivationOutput = Activation_Softmax()
        
        #Loss
        self.true_values = np.array([0,0,0,0,0])
        self.Loss = Loss_CatergoricalCrossentropy()



        
 

    def run_network(self, inputs):
        self.inputs = inputs
        #INPUT layer -> Hidden Layer #1
        self.Layer1.forward(self.inputs)
        self.Activation1.forward(self.Layer1.output)

        #Layer 1--> Hidden Layer #2
        self.Layer2.forward(self.Activation1.output)
        self.Activation2.forward(self.Layer2.output)
        
        #Layer 2 --> Output Layer
        self.OutputLayer.forward(self.Activation2.output)
        self.ActivationOutput.forward(self.OutputLayer.output) #Softmax

        #Probabilities
        self.probabilities = self.ActivationOutput.output 
        
        #Loss
        loss_output = self.Loss.forward(self.probabilities, self.true_values) #Compare output to constant true values
        self.hasRun = True




    def save_model(self):
        np.save("trained_model/weights1.npy", self.Layer1.weights)
        np.save("trained_model/biases1.npy", self.Layer1.biases)

        np.save("trained_model/weights2.npy", self.Layer2.weights)
        np.save("trained_model/biases2.npy", self.Layer2.biases)

        np.save("trained_model/weightsOutput.npy", self.OutputLayer.weights)
        np.save("trained_model/biasesOutput.npy", self.OutputLayer.biases)


    def load_model(self):
        try:
            self.Layer1.weights = np.load("trained_model/weights1.npy")
            self.Layer1.biases = np.load("trained_model/biases1.npy")

            self.Layer2.weights = np.load("trained_model/weights2.npy")
            self.Layer2.biases = np.load("trained_model/biases2.npy")

            self.OutputLayer.weights = np.load("trained_model/weightsOutput.npy")
            self.OutputLayer.biases = np.load("trained_model/biasesOutput.npy")

            return 0

        except:
            return 1

    def propagate_backward(self):
        #Dvalues
        self.dvalues = self.Loss.backward(self.probabilities, self.true_values)

        self.OutputLayer.backward(self.dvalues, self.learning_rate) #Get dinputs for other layers


        #Optimise layer 2
        self.Layer2.backward(self.OutputLayer.dinputs, self.learning_rate)

        #Optimise layer 1
        self.Layer1.inputs = np.array(self.Layer1.inputs).reshape(1, -1)#Reshape input to work
        self.Layer1.backward(self.Layer2.dinputs,self.learning_rate)

    

