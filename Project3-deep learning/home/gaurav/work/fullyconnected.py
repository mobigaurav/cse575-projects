import numpy as np 
class FullyConnected:
    # Initialization of Fully-Connected Layer
    def __init__(self, num_inputs, num_outputs, learning_rate, name):
        self.weights = 0.01*np.random.rand(num_inputs, num_outputs)
        self.bias = np.zeros((num_outputs, 1))
        self.lr = learning_rate
        self.name = name
    
    # Forward Propagation
    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(self.inputs, self.weights) + self.bias.T
    
    # Backward Propagation
    def backward(self, dy):

        if dy.shape[0] == self.inputs.shape[0]:
            dy = dy.T
        dw = dy.dot(self.inputs)
        db = np.sum(dy, axis=1, keepdims=True)
        dx = np.dot(dy.T, self.weights.T)

        self.weights -= self.lr * dw.T
        self.bias -= self.lr * db

        return dx
    
    # Extract weights and bias for storage
    def extract(self):
        return {self.name+'.weights':self.weights, self.name+'.bias':self.bias}
    
    # Feed the pretrained weights and bias for models 
    def feed(self, weights, bias):
        self.weights = weights
        self.bias = bias

### Flatten function to convert 4D feature maps into 3D feature vectors
class Flatten:
    def __init__(self):
        pass
    def forward(self, inputs):
        self.C, self.W, self.H = inputs.shape
        return inputs.reshape(1, self.C*self.W*self.H)
    def backward(self, dy):
        return dy.reshape(self.C, self.W, self.H)
    def extract(self):
        return