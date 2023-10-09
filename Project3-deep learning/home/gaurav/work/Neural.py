from convolution import Convolution2D
from maxpooling import Maxpooling2D
from fullyconnected import FullyConnected, Flatten
from activation import ReLu, Softmax
import numpy as np 

### Cross-Entropy Loss function
def cross_entropy(inputs, labels):
    out_num = labels.shape[0]
    p = np.sum(labels.reshape(1,out_num)*inputs)
    loss = -np.log(p)
    return loss

"""
This step shows how to define a simple CNN with all kind of layers which we introduced above.
"""
class Net:
    def __init__(self):
        # input: 28x28
        # output: 1x4 (only a subset, containing 4 classes, of the MNIST will be used)
        # conv1:  {(28-5+0x0)/2+1} -> (12x12x6) (output size of convolutional layer)
        # maxpool2: {(12-2)/2+1} -> (6x6)x6 (output size of pooling layer)
        # fc3: 216 -> 32
        # fc4: 32 -> 4
        # softmax: 4 -> 4
        lr = 0.001
        self.layers = []
        self.layers.append(Convolution2D(inputs_channel=1, num_filters=6, kernel_size=5, padding=0, stride=2, learning_rate=lr, name='conv1'))
        self.layers.append(ReLu())
        self.layers.append(Maxpooling2D(pool_size=2, stride=2, name='maxpool2'))
        self.layers.append(Flatten())
        self.layers.append(FullyConnected(num_inputs=6*6*6, num_outputs=32, learning_rate=lr, name='fc3'))
        self.layers.append(ReLu())
        self.layers.append(FullyConnected(num_inputs=32, num_outputs=4, learning_rate=lr, name='fc4'))
        self.layers.append(Softmax())
        self.lay_num = len(self.layers)
    
    ### Function for train the network
    def train(self, data, label):
        batch_size = data.shape[0]
        loss = 0
        acc = 0
        for b in range(batch_size):
            x = data[b]
            y = label[b]
            # forward pass
            for l in range(self.lay_num):
                output = self.layers[l].forward(x)
                x = output
            loss += cross_entropy(output, y)
            if np.argmax(output) == np.argmax(y):
                acc += 1
            # backward pass
            dy = y
            for l in range(self.lay_num-1, -1, -1):
                dout = self.layers[l].backward(dy)
                dy = dout
        return loss, acc