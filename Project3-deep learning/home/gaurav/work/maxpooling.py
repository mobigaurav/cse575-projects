import numpy as np 
class Maxpooling2D:
    # Initialization of MaxPooling layer
    def __init__(self, pool_size, stride, name):
        self.pool = pool_size
        self.s = stride
        self.name = name
    
    # Forward propagation
    def forward(self, inputs):
        self.inputs = inputs
        C, W, H = inputs.shape
        new_width = (W - self.pool)//self.s + 1
        new_height = (H - self.pool)//self.s + 1
        out = np.zeros((C, new_width, new_height))
        for c in range(C):
            for w in range(W//self.s):
                for h in range(H//self.s):
                    out[c, w, h] = np.max(self.inputs[c, w*self.s:w*self.s+self.pool, h*self.s:h*self.s+self.pool])
        return out
    
    # Backward propagation
    def backward(self, dy):
        C, W, H = self.inputs.shape
        dx = np.zeros(self.inputs.shape)
        
        for c in range(C):
            for w in range(0, W, self.pool):
                for h in range(0, H, self.pool):
                    st = np.argmax(self.inputs[c,w:w+self.pool,h:h+self.pool])
                    (idx, idy) = np.unravel_index(st, (self.pool, self.pool))
                    dx[c, w+idx, h+idy] = dy[c, w//self.pool, h//self.pool]
        return dx
    
    # No weights and bias for pooling layer to store
    def extract(self):
        return 