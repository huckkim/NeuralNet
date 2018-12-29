import numpy as np

class NeuralNet:
    # Initalizes the structure of Neural Network
    def __init__(self, nInput=2, nOutput=1, nHiddenLayers=1, nHiddenNodes=3, learningRate=0.01):

        self.n_h_layers = nHiddenLayers
        self.n_h_nodes = nHiddenNodes
        self.learningRate = learningRate

        # Weight initialization
        self.w_ih = np.random.randn(nInput + 1, nHiddenNodes)
        #print(self.w_ih)
        self.w_ho = np.random.randn(nHiddenNodes+1, nOutput)
        #print(self.w_ho)
        self.w_h = [np.random.randn(nHiddenNodes+1, nHiddenNodes) for x in range(0, nHiddenLayers)]
        #print(self.w_h[0])
        self.l_h = []

    # Activation function
    def activation(self, x):
        return 1/(1+np.exp(-x))
    
    # Derivative of activation function
    def activationPrime(self, x):
        return x*(1-x) 

    def error(self, y, o):
        return np.power(y-o, 2) / 2

    # Gives the predicted output
    def forward(self, X):
        self.l_h.append(np.dot(np.append(X, 1), self.w_ih)) # Initalize the first hidden layer
        self.l_h[0] = self.activation(self.l_h[0]) # Apply activation function
        self.l_h[0] = np.append(self.l_h[0], 1)
        
        # Fix for biases cause they don't work
        for x in range(1, self.n_h_layers):
            self.l_h.append(np.dot(self.l_h[x-1], self.w_h[x-1])) # dot the previous hidden layer with the weight to get new layer
            self.l_h[x] = self.activation(self.l_h[x])  # apply activation funciton
            self.l_h[x] = np.append(self.l_h[x], 1) # add 1 for the bias

        o = np.dot(self.l_h[self.n_h_layers - 1], self.w_ho) # Calc the output layer from last hidden layer
        #print(self.l_h[self.n_h_layers - 1])
        #print(self.w_ho)
        #print(o)
        o = self.activation(o) # Apply last activation function
        return o
    
    # Backprops comparing and readjusting to y
    def backward(self, X, y, o):
        delta = []
        d_error = (o - y)
        d_out = self.activationPrime(o) # 
        delta.append(self.l_h[self.n_h_layers - 1][:-1][:,None] * (d_error * d_out))
        #self.w_ho[:-1] -= delta[0] * self.learningRate
        #print(self.w_ho)
        for x in reversed(range(self.n_h_layers)):
            if x == 0:
                d_error = d_error * d_out
                d_error = np.dot(d_error, self.w_ho[:-1].T)
                d_out = self.activationPrime(self.l_h[0])[:-1]
                delta.append(X[:,None] * (d_error*d_out))
        self.w_ho[:-1] -= delta[0] * self.learningRate
        print(self.w_ho)
        self.w_ih[:-1] -= delta[1] * self.learningRate
        print(self.w_ih)

    def train(self, X, y):
        o = self.forward(X)
        print("Total error = ", np.sum(self.error(y, o)))
        self.backward(X, y, o)

# ["Card Number", "Location", "Number of Players"]
nn = NeuralNet(nInput=3, nOutput=1, nHiddenNodes=2, learningRate=0.01)

print(nn.forward(np.asarray([2, 3, 5])))