import numpy as np

class NeuralNet:
    # Initalizes the structure of Neural Network
    def __init__(self, nInput=2, nOutput=1, nHiddenLayers=1, nHiddenNodes=3, learningRate=0.01):

        self.n_h_layers = nHiddenLayers
        self.n_h_nodes = nHiddenNodes
        self.learningRate = learningRate

        # Weight initialization
        self.w_ih = np.random.randn(nInput, nHiddenNodes)
        #print(self.w_ih)
        self.w_ho = np.random.randn(nHiddenNodes, nOutput)
        #print(self.w_ho)
        self.w_h = [np.random.randn(nHiddenNodes, nHiddenNodes) for x in range(0, nHiddenLayers)]
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
        self.l_h.append(np.dot(X, self.w_ih)) # Initalize the first hidden layer
        self.l_h[0] = self.activation(self.l_h[0]) # Apply activation function

        for x in range(1, self.n_h_layers):
            self.l_h.append(np.dot(self.l_h[x-1], self.w_h[x-1])) # dot the previous hidden layer with the weight to get new layer
            self.l_h[x] = self.activation(self.l_h[x])  # apply activation funciton
        
        print("w_ho \n", self.w_ho)
        o = np.dot(self.l_h[self.n_h_layers - 1], self.w_ho) # Calc the output layer from last hidden layer
        o = self.activation(o) # Apply last activation function
        return o
    
    # Backprops comparing and readjusting to y
    def backward(self, X, y, o):
        o_error = y - o
        o_delta = o_error * self.activationPrime(o)
        print("o__delta \n", o_delta)
        print("l_h[n_h_layers].T dot o_delta \n",np.dot(self.l_h[self.n_h_layers - 1][:,None], o_delta))
        print("w_ho \n", self.w_ho)
        print("self.l_h[n_h_layers - 1] \n", self.l_h[self.n_h_layers - 1])
        print("self.l_h[n_h_layers - 1].T \n", self.l_h[self.n_h_layers - 1][:,None].shape)
        print(o_delta.shape)
        self.w_ho += np.dot(self.l_h[self.n_h_layers - 1][:,None], o_delta)[:,None]

        for x in reversed(range(self.n_h_layers)):
            print("x : ", x)
            if x == self.n_h_layers - 1:
                h_error = np.dot(o_error, self.w_ho.T)
                h_delta = h_error * self.activationPrime(self.l_h[x])

            h_error = np.dot(h_error, self.w_h[x].T)
            h_delta = h_error * self.activationPrime(self.l_h[x])
            self.w_h[x] += np.dot(self.l_h[x].T, h_delta)
        
        print("w_ih \n", self.w_ih)
        print("w_ih.T \n", self.w_ih.T)

        i_error = np.dot(h_error, self.w_h[0].T)
        i_delta = i_error * self.activationPrime(self.l_h[0])
        print("i_delta\n",i_delta)
        print("i_delta.T\n", i_delta.T)
        print("X \n", X)
        print("X.T \n", X[:,None])
        print("X.T dot i_delta \n", np.dot(np.asmatrix(X).T, np.asmatrix(i_delta)))
        self.w_ih += np.dot(X[:,None], i_delta)

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

nn = NeuralNet(nInput=2, nOutput=1, nHiddenLayers=3, nHiddenNodes=3)

#print(nn.forward(np.asarray([1, 2])))

nn.train(np.asarray([1, 2]), np.asarray([1]))