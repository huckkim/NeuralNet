import nnet
import game
import numpy as np

game = game.Gameplay()
nn = nnet.NeuralNet(nInput=3, nOutput=1, nHiddenNodes=2, learningRate=0.01, nHiddenLayers=1)

win = 0
lose = 0
for x in range(0, 10000):
    info = game.start()
    print(np.asarray(info))
    outcome = game.play(nn.forward(np.asarray(info)))
    print(outcome)
    if outcome == 0:
        print("win")
        win += 1
    else:
        print("lose")
        lose += 1
    nn.train(np.asarray(info), np.asarray(outcome))
    
