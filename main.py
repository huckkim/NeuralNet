import nnet
import game
import numpy as np

X = [[0, 0],
        [0, 1],
        [1, 1],
        [1, 0]]
y = [[0],
     [1],
     [0],
     [1]]

game = game.Gameplay()
nn = nnet.NeuralNet(nInput=2, nOutput=1, nHiddenNodes=2, learningRate=0.01)

for x in range(0, 10000):
    print("X:", X[x%4])
    print("y:", y[x%4])
    nn.train(np.asarray(X[x%4]), np.asarray(y[x%4]))

#win = 0
#lose = 0
#for x in range(0, 100000):
#    info = game.start()
#    if nn.forward(np.asarray(info)) > 0.5:
#        outcome = game.play(1)
#    else:
#        outcome = 1
#    print(outcome)
#    if outcome == 0:
#        print("win")
#        win += 1
#    else:
#        print("lose")
#        lose += 1
#        print(info)
#    nn.train(np.asarray(info), np.asarray(int(outcome)))
#
#print(win)
#print(lose)