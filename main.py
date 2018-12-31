import nnet as nn
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

game = game.Gameplay(nPlayers=2)
nn = nn.NeuralNet(nInput=3, nOutput=1, nHiddenNodes=2, learningRate=0.5 )

win = 0
lose = 0
for x in range(0, 600000):
    game.start()
    info = game.getInfo()
    #print(info)
    move = nn.forward(np.asarray(info))
    if move > 0.5:
        move = 1
    else:
        move = 0

    outcome = game.play(move)

    if outcome == 1:
        #print("win")
        win += 1
        nn.train(np.asarray(info), np.asarray(move))
    else:
        #print("lose")
        lose += 1
        nn.train(np.asarray(info), np.asarray(1 - move))

print(nn.forward(np.asarray([2, 0, 0])))
print(nn.forward(np.asarray([2, 0, 1])))
print(nn.forward(np.asarray([2, 0, 2])))
print(nn.forward(np.asarray([2, 0, 3])))
print(nn.forward(np.asarray([2, 0, 4])))
print(nn.forward(np.asarray([2, 0, 5])))
print(nn.forward(np.asarray([2, 0, 6])))
print(nn.forward(np.asarray([2, 0, 7])))
print(nn.forward(np.asarray([2, 0, 8])))
print(nn.forward(np.asarray([2, 0, 9])))
print(nn.forward(np.asarray([2, 0, 10])))
print(nn.forward(np.asarray([2, 0, 11])))
print(nn.forward(np.asarray([2, 0, 12])))

print("dealer")
print(nn.forward(np.asarray([2, 1, 0])))
print(nn.forward(np.asarray([2, 1, 1])))
print(nn.forward(np.asarray([2, 1, 2])))
print(nn.forward(np.asarray([2, 1, 3])))
print(nn.forward(np.asarray([2, 1, 4])))
print(nn.forward(np.asarray([2, 1, 5])))
print(nn.forward(np.asarray([2, 1, 6])))
print(nn.forward(np.asarray([2, 1, 7])))
print(nn.forward(np.asarray([2, 1, 8])))
print(nn.forward(np.asarray([2, 1, 9])))
print(nn.forward(np.asarray([2, 1, 10])))
print(nn.forward(np.asarray([2, 1, 11])))
print(nn.forward(np.asarray([2, 1, 12])))

print(win)
print(lose)