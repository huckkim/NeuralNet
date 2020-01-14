# NeuralNet
Basic neural network made from scratch in python

## Goal
The goal of this project was to try making a neural net from scratch to predict the optimal moves for a very simple card game. It would go through multiple learning cycles to create a list of parameters on when to "pass" the ace. It would give percentage for each corresponding card on when to pass. 
## Rules of the Game
The rules of the game are simple in which there are a certain number of players and each person draws a card. They can choose to either keep their card or to switch with the person in front of them. The person in front must trade unless they have a king in which they show the king and the does not have to switch. At the end of the round the person with the lowest card loses.
## How to use the AI
Simply adjust the main file where it states
```python
game = game.Gameplay(nPlayers=2)
nn = nn.NeuralNet(nInput=3, nOutput=1, nHiddenNodes=18, learningRate=1 )
```
Where nPlayers controls the number of total players including the dealer. The second line controls the neural net settings.
## Some Results
Ace      [0.75833777] \
Two      [0.74612194] \
Three    [0.67557365] \
Four     [0.61103844] \
Five     [0.56212891] \
Six      [0.52811953] \
Seven    [0.50562498] \
Eight    [0.49118699] \
Nine     [0.48208651] \
Ten      [0.47641343] \
Jack     [0.47290192] \
Queen    [0.47074067] \
King     [0.4694212] 

Dealer                \
Ace      [0.6590281]  \
Two      [0.61276124] \
Three:   [0.56244855] \
Four     [0.5274118]  \
Five     [0.50474613] \
Six      [0.49045934] \
Seven    [0.48156395] \
Eight    [0.47606147] \
Nine     [0.47267024] \
Ten      [0.47058466] \
Jack     [0.46930367] \
Queen    [0.46851752] \
King     [0.46803535] \
338241                \
261759               
