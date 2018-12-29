from array import array
import random

class Gameplay:
    def __init__(self, numPlayers = 10):
        self.position = 0
        self.info = []
        self.numPlayers = numPlayers
        self.deck = []
        self.playersCard = [] # inits the players card
        
    def start(self):
        self.deck = [i for i in range(52)] # init deck with 52 cards
        self.deck = sorted(self.deck, key=lambda k: random.random()) # 'shuffles the deck'
        self.position = random.randint(0, self.numPlayers-1) # assigns the player to a random spot
        for x in range(0, self.numPlayers):
            self.playersCard.append(self.deck.pop() % 13) # assigns the  players their cards

        self.info.append(self.numPlayers)
        self.info.append(self.playersCard[self.position] % 13)
        self.info.append(self.position)

        return self.info
    
    def play(self, move):
        minCard = min(self.playersCard)
        #print(minCard)
        for x in range(0, self.numPlayers):
            if x == 0:
                playMove = move
            else:
                playMove = random.randint(0,1)
            if x == self.numPlayers - 1:
                if playMove == 1:
                    self.playersCard[x] = self.deck[0]
                if self.playersCard[x] == minCard:
                        return 0
            else:
                if playMove == 1:
                    self.playersCard[x] , self.playersCard[x+1] = self.playersCard[x+1], self.playersCard[x]
        return 1
        
#game = Gameplay()
#print("numPlayer, player card, position")
#print(game.start())
#print(game.play(1))