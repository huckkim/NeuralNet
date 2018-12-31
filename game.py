from array import array
import random

class Gameplay:
    def __init__ (self, nPlayers = 10, position=-1):
        self.nPlayers = nPlayers
        self.randomDeck = []
        self.playersCard = []
        self.position = random.randint(0, nPlayers-1) if position < 0 else position

    def start(self):
        self.playersCard[:] = []
        self.randomDeck = [x%13 for x in range(52)]
        random.shuffle(self.randomDeck)

        for x in range(self.nPlayers):
            self.playersCard.append(self.randomDeck.pop())
    
    def getInfo(self):
        return [self.nPlayers, self.position, self.playersCard[self.position]]
    
    def play(self, move):
        #print("before switching: ", self.playersCard)
        #print("dealer switch: ", self.randomDeck[0])
        for x in range(self.nPlayers):
            if x == self.position:
                choice = move # my move
            else:
                choice = random.randint(0,1) # bot move
            if choice == 1:
                if x == self.nPlayers - 1:
                    self.playersCard[x] = self.randomDeck[0]
                else:
                   if self.playersCard[x+1] != 12:
                       self.playersCard[x], self.playersCard[x+1] = self.playersCard[x+1], self.playersCard[x]
        #print("after switching: ", self.playersCard)
        minCard = min(self.playersCard)
        if self.playersCard[self.position] == minCard:
            return 0 # lose
        else:
            return 1 # win