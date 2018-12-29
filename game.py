from array import array
import random

class Gameplay:
    def __init__(self, numPlayers = 10):
        self.position = 0 # inits the players position to zero
        self.info = [] # inits the info tuple that is fed to to the AI
        self.deck = [] # inits the deck
        self.playersCard = [] # inits the players card
        self.numPlayers = numPlayers # saves the number of players 
        
    def start(self):
        self.deck = [i for i in range(52)] # init deck with 52 cards
        self.deck = sorted(self.deck, key=lambda k: random.random()) # 'shuffles the deck'
        #print(self.deck)
        self.position = random.randint(0, self.numPlayers-1) # assigns the player to a random spot
        #print(self.position)
        for x in range(0, self.numPlayers):
            self.playersCard.append(self.deck.pop() % 13) # assigns the  players their cards
        self.info = []
        #print(self.info)
        self.info.append(self.numPlayers) # adds the number of plyaers 
        self.info.append(self.playersCard[self.position] % 13) # adds what card the player has
        self.info.append(self.position) # adds his position on the table

        return self.info
    
    def play(self, move):
        #print(self.playersCard)
        minCard = min(self.playersCard) # saves the minimum card 
        for x in range(0, self.numPlayers):
            playMove = (move if x == self.position else random.randint(0,1)) # if it's the players turn use their move else use random move

            if x == self.numPlayers - 1: # if its the last person to go use special rules
                if playMove == 1: # if they choose to switch pull a random card from the deck
                    self.playersCard[x] = self.deck[0] % 13
            else:
                if playMove == 1:
                    self.playersCard[x] , self.playersCard[x+1] = self.playersCard[x+1], self.playersCard[x]
        #print(self.playersCard)
        if self.playersCard[self.position] == minCard:
            return True 
        else:
            return False 

