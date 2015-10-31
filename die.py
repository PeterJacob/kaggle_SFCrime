import numpy as np
import random

def play_game():
    last_digit = random.randint(1,7)
    while True:
        current_digit = random.randint(1,7)
        
        if last_digit==3 and current_digit==6:
            return 1
        
        if last_digit==4 and current_digit==4:
            return 2
        
        if last_digit==6 and current_digit==1:
            return 3

winner = []
for i in range(100):
    print i
    winner.append(play_game())

winner = np.array(winner)
print("Art: {}".format(sum(winner==1)))
print("Kel: {}".format(sum(winner==2)))
print("Marvin: {}".format(sum(winner==3)))
