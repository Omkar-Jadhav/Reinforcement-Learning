import matplotlib.pyplot as plt 
import numpy as np

def plotLearning(scores, filename, window=5):   
    N = len(scores)
    
    x = [i for i in range(N)]
    running_avg = np.empty(N)
    
    for i in range(N):
        running_avg[i] = np.mean(scores[max(0,i-window) : (i+1)])        
        
    plt.ylabel('Score')       
    plt.xlabel('Game')                     
    plt.plot(x, running_avg)
    plt.savefig(filename)