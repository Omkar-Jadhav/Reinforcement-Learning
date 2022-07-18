import gym
import collections
import numpy as np
from ppo_torch import *
from utils import plotLearning

if __name__ == '__main__':
    env =gym.make('CartPole-v0')
    N=20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions= env.action_space.n, batch_size = batch_size,
                  alpha = alpha, n_epochs=n_epochs, 
                  input_dims= env.observation_space.shape)
    
    n_games = 300
    
    figure_file = 'plots/Cartpole.png'
    
    best_score = env.reward_range[0]
    score_history = []
    
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    
    load_checkpoint = False
    
    if load_checkpoint:
        agent.load_models()
    
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            
            observation_, reward, done, info = env.step(action)
            score += reward
            
            n_steps+=1
            
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N ==0:
                if not load_checkpoint:
                    agent.learn()
                    learn_iters += 1
                
            observation = observation_
            
        score_history.append(score)
        
        avg_score = np.mean(score_history[-20:])
            
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
                
        print('Episode ',i , 'Score %.1f' %score, 'Avg score %.1f' %avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
            
    x = [i+1 for i in range(len(score_history))]
    plotLearning(x= x,scores= score_history, filename= figure_file, window=20)
    