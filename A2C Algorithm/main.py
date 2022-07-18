#%%
import gym
import numpy as np
from actor_critic import Agent
from utils import plotLearning
import os
#%%
if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    agent = Agent(alpha=1e-5, n_actions= env.action_space.n)
    file_name = 'cartpole.png'
    figure_file = 'plots/' + file_name
    
    n_episodes = 1500
    
    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False
    
    if load_checkpoint:
        agent.load_models()

    for i in range(n_episodes):
        observation = env.reset()
        done = False
        score = 0
        
        while not done:
            action = agent.choose_action(observation)
            observation_, reward,done, info = env.step(action)
            score += reward
            if not load_checkpoint:
                agent.learn(observation, reward, observation_, done)
                
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-20:])
        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
            
        print('Episode ', i, 'Score %.1f' %score, 'avg_score %.1f' %avg_score)
    x = [i+1 for i in range(n_episodes)]
    plotLearning(x= x,scores= score_history, filename= figure_file, window=20)
        