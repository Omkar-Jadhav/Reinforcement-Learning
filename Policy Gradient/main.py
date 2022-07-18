#%%
import gym
import numpy as np
from PL_agent import Agent
from utils import plotLearning
import os
import pickle as pkl
import tensorflow as tf
#%%
if __name__ == '__main__':
    print(tf.__version__)
    algo = 'Policy Gradient'
        
    env_name = 'LunarLander-v2'
    env = gym.make(env_name)
    score_history = []
    n_episodes = 20
    load_checkpoint = False
    train = True
    
    if(not os.path.exists(os.path.join('/models/',str(algo)))):
        os.makedirs(os.path.join('/models/',str(algo)+'/'))
    if(not os.path.exists(os.path.join('/plots/',str(algo)))):
        os.makedirs(os.path.join('/plots/',str(algo)+'/'))
    if(not os.path.exists(os.path.join('/pickle/',str(algo)))):
        os.makedirs(os.path.join('/pickle/',str(algo)+'/'))
    
    filename = '/models/'+str(algo)+'/'+'agent n_games' + str(n_episodes)
    
    agent = Agent(alpha= 0.0005, gamma = 0.99, n_actions= 4, filename= filename, algo = algo)
    
    prev_n_episode = 0
    best_score = -np.inf
    
    if load_checkpoint:
        agent.policy = agent.load_model
        score_history = pkl.load(agent.pklObj)
        prev_n_episode = pkl.load(agent.pklObj)
             
    for i in range(prev_n_episode, n_episodes):
        done = False
        score = 0
        observation = env.reset()
        
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward)
            
            observation = observation_
            score += reward
            
        score_history.append(score)
        if train:
            agent.learn()
        
        avg_score = np.mean(score_history[-20:])
        
        if train:
            if score >= best_score:
                agent.save_model()
                pkl.dump(score_history, agent.pklObj)
                pkl.dump(i, agent.pklObj)
                
                best_score = score
                
        
        print('Episode ',i, 'score %.1f ' %score, 'best score %.1f' %best_score,'avg_score %.1f' %avg_score )
        
    filename = 'lunar_lander.png'
    plotLearning(score_history, filename=filename, window = 20)
    