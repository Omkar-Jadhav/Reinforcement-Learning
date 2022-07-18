#%%
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
import numpy as np
from networks import PolicyGradientNetwork
import os
import pickle as pkl
#%%
class Agent:
    def __init__(self,filename, algo,env_name,n_actions, directory, alpha= 0.003, gamma = 0.99, fc1_dims =256, fc2_dims = 256):
        self.gamma = gamma
        self.alpha = alpha
        self.n_actions = n_actions
        self.fc2_dims = fc2_dims
        self.fc1_dims = fc1_dims
        
        self.state_memory =[]
        self.action_memory = []
        self.reward_memory = []
        self.policy = PolicyGradientNetwork(n_actions = self.n_actions)
        self.policy.compile(optimizer=Adam(learning_rate=self.alpha))
        
        self.filename = filename
        self.pklFile = os.path.join(directory + 'pickle/'+str(env_name)+'/','pickle.pkl')
        
    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype = tf.float32)
        probs = self.policy(state)
        
        action_probs = tfp.distributions.Categorical(probs = probs )       # Makes sort of onehot encoded object for the index of the  
                                                                            # probability https://www.youtube.com/watch?v=421uW9aZHio from 6:09
        action = action_probs.sample()
        
        return action.numpy()[0]
    
    def store_transition(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        
    def learn(self):
        actions = tf.convert_to_tensor(self.action_memory, dtype = tf.float32)
        rewards = tf.convert_to_tensor(self.reward_memory)
        
        G = np.zeros_like(rewards)
        
        for i in range(len(rewards)):
            G_sum = 0
            discount = 1
            for k in range(i, len(rewards)):
                G_sum += rewards[k]*discount
                discount = self.gamma*discount
                
            G[i] = G_sum
            
        with tf.GradientTape() as tape:
            loss = 0.0
            
            for idx, (g, state) in enumerate(zip(G, self.state_memory)):
                state = tf.convert_to_tensor([state], dtype= tf.float32)
                probs = self.policy.call(state)
                action_probs = tfp.distributions.Categorical(probs = probs)
                log_probs = action_probs.log_prob(actions[idx])                
                
                loss += -g *tf.squeeze(log_probs)
                
        gradient = tape.gradient(loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(gradient, self.policy.trainable_variables))
        
        self.action_memory = []
        self.state_memory = []
        self.reward_memory = []
        
    def save_model(self):
        self.policy.save(self.filename)
                
        self.pklObj = open(self.pklFile,'wb')
        print('... saving model...')
        
    def load_model(self):
        model = tf.keras.models.load_model(self.filename)
        
        self.pklObj = open(self.pklFile,'rb')
        print('... loading model...')
        
        return model
        