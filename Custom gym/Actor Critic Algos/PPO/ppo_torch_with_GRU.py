import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import pickle as pkl
#%%
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size) 
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]   #Creates random batch indices of size (batch_size) for every 'N' episodes

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,chkpt_dir,
            fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        
        self.hx = T.zeros(1, 256)
        
        self.conv1 = nn.Conv2d(input_dims[0], 5, 4, 1, 1)
        self.fc_input_dims = self.calc_conv_size(input_dims)
        
        self.gru_actor = nn.GRUCell(self.fc_input_dims, fc1_dims)
        
        self.actor = nn.Sequential(
                nn.Linear(fc1_dims, fc2_dims),
                nn.ELU(),
                nn.Linear(fc2_dims, n_actions),
                nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv_state = conv1.view(conv1.size()[0], -1)
        
        self.hx = self.gru_actor(conv_state, (self.hx))
        
        dist = self.actor(self.hx)
        dist = Categorical(dist)
        
        return dist
    
    def calc_conv_size(self, input_dims):
        state_buffer = T.zeros(1, *input_dims)
        dims = self.conv1(state_buffer)
        return int(np.prod(dims.size()))

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha,chkpt_dir, fc1_dims=256, fc2_dims=256,
            ):
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        
        self.hx = T.zeros(1, 256)
        self.conv1 = nn.Conv2d(*input_dims, 5, 4, 1, 1)
        self.fc_input_dims = self.calc_conv_size(input_dims)
        self.gru_critic = nn.GRUCell(self.fc_input_dims, fc1_dims)
        
        
        self.critic = nn.Sequential(
                nn.Linear(fc1_dims, fc2_dims),
                nn.ELU(),
                nn.Linear(fc2_dims, 1)
        )
    
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv_state = conv1.view(conv1.size()[0], -1)
        self.hx = self.gru_critic(conv_state, (self.hx))
        
        value = self.critic(self.hx)
        return value
    
    def  calc_conv_size(self, input_dims):
        state_buffer = T.zeros(1, *input_dims)
        dims = self.conv1(state_buffer)
        return int(np.prod(dims.size()))
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent:
    def __init__(self, n_actions, input_dims, model_file, pickle_file, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha, chkpt_dir=model_file)
        self.critic = CriticNetwork(input_dims, alpha, chkpt_dir= model_file)
        self.memory = PPOMemory(batch_size)
        
        self.pklFile = os.path.join(pickle_file,'pickle.pkl')
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        
        self.pklObj = open(self.pklFile, 'wb')
        pkl.dump(self.memory, self.pklObj)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        
        self.pklObj = open(self.pklFile, 'rb')
        self.memory = pkl.load(self.pklObj)

    def choose_action(self, observation):
        
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])  #Calculating advanteges for the timestep 't'
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()
                

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                
                self.actor.hx.detach()
                self.critic.hx.detach()
                
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()         