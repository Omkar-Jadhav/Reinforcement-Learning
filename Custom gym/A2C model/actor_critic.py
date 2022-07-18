import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pickle as pkl
#%%
class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, chkpt_dir, fc1_dims=256, fc2_dims=256):
        super(ActorCriticNetwork, self).__init__()
        
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_critic')
        
        self.hx = T.zeros(1, 256)
        
        self.conv1 = nn.Conv2d(input_dims[0], 5, 4, 1, 1)
        self.fc_input_dims = self.calc_conv_size(input_dims)
        
        self.gru = nn.GRUCell(self.fc_input_dims, fc1_dims)
        
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)
        self.v = nn.Linear(fc2_dims, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv_flatten = conv1.view(conv1.size()[0], -1)
        
        self.hx = self.gru(conv_flatten, (self.hx))
        
        x = F.relu(self.fc2(self.hx))
        pi = F.softmax(self.pi(x), dim = 1)
        v = self.v(x)

        return (pi, v)
    
    def calc_conv_size(self, input_dims):
        state_buffer = T.zeros(1, *input_dims)
        dims = self.conv1(state_buffer)
        return int(np.prod(dims.size()))
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent():
    def __init__(self, lr, pickle_file, input_dims, n_actions, chkpt_dir, fc1_dims = 256, fc2_dims = 256,
                 gamma=0.99):
        self.gamma = gamma
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.actor_critic = ActorCriticNetwork(lr, input_dims, n_actions, chkpt_dir, fc1_dims=256, fc2_dims=256)
        self.log_prob = None
        
        self.pklFile = os.path.join(pickle_file,'pickle.pkl')

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor_critic.device)
        probabilities, _ = self.actor_critic.forward(state)
        probabilities = F.softmax(probabilities, dim=1)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        self.log_prob = log_prob

        return action.item()

    def learn(self, state, reward, state_, done):
        self.actor_critic.optimizer.zero_grad()

        state = T.tensor([state], dtype=T.float).to(self.actor_critic.device)
        state_ = T.tensor([state_], dtype=T.float).to(self.actor_critic.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actor_critic.device)

        _, critic_value = self.actor_critic.forward(state)
        _, critic_value_ = self.actor_critic.forward(state_)

        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        actor_loss = -self.log_prob*delta
        critic_loss = delta**2
        
        self.actor_critic.hx.detach()

        (actor_loss + critic_loss).backward()
        self.actor_critic.optimizer.step()
        
    def save_models(self):
        print('... saving models ...')
        self.actor_critic.save_checkpoint()
                
        self.pklObj = open(self.pklFile, 'wb')

    def load_models(self):
        print('... loading models ...')
        self.actor_critic.load_checkpoint()
                
        self.pklObj = open(self.pklFile, 'rb')