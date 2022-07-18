#%%
import re
import gym
from gym import spaces
import numpy as np
import pandas as pd


max_account_balance = 1000000
max_Volume = 100000000
max_share_price = 10000
max_num_shares = 100000
max_steps = 2000
initial_account_balance = 10000

class CustomContinuousEnv(gym.Env):
    '''
    observation space -
    [['Open' for currrent step and prev 5 timesteps],
    ['High' for currrent step and prev 5 timesteps],
    ['Low' for currrent step and prev 5 timesteps],
    ['Close' for currrent step and prev 5 timesteps],
    [1. account balnce, 2.max_net_worth, 3. Total shares held, 
    4.Avg cost, 5, total share sold, 6. Total sold value ]]
    
    action_space - [[Buy, sell, hold], [% Amount]]
    
    reward - current_balance * (current time step/ max time step)
    '''
    metadata = {'render.modes': ['human']}
    def __init__(self, df):
        super().__init__()
        
        self.df = df
        self.reward_range = (0, max_account_balance)
        
        #Actions to be performed x% Buy, x% sell, x% hold
        # index - 0 is the different actions
        # index - 1 is the % buy/sell/hold
        self.action_space = spaces.Box(Low = np.array([0,0]), High = np.array([3,1]))
        
        
        #Observation space contains OHLC, current balance, stock position
        self.observation_space = spaces.Box(Low = 0, High = 1, shape=(6,6))
        
    def _next_observation(self):
        #Selects random point in the dataset, and take its last six time steps, and appends it with account information
        frame = np.array([
            self.df.loc[self.current_step: self.current_step+5, 'Open'].values / max_share_price,
        
            self.df.loc[self.current_step: self.current_step+5, 'High']/ max_share_price,
            
            self.df.loc[self.current_step: self.current_step+5, 'Low']/ max_share_price,
            
            self.df.loc[self.current_step: self.current_step+5, 'Close']/ max_share_price,
            
            self.df.loc[self.current_step: self.current_step+5, 'Volume ']/ max_Volume,
        ])
        
        #Appending the additional data of the user
        
        obs = np.append(frame,
                        [[
                            self.balance/ max_account_balance,
                            self.max_net_worth /max_account_balance,
                            self.shares_held / max_num_shares,
                            self.avg_cost/max_share_price,
                            self.total_shares_sold / max_num_shares,
                            # self.total_shares_bought / max_num_shares,
                            self.total_sales_value / (max_num_shares * max_share_price),
                            # self.total_buy_value / (max_num_shares * max_share_price)
                        ]], axis = 0)
        
        return obs
    
    def _take_action(self, action):
        # Set the current price to a random price 
        current_price = np.random.uniform(self.df.loc[self.current_step,'Open'], self.df.loc[self.current_step, 'Close'])
        
        action_type = int(action[0])
        amount = action[1]
        
        if action_type ==0:
            # Buy %amount of balance in shares
            total_possible = int(self.balance/ current_price)
            shares_bought = int(total_possible * amount)
            self.total_shares_bought += shares_bought
            self.total_buy_value += shares_bought * current_price
            prev_cost = self.avg_cost * self.shares_held
            
            additional_cost = shares_bought * current_price
            
            self.balance -= additional_cost
            self.avg_cost = (prev_cost + additional_cost)/ (self.shares_held + shares_bought)
            
            self.shares_held += shares_bought
            
        if action_type == 1:
            # sell amount % shares held
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold* current_price
            
        self.net_worth = self.balance + self.shares_held * current_price 
        
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
            
        if self.shares_held == 0:
            self.avg_cost = 0
            
    def step(self, action):
        #Excecute one time step within the envirnoment
        self._take_action(action)
        
        self.current_step += 1
         
        # If timestep comes to the end of the workspace
        if self.current_step > len(self.df.loc[:, 'Open'].values) - 6:
            self.current_step = 0
        
        # To take the of the long term trading and not take risky short term bets
        delay_modifier = (self.current_step / max_steps)
        reward = self.balance * delay_modifier
        
        done = (self.net_worth <=0)
        
        obs = self._next_observation()
        
        return obs, reward, done, {}
    
    def reset(self):
        # Reset the state of the envirnoment to an initial state
        self.balance = initial_account_balance
        self.net_worth = initial_account_balance
        self.max_net_worth = initial_account_balance
        
        self.shares_held = 0
        self.avg_cost = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.total_shares_bought = 0
        self.total_buy_value = 0
        
        #set the current step to random point in the data frame
        self.current_step = np.random.randint(0, len(self.df.loc[:, 'Open'].values) -6)
        
        return self._next_observation()
    
    def render(self, mode = 'human', Close =  False):
        #rendering the envirnoment
        profit = self.net_worth - initial_account_balance
        
        print(f'Step : {self.current_step}')
        print(f'Balance : {self.balance}')
        print(f'Shares held {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(f'Avg cost for held shares: {self.avg_cost} (Total sales values: {self.total_sales_value})')
        print(f'Networth: {self.net_worth} (Max net worth : {self.max_net_worth})')
        print(f'Profit: {profit}')
        
        
        

#%% Discrete action space envirnoment
max_account_balance = 1000000
max_Volume = 100000000
max_share_price = 10000
max_num_shares = 100000
max_steps = 2000
initial_account_balance = 10000

class CustomDiscreteEnv(gym.Env):
    '''
    observation space -
    [['Open' for currrent step and prev 5 timesteps],
    ['High' for currrent step and prev 5 timesteps],
    ['Low' for currrent step and prev 5 timesteps],
    ['Close' for currrent step and prev 5 timesteps],
    [1. account balnce, 2.max_net_worth, 3. Total shares held, 
    4.Avg cost, 5, total share sold, 6. Total sold value ]]
    
    action_space - [Buy, sell, hold]
    
    reward - profit / loss
    '''
    metadata = {'render.modes': ['human']}
    def __init__(self, df):
        super().__init__()
        
        self.df = df
        self.reward_range = (0, max_account_balance)
        
        self.action_cntr = 0 
        self.short_trade = False
        
        #Actions to be performed x% Buy, x% sell, x% hold
        # index - 0 is the different actions
        # index - 1 is the % buy/sell/hold
        self.action_space = spaces.Discrete(3)
        
        
        #Observation space contains OHLC, current balance, stock position
        self.observation_space = spaces.Box(low = 0, high = 1, shape=(6,6))
        
    def _next_observation(self):
        #Selects random point in the dataset, and take its last six time steps, and appends it with account information
        frame = np.array([
            self.df.loc[self.current_step: self.current_step+5, 'Open'].values / max_share_price,
        
            self.df.loc[self.current_step: self.current_step+5, 'High'].values/ max_share_price,
            
            self.df.loc[self.current_step: self.current_step+5, 'Low'].values/ max_share_price,
            
            self.df.loc[self.current_step: self.current_step+5, 'Close'].values/ max_share_price,
            
            self.df.loc[self.current_step: self.current_step+5, 'Volume'].values/ max_Volume,
        ])
        
        #Appending the additional data of the user
        
        # Set the current price to a random price 
        self.current_price = np.random.uniform(self.df.loc[self.current_step,'Open'], self.df.loc[self.current_step, 'Close'])
        
        obs = np.append(frame,
                        [[
                            self.shares_held /max_num_shares,           #If share held < 0 and action_cntr%2 != 0 then its a short position therefore agent needs to learn next action should be a buy at appropriate price              
                            self.action_cntr/2,
                            self.current_price / max_share_price,        
                            0,
                            0,
                            0       
                        ]], axis = 0)
        
        obs = obs.reshape(1, obs.shape[0], obs.shape[1])
        
        return obs
    
    def _take_action(self, action):
               
        
        if action ==0:
            # Buy %amount of balance in shares
            if(self.balance >= self.current_price):
                total_possible = int(self.balance/ self.current_price) if not self.short_trade else int(-self.shares_held)
                self.shares_bought = total_possible
                buy_cost = self.shares_bought * self.current_price
                
                self.total_shares_bought += self.shares_bought
                self.total_buy_value += self.shares_bought * self.current_price
                     
                self.balance -= buy_cost
           
                self.shares_held += self.shares_bought
                
                self.action_cntr += 1
            
        if action == 1:
            # sell amount % shares held
            if self.shares_held >= 0:
                total_possible = int(self.balance/ self.current_price)
                self.shares_sold = int(self.shares_held) if (self.shares_held>0) else total_possible        #Condition to check the short selling opportunity
                 
                self.balance += self.shares_sold * self.current_price
                # For a short trade 
                if(self.shares_held == 0):
                    self.short_trade= True
                self.shares_held -= self.shares_sold
                
                self.total_shares_sold += self.shares_sold
                self.total_sales_value += self.shares_sold* self.current_price
                
                self.action_cntr += 1
            
            
        if action == 2:
            #Hold i.e. do nothing
            pass
                        
        self.net_worth = self.balance + self.shares_held * self.current_price 
        
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
        
            
    def step(self, action):
        #Excecute one time step within the envirnoment
        self._take_action(action)
        
        self.current_step += 1
         
        # If timestep comes to the end of the workspace
        if self.current_step > len(self.df.loc[:, 'Open'].values) - 6:
            self.current_step = 0
        
        # To take the of the long term trading and not take risky short term bets
        # delay_modifier = (self.current_step / max_steps)
        reward = 0.0
        if (self.action_cntr %2 ==0):
            reward = (self.balance - self.prev_balance)  #if not self.short_trade else (self.prev_balance -self.balance)
            self.short_trade = False # reseting the short trade flag if triggered in the short trade
            self.prev_balance = self.balance
        
        done = (self.net_worth <=0)
        
        obs = self._next_observation()
        
        return obs, reward, done, {}
    
    def reset(self):
        # Reset the state of the envirnoment to an initial state
        self.balance = initial_account_balance
        self.prev_balance = initial_account_balance
        self.net_worth = initial_account_balance
        self.max_net_worth = initial_account_balance
        
        self.shares_held = 0
        self.avg_cost = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.total_shares_bought = 0
        self.total_buy_value = 0
        self.shares_sold = 0
        self.shares_bought = 0
        self.action_cntr = 0
        #set the current step to random point in the data frame
        self.current_step = np.random.randint(0, len(self.df.loc[:, 'Open'].values) -6)
        
        # self.prev_action_price = np.random.randint(self.df.loc[self.current_step, 'Open'].values, self.df.loc[self.current_step, 'Close'].values)
        
        return self._next_observation()
    
    def render(self, mode = 'human', Close =  False):
        #rendering the envirnoment
        profit = self.net_worth - initial_account_balance
        
        print(f'Step : {self.current_step}')
        print(f'Balance : {self.balance}')
        print(f'Shares held {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(f'Avg cost for held shares: {self.avg_cost} (Total sales values: {self.total_sales_value})')
        print(f'Networth: {self.net_worth} (Max net worth : {self.max_net_worth})')
        print(f'Profit: {profit}')


#%% Testing the envirnoment       
# import pandas as pd
# df = pd.read_csv('D:\Reinforcement Learning\My codes\Custom gym/AAPL.csv')
# test = CustomDiscreteEnv(df)

# test.reset()
# for i in range(500):
#     action= np.random.randint(0,3)
#     obs, reward, done,_  = test.step(action)
#     print('reward :' + str(reward) +' action cntr: '+ str(test.action_cntr) )
#     # print(obs[0, 5,:])