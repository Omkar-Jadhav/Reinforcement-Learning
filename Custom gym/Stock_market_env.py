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
max_steps = 500
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
        super(CustomContinuousEnv, self).__init__()
        
        self.df = df
        self.reward_range = (0, max_account_balance)
        self.first_step =0
        
        #Actions to be performed x% Buy, x% sell, x% hold
        # index - 0 is the different actions
        # index - 1 is the % buy/sell/hold
        self.action_space = spaces.Box(low = np.array([0,0]), high = np.array([3,1]))
        
        
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
        
        obs = np.append(frame,
                        [[
                            self.balance/ max_account_balance,
                            self.max_net_worth /max_account_balance,
                            self.shares_held / max_num_shares,
                            self.avg_cost/max_share_price,
                            self.total_shares_sold / max_num_shares,
                            self.total_sales_value / (max_num_shares * max_share_price)
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
            if(shares_bought>0):
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
            if(shares_sold>0):
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
         
        
        # To take the of the long term trading and not take risky short term bets
        delay_modifier = (self.current_step / max_steps)
        reward = self.balance * delay_modifier
        
        done = (self.net_worth <=0)
        
        # If timestep comes to the end of the workspace
        if self.current_step > len(self.df.loc[:, 'Open'].values) - 6:
            self.current_step -=1
            done = True
        
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
        self.first_step = self.current_step
        
        return self._next_observation()
    
    def render(self, mode = 'human', Close =  False):
        #rendering the envirnoment
        profit = self.net_worth - initial_account_balance
        
        print(f'Step : {self.current_step}')
        print(f'Balance : {self.balance}')
        # print(f'Shares held {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(f'Avg cost for held shares: {self.avg_cost} (Total sales values: {self.total_sales_value})')
        print(f'Networth: {self.net_worth} (Max net worth : {self.max_net_worth})')
        print(f'Profit: {profit}')
        
           
        
        

#%% Discrete action space envirnoment
import matplotlib.pyplot as plt
max_account_balance = 1000000
max_Volume = 100000000
max_share_price = 10000
max_num_shares = 100000
max_steps = 2000
initial_account_balance = 100000

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
        self.long_trade  = False
        self.first_step =0
        
        self._position_history = [0]*len(df.loc[:, 'Close'].values)
        self._trade_completion_history = [0]*len(df.loc[:, 'Close'].values)
        
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
               
        
        if action ==1:
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
                
                if(self.action_cntr%2!=0):
                    self.long_trade = True
                    self.short_trade = False
                    self.position = 'Long'
                    self._position_history[self.current_step] = self.position
            
        if action == 2:
            # sell amount % shares held
            if self.shares_held >= 0:
                total_possible = int(self.balance/ self.current_price)
                self.shares_sold = int(self.shares_held) if (self.shares_held>0) else total_possible        #Condition to check the short selling opportunity
                 
                self.balance += self.shares_sold * self.current_price
                
                self.shares_held -= self.shares_sold
                
                self.total_shares_sold += self.shares_sold
                self.total_sales_value += self.shares_sold* self.current_price
                
                self.action_cntr += 1
                # For a short trade 
                if(self.action_cntr%2 != 0):
                    self.short_trade= True
                    self.long_trade = False
                    self.position = 'Short'
                    self._position_history[self.current_step] = self.position
            
            
        if action == 0:
            #Hold i.e. do nothing
            self.position = 'Hold'
            self._position_history[self.current_step] = self.position
            
                        
        self.net_worth = self.balance + self.shares_held * self.current_price 
        
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
        
            
    def step(self, action):
        #Excecute one time step within the envirnoment
        self.current_step += 1  #For taking price of next time step
        self._take_action(action)
        
        
         
        
        # To take the of the long term trading and not take risky short term bets
        # delay_modifier = (self.current_step / max_steps)
        self.reward = 0
        if (self.action_cntr %2 ==0):
            self.reward = (self.balance - self.prev_balance)  #if not self.short_trade else (self.prev_balance -self.balance)
            self.short_trade = False # reseting the short trade flag if triggered in the short trade
            self.prev_balance = self.balance
            self.position = 'Squared-off'
            # self._position_history.append(self.position)        
            
        if(self.position ==  'Squared-off'):
            self._trade_completion_history[self.current_step] = 1
            self.position = "_"
        
        done = (self.net_worth <=0)
        
        # If timestep comes to the end of the workspace
        if self.current_step == len(self.df.loc[:, 'Open'].values) - 6:
            self.current_step -= 1
            done = True
        
        obs = self._next_observation() 
        
        return obs, self.reward, done, {}
    
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
        self.first_step = self.current_step
        
        
        # self.prev_action_price = np.random.randint(self.df.loc[self.current_step, 'Open'].values, self.df.loc[self.current_step, 'Close'].values)
        
        return self._next_observation()
    
    def render(self, mode = 'human', Close =  True):
        #rendering the envirnoment
        profit = self.net_worth - initial_account_balance
        trade = 'Long'
        if(self.short_trade):
            trade = 'Short'
        
        transaction_is = 'Open' if(self.action_cntr%2!=0) else 'Settled' 
        
        print(f'Step : {self.current_step}')
        print(f'Balance : {self.balance}')
        print(f'Next trade is:  {trade} (Transaction is : {transaction_is})')
        print(f'Avg cost for held shares: {self.avg_cost} (Total sales values: {self.total_sales_value})')
        print(f'Networth: {self.net_worth} (Max net worth : {self.max_net_worth})')
        print(f'Net profit: {profit} (Reward: {self.reward})')
        print('-------------------------------------------------------------------------')
        
    def render_all(self, mode ='human'):
        window_ticks = np.arange(len(self._position_history))
        # plt.plot(self.df['Close'])
        for i, tick in enumerate(self._position_history):
            if(self._position_history[i]!=0):
                first_step = i
                break
             

        self.short_ticks = []
        self.long_ticks = []
        self.squared_off_ticks = []
        self.hold_ticks = []
        
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == 'Short':
                self.short_ticks.append(tick)
            elif self._position_history[i] == 'Long':
                self.long_ticks.append(tick)
            elif self._position_history[i] == 'Hold':
                self.hold_ticks.append(tick)
            
            
            if self._trade_completion_history[i] == 1:
                self.squared_off_ticks.append(tick)
                
        # plt.figure(figsize=(20,10))
        plt.plot(self.df.loc[first_step:, 'Close'].index.values,self.df.loc[first_step:,'Close'].values)
        plt.plot(self.short_ticks, self.df.loc[self.short_ticks, 'Close'].values, 'ro')
        plt.plot(self.long_ticks, self.df.loc[self.long_ticks, 'Close'].values, 'go')
        plt.plot(self.squared_off_ticks, self.df.loc[self.squared_off_ticks, 'Close'].values, 'kx')

        # plt.suptitle(
        #     "Total Reward: %.6f" % self._total_reward + ' ~ ' +
        #     "Total Profit: %.6f" % self._total_profit
        # )


#%% Testing the envirnoment       
# import pandas as pd
# df = pd.read_csv('D:\Reinforcement Learning\My codes\Custom gym/AAPL.csv')
# test = CustomDiscreteEnv(df)

# test.reset()
# for i in range(500):
#     action= np.random.randint(0,3)
#     obs, reward, done,_  = test.step(action)
#     if(reward!=0):
#         print('reward :' + str(reward) +' action cntr: '+ str(test.action_cntr) )
    # print(obs[0, 5,:])
    
    
#%%
import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 6), dtype=np.float16)

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df.loc[self.current_step: self.current_step +
                        5, 'Open'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'High'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'Low'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'Close'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'Volume'].values / MAX_NUM_SHARES,
        ])

        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ]], axis=0)

        return obs

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = random.uniform(
            self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])

        action_type = action[0]
        amount = action[1]

        if action_type < 1:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = (
                prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

        elif action_type < 2:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'Open'].values) - 6:
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = self.balance * delay_modifier
        done = self.net_worth <= 0

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(
            0, len(self.df.loc[:, 'Open'].values) - 6)

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
    