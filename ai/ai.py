#AI for the car

# Importing libraries
import numpy as np #Working with arrays
import random #taking random samples
import os #load and save the model
import torch #only lib that can handle dynamic graphs
import torch.nn as nn#neural network
import torch.nn.functional as F #functions for nn 
import torch.optim as optim #optimizer for stochastic gradient descent
import torch.autograd as autograd 
from torch.autograd import Variable #convert for tensors to gradient

# Creating the architecture of the Neural network

class Network(nn.Module):
    #Inheriting from nn.Module
    def __init__(self, input_size, nb_action):# input_size - 5 : 3 signals + 2 orientations(goal) # nb_action - l,r,f,b
        super(Network, self).__init__() #uses nn.Module
        self.input_size = input_size
        self.nb_action = nb_action
        #2 full connections for each hidden layer.-|-
        self.fc1 = nn.Linear(input_size, 30) #-| After a lot of experimenting 30!
        self.fc2 = nn.Linear(30, nb_action) #|-

    def forward(self, state): #Forward propogation using rectifier activation cz non-linear problem and get Q values
        #Activating hidden neurons
        x = F.relu(self.fc1(state)) #Rectifier function to activate hidden neuron x
        q_values = self.fc2(x) #getting Q values from fc2 that has neural connections from x
        return q_values

# Experience Replay for long term memory
class ReplayMemory(object): #From the memory of last 100 events we take batches to make next update, move by selecting the next action
    def __init__(self,capacity): #Future objects 
        self.capacity = capacity # 100 
        self.memory = [] #list of memories

    def push(self, event): #append new event to memory and make sure the list is < 100
        #event = (last state, new state, last action, last reward)
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size): #Sample from the memory( last 100 events)
        #while event = (last state, new state, last action, last reward)
        #zip(*random.sample()) - zips the tuples in the format (state1,state2) (act1,act2) (r1,r2)
        samples = zip(*random.sample(self.memory, batch_size)) # zipping random samples from memory of fixed batch size 
        return map(lambda x: Variable(torch.cat(x, 0)),samples) #concatnate samples to first dimention then convert them to torch variable(tensor and gradient)

# Implementing Deep Q Learning

class Dqn(): 

    def __init__(self, input_size, nb_action, gamma): # Takes Network class and ReplayMemory
        self.gamma = gamma
        self.reward_window = []  # Mean of last 100 rewards changing with time
        self.model = Network(input_size, nb_action)# The Neural Network
        self.memory = ReplayMemory(100000) # Number of events
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001) # Connects adam optim to our neural network
        self.last_state = torch.Tensor(input_size).unsqueeze(0) # 3 signals + 2 orientations => converted to torch vector and additional dimension for batch
        self.last_action = 0 # action2rotation [0,20,-20]
        self.last_reward = 0 # -1 or +1 
    
    def select_action(self, state):
        #Softmax - distribution of probabilities for Q values (sum up to 1)/ if argmax it takes max of Q values (not experimenting other q values)
        probs = F.softmax(self.model(Variable(state, volatile = True))*7) #T =7 #No gradient here
        #Higher the T value more confident the action
        action = probs.multinomial() 
        return action.data[0,0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        # Markov's decision process
        outputs = self.model()