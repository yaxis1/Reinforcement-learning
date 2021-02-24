#AI for the car

#Importing libraries
import numpy as np #Working with arrays
import random #taking random samples
import os #load and save the model
import torch #only lib that can handle dynamic graphs
import torch.nn as nn#neural network
import torch.nn.functional as F #functions for nn 
import torch.optim as optim #optimizer for stochastic gradient descent
import torch.autograd as autograd 
from torch.autograd import Variable #convergence for tensors to gradient

#Creating the architecture of the Neural network
class Network(nn.Module):
    #Inheriting from nn.Module
    def __init__(self, input_size, nb_action):# input_size - 5 : 3 signals + 2 orientations(goal) # nb_action - l,r,f,b
        
        super(Network, self).__init__() #uses nn.Module
        self.input_size = input_size
        self.nb_action = nb_action
        #2 full connections for each hidden layer.-|-
        self.fc1 = nn.Linear(input_size, 30) #-| After a lot of experimenting 30!
        self.fc2 = nn.Linear(30, nb_action) #|-

    def forward(self):
        #Rectifier activation cz non-linear problem
        pass