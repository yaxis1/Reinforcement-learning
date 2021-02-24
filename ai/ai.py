#AI for the car

#Importing libraries
import numpy as np #Working with arrays
import random #taking random samples
import os #load and save the model
import torch #only lib that can handle dynamic graphs
import torch.nn #neural network
import torch.nn.functional as F #functions for nn 
import torch.optim as optim #optimizer for stochastic gradient descent
import torch.autograd as autograd 
from torch.autograd import Variable #convergence for tensors to gradient