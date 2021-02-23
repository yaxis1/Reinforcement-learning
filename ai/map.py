# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# Importing the Dqn object from Artificial Intelligence in ai.py
from ai import Dqn

Dqn.brain

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0 # the total number of points in the last drawing
length = 0 # the length of the last drawing

# Getting our AI, which is called "brain", and that contains our neural network that represents our Q-function
brain = Dqn(5,4,0.9) # 5 sensors, 4 actions, gama = 0.9
action2rotation = [0,20,-20,-10] # action = 0 => no rotation, action = 1 => rotate 20 degres, action = 2 => rotate -20 degres, action = 3 => reverse -10 
last_reward = 0 # initializing the last reward
scores = [] # initializing the mean score curve (sliding window of the rewards) with respect to time




