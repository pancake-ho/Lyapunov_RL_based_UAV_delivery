import numpy as np
import pandas as pd
import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import idlelib
import PIL.Image

from torchvision.transforms import ToTensor
from matplotlib import rcParams

EP_LEN = 60
N_VERTIPORT = 5
N_USER = 200
N_COMMAGENT = 10
N_DQNAGENT = 0
GRID = 32000
OBSERVABLE = GRID * 3
H_MAX = 600
H_MIN = 0
VELOCITY = 4425
MTOW = 4

font = {'size' : 10}
plt.rc('font', **font)
rcParams.update({'figure.autolayout': True})

class Vertiport:
    def __init__(self, id=None, x=None, y=None, z=None):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
    

class User:
    def __init__(self, id=None, Departure=None, Arrival=None):
        self.id = id
        self.Arrival = Arrival
        self.Departure = Departure
        self.connect = {
            'drone': -1,
            'isOnboard': 0,
        }
    
    def is_support(self):
        return self.connect['drone'], self.connect['isOnboard'], self.connect['isArrive']


class Drone:
    def __init__(self, id=None, x=None, y=None, z=None, vertiport=None):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.vertiport = vertiport
        self.onboarding = 0
        self.counter = 4
        self.takeoff = 0
        self.MTOW = MTOW