"""
Run this with:
xvfb-run -a -s "-screen 0 1400x900x24" -- python3 ga.py
"""

import numpy as np 
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool, Queue
from copy import deepcopy
from time import time
import gym
import random

from copy import deepcopy
from operator import attrgetter
import pickle
from model import Model
import gc
from population import *
from utils import *




model_save_path = './ga_ckpt/'
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
if not os.path.exists(model_save_path+'pkl/'):
    os.mkdir(model_save_path+'pkl/')


if __name__=='__main__':
    multiprocessing.set_start_method('spawn')  # or hang
    
    no_rew_early_stop = 15
    pop_size = 64
    p_keep = 0.3
    n_candidate_eval=12
    n_candidates=3

    #mt = Model()
    #mt.summary()
    #population = NewPopulation(pop_size=pop_size, p_keep=p_keep, n_candidate_eval=n_candidate_eval, n_candidates=n_candidates, 
    #                    no_rew_early_stop=no_rew_early_stop, rnn_size=8, controller_size=2048)
    #population = Population(pop_size=64, p_keep=0.25, rnn_size=8, controller_size=2048)
    population = Population(pop_size=200, p_keep=0.35, rnn_size=256, controller_size=128, 
            individual=Model, n_candidate_eval=8, hidden=True, use_prev_act=False)

    population.train(disp_best=False, save_best=True)