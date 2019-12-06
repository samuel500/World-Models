import numpy as np 
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool, Queue
from copy import deepcopy
from time import time
import gym
import random

from copy import deepcopy
from operator import attrgetter
import pickle
from model import Model, empty_class
import gc

from utils import *


class Population:

    def __init__(self, pop_size=64, n_process=None, n_eval=1, p_keep=0.5, n_candidate_eval=24, n_candidates=3, no_rew_early_stop=20,
                    rnn_size=256, controller_size=128, output_size=3, n_keep=None, hidden=True, use_prev_act=False, individual=Model):

        self.pop_size = pop_size #224
        self.n_process = n_process
        self.n_candidate_eval = n_candidate_eval
        self.n_candidates = n_candidates
        self.no_rew_early_stop = no_rew_early_stop
        if n_process is None:
            self.n_process = os.cpu_count()
        self.n_eval = n_eval
        self.p_keep = p_keep
        if n_keep:
            self.p_keep = n_keep/pop_size
        self.rnn_size = rnn_size
        self.controller_size = controller_size
        self.output_size = output_size

        self.models = [individual(no_rew_early_stop, rnn_size=rnn_size, controller_size=controller_size,
            hidden=hidden, use_prev_act=use_prev_act) for _ in range(self.pop_size)]
        self.models[0].summary()
        self.gen = 0
        self.elite = None

        self.results_queue = Queue()
        self.training_queue = Queue()
        self.pool = Pool(self.n_process, distribute, (self.training_queue, self.results_queue,individual))    


    def update_no_rew_early_stop(self):
        try:
            with open('no_rew_early_stop.info', 'r') as file:
                new_value = int(file.read())
                self.no_rew_early_stop = new_value
            for i in range(len(self.models)):
                self.models[i].no_rew_early_stop = new_value
        except:
            pass


    def evaluate(self, models, n=1, update_fitness=True):

        for i, model in enumerate(models):
            info = {'weights': model.get_weights(), 'attr': model.get_pickle_obj(), 'id': i}
            for _ in range(n):
                self.training_queue.put(info)

        results = []
        while len(results) < len(models)*n:
            result = self.results_queue.get()
            results.append(result)
       
        for result in results:
            models[result['id']].add_results(result['result'], update_fitness=update_fitness)


    def next_gen(self):
        self.update_no_rew_early_stop()


        self.evaluate(self.models)

        self.models.sort(key=attrgetter('fitness'), reverse=True)

        n_survivors = int(self.p_keep*self.pop_size)
        survivors = self.models[:n_survivors]


        candidates = self.models[:self.n_candidates]

        self.evaluate(candidates, n=self.n_candidate_eval, update_fitness=False)

        candidates.sort(key=attrgetter('mean_result'), reverse=True)
        if not self.elite:
            self.elite = candidates[0]

        if candidates[0].fitness > self.elite.fitness or len(self.elite.results[str(self.elite.generation)]) < 4:
            self.elite = candidates[0]
        else:
            to_buffer('candidate[0] fitness:'+ str(candidates[0].fitness))
            candidates[0].fitness = 999

        self.elite.fitness = 1000

        survivors.sort(key=attrgetter('fitness'), reverse=True)
        #self.models.sort(key=attrgetter('fitness'), reverse=True)
        for i in range(len(survivors)):
            survivors[i].add_rank(i)

        i=1
        while len(survivors)+i < self.pop_size+1:
            if len(survivors) <= 2:
                winner = candidates[0]
            else:
                competitors = np.random.choice(survivors, size=2, replace=False) # tournament selection
                winner = max(competitors, key=attrgetter('fitness'))
            if self.models[-i] is winner:
                raise
            self.models[-i].copy_model(winner)
            self.models[-i].mutate()
            
            i+=1

        return candidates[0]

    def train(self, generations=1500, disp_best=True, save_best=True):

        if disp_best:
            env = gym.make('CarRacing-v0', verbose=0)
            env = wrap_env(env, W, H, gray_scale=False)


        g = 0
        best_score = 0
        while g < generations:

       
            st = time()

            print('Generation', g)
            to_buffer('Generation' + str(g))
      
            best_model = self.next_gen()
            print(best_model)
            to_buffer(best_model.__repr__())
            if disp_best:
                best_model.evaluate(env, disp=True, n=1) #, vae_model, disp=True, n=1)
            if save_best and best_model.mean_result > best_score:
                best_model.save_all(name='{generation:04d}'.format(generation=g) + 'result:' + str(best_model.mean_result))
                best_score = best_model.mean_result
            g += 1
            print('T:', time()-st)
            to_buffer('T:' + str(time()-st))


def distribute(in_queue, out_queue, individual):
    print('starting', os.getpid())

    
    s = np.random.randint(1e8)
    np.random.seed(s+os.getpid()) # different seed for each process
    env = gym.make('CarRacing-v0', verbose=0)
    env = wrap_env(env, W, H, gray_scale=False)
    
    m = individual(rnn_size=256, controller_size=256)



    while True:
        model_info = in_queue.get()
        if model_info['attr'].rnn_size != m.rnn_size or model_info['attr'].controller_size != m.controller_size\
                or model_info['attr'].output_size != m.output_size or model_info['attr'].hidden != m.hidden\
                or model_info['attr'].use_prev_act != m.use_prev_act:
            tf.keras.backend.clear_session()
            del m
            gc.collect()

            m = individual(rnn_size=model_info['attr'].rnn_size, controller_size=model_info['attr'].controller_size,
                 output_size=model_info['attr'].output_size, hidden=model_info['attr'].hidden, use_prev_act=model_info['attr'].use_prev_act)
            
        m.set_weights(model_info['weights'])
        m.copy_model(model_info['attr'], from_pickle=True)

        res = m.evaluate(env)

        ret = {'id': model_info['id'], 'result': res}
        out_queue.put(ret)
        tf.keras.backend.clear_session()

    env.close()


def to_buffer(text):
    try:
        with open("buffer.txt", "a") as myfile:
            myfile.write(text + '\n')
    except:
        print('error appending to file')
        pass