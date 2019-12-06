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
import multiprocessing
from multiprocessing import Pool
from copy import deepcopy
from time import time
import gym
from atari_wrappers import WarpFrame
import random

from vae import VAE
from copy import deepcopy
from operator import attrgetter
import pickle

class empty_class:
    pass
'''
GA:
theta with gaussian noise,
size of gaussian noise (sigma)
probability of updating parts of the network (rnn and/or controller)
'''


def wrap_env(env, width=84, height=84, gray_scale=False):
    env = WarpFrame(env, width, height, gray_scale)
    return env


class Model(tf.keras.Model):

    def __init__(self, no_rew_early_stop=10):
        super().__init__()
        rnn_size = 256
        latent_space = 32

        
        self.rnn = tf.keras.layers.LSTM(rnn_size, return_state=True)
        self.rnn.build(input_shape=(1, 1, 32+3))
        #init = tf.initializers.variance_scaling()

        self.controller = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(rnn_size+latent_space,)),
                tf.keras.layers.Dense(units=256, activation=tf.nn.relu), #, kernel_initializer=init),
                tf.keras.layers.Dense(units=3, activation=None)
            ],
            name='controller'
        )

        self._epsilon = 0.01
        self._both_mutation_p = 0.9
        self._controller_mutation_p = 0.5

        self.generation = 0
        self.results = {}
        self.no_rew_early_stop = no_rew_early_stop
        self.rank_hist = []
        self.history = {}


    def __repr__(self):
        ret = ''.join(['Model(gen=', str(self.generation), ', eps=', str(self.epsilon), ', both_p=', str(self.both_mutation_p),\
            ', controller_p=', str(self.controller_mutation_p), ', no_rew=', str(self.no_rew_early_stop) ,', acc=', str(self.mean_result) ,')'])
        return ret

    def forward(self, obs, vae):
        if self.prev_act is None:
            self.prev_act = np.array([0,0,0])

        obs = obs.astype(np.float32)
        obs /= 255
        obs = obs.reshape((1, *obs.shape))

        z = vae.latent(obs)
        #z = tf.reshape(z, (1, *z.shape))

        rnn_in = tf.concat([z[0], self.prev_act], 0)
        rnn_in = tf.reshape(rnn_in, (1,1, *rnn_in.shape))
        if self.h is None:
            _, self.h, self.c = self.rnn(rnn_in)
        else:
            _, self.h, self.c = self.rnn(rnn_in, initial_state=[self.h, self.c])

        #print(r)
        #raise
        c_in = tf.concat([z[0],self.h[0]], 0) # batch D?
        c_in = tf.reshape(c_in, (1, *c_in.shape))
        acts = self.controller(c_in)
        acts = np.array(acts)[0]
        #acts[0] = tf.math.tanh(acts[0])
        #acts[1:] = tf.math.sigmoid(acts[1:])

        acts = np.tanh(acts)
        return acts


    def mutate(self):

        def mutate_weights(sequence):
            layers = sequence.get_weights()
            for i in range(len(layers)):
                layers[i] += np.random.normal(scale=self.epsilon, size=layers[i].shape)
            return layers
        self.history[str(self.generation)] = {'epsilon': self.epsilon, 'both_mutation_p': self.both_mutation_p,
             'controller_mutation_p': self.controller_mutation_p, 'no_rew_early_stop': self.no_rew_early_stop
             }
        self.epsilon += np.random.choice([0.001,-0.001] + 8*[0])
        self.both_mutation_p += np.random.choice([0.05,0.05,-0.05,-0.05,0.1,-0.1] + 14*[0])
        self.controller_mutation_p += np.random.choice([0.05,0.05,-0.05,-0.05,0.1,-0.1] + 14*[0])

        if random.random() < self.both_mutation_p:
            self.controller.set_weights(mutate_weights(self.controller))
            #print('rnn', self.rnn.get_weights())
            self.rnn.set_weights(mutate_weights(self.rnn))
        else:
            if random.random() < self.controller_mutation_p:
                self.controller.set_weights(mutate_weights(self.controller))
            else:
                self.rnn.set_weights(mutate_weights(self.rnn))
        self.generation += 1


    def evaluate(self, env, vae, n=1, disp=False):
        
        no_rew_early_stop = self.no_rew_early_stop
        rewards = []
        for r in range(n):
            #self.rnn.reset_states()
            done = False
            obs = env.reset()
            last_rew = 0
            tot_rew = 0
            self.h = None
            self.c = None
            self.prev_act = None

            for i in range(10000):
                if done:
                    break
                if disp:
                    env.render()

                act = self.forward(obs, vae)
                #print(act)
                self.prev_act = act
                obs, rew, done, _ = env.step(act)
                tot_rew += rew
                if rew < 0:
                    last_rew += 1
                else:
                    last_rew = 0

                if last_rew > no_rew_early_stop and i > 30:
                    #print('break @', i)
                    break
            rewards.append(tot_rew)
        self.h = None
        self.c = None
        self.prev_act = None
        
        return rewards


    def add_results(self, rewards):
        if str(self.generation) not in self.results:
            self.results[str(self.generation)] = []
        self.results[str(self.generation)] += rewards
        self.fitness = self.mean_result

    def add_rank(self, r):
        self.rank_hist.append(r)

    def copy_model(self, model, from_pickle=False):
        self._epsilon = model._epsilon
        self._both_mutation_p = model._both_mutation_p
        self._controller_mutation_p = model._controller_mutation_p

        self.generation = model.generation
        self.results = deepcopy(model.results)
        self.no_rew_early_stop = model.no_rew_early_stop
        self.rank_hist = deepcopy(model.rank_hist) 
        self.history = deepcopy(model.history)

        if not from_pickle:
            self.set_weights(model.get_weights())

    
    def get_pickle_obj(self):
        to_pickle = empty_class()
        to_pickle._epsilon = self._epsilon
        to_pickle._both_mutation_p = self._both_mutation_p
        to_pickle._controller_mutation_p = self._controller_mutation_p

        to_pickle.generation = self.generation
        to_pickle.results = deepcopy(self.results)
        to_pickle.no_rew_early_stop = self.no_rew_early_stop
        to_pickle.rank_hist = deepcopy(self.rank_hist)
        to_pickle.history = deepcopy(self.history)
        return to_pickle

    def save_all(self, name=None, save_path='./new_ga_ckpt/'):
        if name is None:
            name = '{generation:04d}'.format(generation=self.generation) + 'result:' + str(self.mean_result)
        full_name = os.path.join(save_path, name)
        self.save_weights(full_name)

        
        to_pickle = self.get_pickle_obj()
        if save_path[-1] != '/':
            save_path += '/'
        if not os.path.exists(save_path+'pkl/'):
            os.mkdir(save_path+'pkl/')
        name += '.pkl'
        save_path += 'pkl/'

        full_name = os.path.join(save_path, name)

        pickle.dump(to_pickle, open(full_name, "wb"))

    def load_all(self, name=None, save_path='./new_ga_ckpt/', latest=False):
        if save_path[-1] != '/':
            save_path += '/'
        if latest:
            model_path = tf.train.latest_checkpoint(save_path)
            name = model_path.replace(save_path, '')
        else:
            model_path = os.path.join(save_path, name)

        self.load_weights(model_path)

        
        save_path += 'pkl/'
        #full_name = os.path.join(save_path, name)
        full_name = save_path + name + '.pkl'
        try:
            pickled = pickle.load(open(full_name, 'rb'))
            self.copy_model(pickled, from_pickle=True)
        except FileNotFoundError:
            print('File', full_name, 'not found. Only weights restored')



    @property
    def mean_result(self):
        return sum(self.results[str(self.generation)])/len(self.results[str(self.generation)])
    


    @property    
    def epsilon(self):
        return self._epsilon
    @epsilon.setter
    def epsilon(self, val):
        self._epsilon = np.clip(val, 0.001, 0.04)
    @property    
    def both_mutation_p(self):
        return self._both_mutation_p
    @both_mutation_p.setter
    def both_mutation_p(self, val):
        self._both_mutation_p = np.clip(val, 0.05, 1)
    @property    
    def controller_mutation_p(self):
        return self._controller_mutation_p
    @controller_mutation_p.setter
    def controller_mutation_p(self, val):
        self._controller_mutation_p = np.clip(val, 0.05, 0.95)



class Population:

    def __init__(self, pop_size=64, n_process=None, n_eval=1, p_keep=0.5, n_candidate_eval=8, n_candidates=3, no_rew_early_stop=20):

        self.pop_size = pop_size #224
        self.n_process = n_process
        self.n_candidate_eval = n_candidate_eval
        self.n_candidates = n_candidates
        if n_process is None:
            self.n_process = os.cpu_count()
        self.n_eval = n_eval
        self.p_keep = p_keep

        self.models = [Model(no_rew_early_stop) for _ in range(self.pop_size)]

        self.gen = 0


    def next_gen(self):

        def split(a, n):
            k, m = divmod(len(a), n)
            return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

        #weights = [[w for w in m.get_weights()] for m in self.models]


        try:
            with open('no_rew_early_stop.info', 'r') as file:
                new_value = int(file.read())
            for i in range(len(self.models)):
                self.models[i].no_rew_early_stop = new_value
        except:
            pass


        weights = [{'weights': [w for w in m.get_weights()], 'attr': m.get_pickle_obj()} for m in self.models]


        split_weights = split(weights, self.n_process)

        results = distribute(next_gen_process, self.n_process, split_weights)
        #print(results)

        for i, result in enumerate(results):
            self.models[i].add_results(result)
        self.models.sort(key=attrgetter('fitness'), reverse=True)
        
        survivors = self.models[:int(self.p_keep*self.pop_size)]

        n_candidate_eval = self.n_candidate_eval
        n_candidates = self.n_candidates
        #weights = [[[w for w in m.get_weights()]]*n_candidate_eval for m in survivors[:n_candidates]]
        weights = [[{'weights': [w for w in m.get_weights()], 'attr': m.get_pickle_obj()}]*n_candidate_eval for m in survivors[:n_candidates]]

        weights = [w for mw in weights for w in mw]
        split_weights = split(weights, self.n_process)

        results = distribute(next_gen_process, self.n_process, split_weights)
        results = [results[i:i + n_candidate_eval] for i in range(0, len(results), n_candidate_eval)]
        results = [[r[0] for r in mr] for mr in results]

        for i, result in enumerate(results): ################################Check this
            survivors[i].add_results(result)

        candidates = survivors[:n_candidates]
        candidates.sort(key=attrgetter('fitness'), reverse=True)
        candidates[0].fitness = 1000

        survivors.sort(key=attrgetter('fitness'), reverse=True)
        for i in range(len(survivors)):
            survivors[i].add_rank(i)

        i=1
        while len(survivors)+i < self.pop_size+1:
            competitors = np.random.choice(survivors, size=2, replace=False) # tournament selection
            winner = max(competitors, key=attrgetter('fitness'))

            
            self.models[-i].copy_model(winner)
            
            self.models[-i].mutate()
            
            i+=1

        return candidates[0]





def distribute(function, n_process, args):

    with Pool(n_process) as pool:
        results = pool.starmap(next_gen_process, zip(range(n_process), args))

    results = [r for pool_res in results for r in pool_res]
    return results

def next_gen_process(id, model_info):

    latent_size = 32
    vae_model = VAE(latent_size)

    checkpoint_dir = './vae_ckpt/'
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    vae_model.load_weights(latest)

    s = np.random.randint(1e8)
    np.random.seed(s+id) # different seed for each process
    env = gym.make('CarRacing-v0', verbose=0)
    env = wrap_env(env, W, H, gray_scale=False)
    
    results = []
    for info in model_info:
        m = Model()
        m.set_weights(info['weights'])
        m.copy_model(info['attr'], from_pickle=True)
        res = m.evaluate(env, vae_model)
        results.append(res)

    env.close()

    return results

'''
1. test all once
2. test top 3 20x
'''
W = 64
H = 64


model_save_path = './new_ga_ckpt/'
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
if not os.path.exists(model_save_path+'pkl/'):
    os.mkdir(model_save_path+'pkl/')


if __name__=='__main__':
    multiprocessing.set_start_method('spawn')  # hang
    latent_size = 32
    vae_model = VAE(latent_size)

    checkpoint_dir = './vae_ckpt/'
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    vae_model.load_weights(latest)


    no_rew_early_stop = 10
    pop_size = 400
    p_keep = 0.25
    n_candidate_eval=12
    n_candidates=3

    population = Population(pop_size=pop_size, p_keep=p_keep, n_candidate_eval=n_candidate_eval, n_candidates=n_candidates, no_rew_early_stop=no_rew_early_stop)

    #best_result = 50
    generations = 1500
    for g in range(1, generations+1):
        print('Generation', g)
        st = time()

        best_model = population.next_gen()
        print(best_model)
        env = gym.make('CarRacing-v0', verbose=0)
        env = wrap_env(env, W, H, gray_scale=False)
        best_model.evaluate(env, vae_model, disp=True, n=1)
        
        env.close()


        #best_model.results = str(best_model.results)
        best_model.save_all(name='{generation:04d}'.format(generation=g) + 'result:' + str(best_model.mean_result))


        #if best_model.mean_result > best_result:

        #    best_model.save_weights(os.path.join(model_save_path,'{generation:04d}'.format(generation=g) + 'result:' + str(best_model.mean_result)))

        #best_model.results = dict(best_model.results)

        print('T:', time()-st)