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
from atari_wrappers import WarpFrame
import random

from vae import VAE
from copy import deepcopy
from operator import attrgetter
import pickle
from model import Model, empty_class
from model2 import Model2, DummyModel
from risi import RisiModel
from merger import merge_all
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

        latent_size = 32
        vae_model = VAE(latent_size)

        checkpoint_dir = './vae_ckpt/'
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        vae_model.load_weights(latest)

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
                best_model.evaluate(env, vae_model, disp=True, n=1) #, vae_model, disp=True, n=1)
            if save_best and best_model.mean_result > best_score:
                best_model.save_all(name='{generation:04d}'.format(generation=g) + 'result:' + str(best_model.mean_result))
                best_score = best_model.mean_result
            g += 1
            print('T:', time()-st)
            to_buffer('T:' + str(time()-st))





def to_buffer(text):
    try:
        with open("buffer.txt", "a") as myfile:
            myfile.write(text + '\n')
    except:
        print('error appending to file')
        pass


class PopulationElite(Population):

    def elite_evaluate(self,models, n=1):
        ensemble_model = Model(rnn_size=self.rnn_size+self.elite.rnn_size, controller_size=self.controller_size+self.elite.controller_size, 
            output_size=self.output_size+self.elite.output_size, no_rew_early_stop=self.no_rew_early_stop)
        for i, model in enumerate(models):
            combined_weights = merge_all([self.elite.get_weights(), model.get_weights()])

            info = {'weights': combined_weights, 'attr': ensemble_model.get_pickle_obj(), 'id': i}
            for _ in range(n):
                self.training_queue.put(info)

        tf.keras.backend.clear_session()
        del ensemble_model
        gc.collect()

        results = []
        while len(results) < len(models)*n:
            result = self.results_queue.get()
            results.append(result)
       
        for result in results:
            models[result['id']].add_results(result['result'])


    def start_elite_support(self, n_init_eval=3, n_elite_candidate_eval=16, n_elite_candidates=3, from_start=True): #######################3
        self.evaluate(self.models, n=n_init_eval)
        self.models.sort(key=attrgetter('fitness'), reverse=True)
        candidates = self.models[:n_elite_candidates]

        self.evaluate(candidates, n=n_elite_candidate_eval)

        candidates.sort(key=attrgetter('fitness'), reverse=True)

        self.elite = candidates[0]
        if from_start:
            tf.keras.backend.clear_session()
            del self.models
            gc.collect()

            self.models = [Model(self.no_rew_early_stop, rnn_size=self.rnn_size, controller_size=self.controller_size) for _ in range(self.pop_size)]


    def evolve_elite_support(self):
        ensemble_model = Model(rnn_size=self.rnn_size+self.elite.rnn_size, controller_size=self.controller_size+self.elite.controller_size, 
            output_size=self.output_size+self.elite.output_size, no_rew_early_stop=self.no_rew_early_stop)
        
        self.elite_evaluate(self.models)

        self.models.sort(key=attrgetter('fitness'), reverse=True)
        survivors = self.models[:int(self.p_keep*self.pop_size)]

        candidates = survivors[:self.n_candidates]

        self.elite_evaluate(candidates, n=self.n_candidate_eval)


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
        ensemble_model.set_weights(merge_all([self.elite.get_weights(), candidates[0].get_weights()]))
        pick = candidates[0].get_pickle_obj()
        ensemble_model.copy_model(pick, from_pickle=True)
        ensemble_model.rnn_size = self.rnn_size + self.elite.rnn_size
        ensemble_model.controller_size = self.controller_size + self.elite.controller_size
        ensemble_model.output_size = self.output_size + self.elite.output_size
        return ensemble_model


    def evolutionary_leap(self, p_leap_keep=0.2, n_ensemble_eval=2):
        ensemble_model = Model(rnn_size=self.rnn_size+self.elite.rnn_size, controller_size=self.controller_size+self.elite.controller_size, 
            output_size=self.output_size+self.elite.output_size, no_rew_early_stop=self.no_rew_early_stop)

        self.elite_evaluate(self.models, n=n_ensemble_eval)
        self.models.sort(key=attrgetter('fitness'), reverse=True)

        n_init_pop = int(self.pop_size*p_leap_keep)

        survivors = self.models[:n_init_pop]


        new_models = [Model(rnn_size=self.rnn_size+self.elite.rnn_size, controller_size=self.controller_size+self.elite.controller_size, 
            output_size=self.output_size+self.elite.output_size, no_rew_early_stop=self.no_rew_early_stop) for _ in range(self.pop_size)]

        for i, model in enumerate(survivors):
            new_models[i].set_weights(merge_all([self.elite.get_weights(), model.get_weights()]))

        tf.keras.backend.clear_session()
        del self.models
        gc.collect()

        self.models = new_models

        new_pop = self.models[:n_init_pop]
        self.evaluate(new_pop)
        new_pop.sort(key=attrgetter('fitness'), reverse=True)

        
        candidates = new_pop[:self.n_candidates]
        self.evaluate(candidates, n=self.n_candidate_eval)

        candidates.sort(key=attrgetter('fitness'), reverse=True)
        candidates[0].fitness = 1000


        i=1
        while n_init_pop+i < self.pop_size+1:
            competitors = np.random.choice(new_pop, size=2, replace=False) # tournament selection
            winner = max(competitors, key=attrgetter('fitness'))
            
            self.models[-i].copy_model(winner)
            self.models[-i].mutate()
            
            i+=1
        #self.rnn_size *= 2
        #self.controller_size *= 2 
        #self.output_size *= 2

        return candidates[0]



    def train(self, generations=1500, disp_best=True):

        # next_gen
        # start_elite_support
        # evolve_elite_support
        # evolutionary_leap
        # repeat

        env = gym.make('CarRacing-v0', verbose=0)
        env = wrap_env(env, W, H, gray_scale=False)
        latent_size = 32
        vae_model = VAE(latent_size)

        checkpoint_dir = './vae_ckpt/'
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        vae_model.load_weights(latest)
            
        n_next_gen = 2
        n_elite_support = 1

        g = 0
        while g < generations:

            for _ in range(n_next_gen):   
                st = time()

                print('Generation', g)
          
                best_model = self.next_gen()
                print(best_model)
                if disp_best:
                    best_model.evaluate(env, vae_model, disp=True, n=1)
                best_model.save_all(name='{generation:04d}'.format(generation=g) + 'result:' + str(best_model.mean_result))
                g += 1
                print('T:', time()-st)

            st = time()

            print('Generation', g)
          
            print('start_elite_support')

            self.start_elite_support()
            print('T:', time()-st)

            for _ in range(n_elite_support):
                st = time()

                print('Generation', g)

                best_model = self.evolve_elite_support()
                print(best_model)
                if disp_best:
                    best_model.evaluate(env, vae_model, disp=True, n=1)
                best_model.save_all(name='{generation:04d}'.format(generation=g) + 'result:' + str(best_model.mean_result))
                g += 1
                print('T:', time()-st)


            st = time()

            print('Generation', g)

            print('Leap')

            best_model = self.evolutionary_leap()        
            print(best_model)

            if disp_best:
                best_model.evaluate(env, vae_model, disp=True, n=1)
            print('T:', time()-st)


            best_model.save_all(name='{generation:04d}'.format(generation=g) + 'result:' + str(best_model.mean_result))
            g += 1

        env.close()


   
class PopulationLeap(Population):

    def evolutionary_leap(self, p_mate=0.25, n_alpha=3, p_mate_keep=0.25, n_init_eval=2, n_ensemble_eval=3):


        self.evaluate(self.models, n=n_init_eval)

        self.models.sort(key=attrgetter('fitness'), reverse=True)
        mates = self.models[:int(p_mate*self.pop_size)]
        alphas = mates[:n_alpha]

        #tot_rnn = sum([m.rnn_size for m in models])
        #tot_con = sum([m.controller_size for m in models])
        ensemble_model = Model(rnn_size=2*self.rnn_size, controller_size=2*self.controller_size, output_size=2*self.output_size, no_rew_early_stop=self.no_rew_early_stop)
        
        all_combined_weights = {}
        n_put = 0
        for i, alpha in enumerate(alphas):
            for y, mate in enumerate(mates):
                if i>=y:
                    continue

                combined_weights = merge_all([alpha.get_weights(), mate.get_weights()])
                all_combined_weights[(i,y)] = combined_weights
                info = {'weights': combined_weights, 'attr': ensemble_model.get_pickle_obj(), 'id': (i,y)}
                for _ in range(n_ensemble_eval):
                    self.training_queue.put(info)
                    n_put += 1


        results = {}
        gotten = 0
        while gotten < n_put:
            result = self.results_queue.get()
            gotten += 1
            if result['id'] not in results:
                results[result['id']] = []
            results[result['id']] += result['result']

        new_pop = sorted(results.items(), key=lambda x: sum(x[1]), reverse=True)
        n_init_pop = int(len(new_pop)*p_mate_keep)
        new_pop = new_pop[:n_init_pop]
        print(new_pop)
        
        tf.keras.backend.clear_session()
        del self.models
        gc.collect()

        self.models = [Model(rnn_size=2*self.rnn_size, controller_size=2*self.controller_size,
                output_size=2*self.output_size, no_rew_early_stop=self.no_rew_early_stop) for _ in range(self.pop_size)]

        for i, (key, res) in enumerate(new_pop):
            self.models[i].add_results(res)
            self.models[i].set_weights(all_combined_weights[key])

        new_pop = self.models[:n_init_pop]
        new_pop.sort(key=attrgetter('fitness'), reverse=True)

        candidates = new_pop[:self.n_candidates]
        self.evaluate(candidates, n=self.n_candidate_eval)

        candidates.sort(key=attrgetter('fitness'), reverse=True)
        candidates[0].fitness = 1000


        i=1
        while n_init_pop+i < self.pop_size+1:
            competitors = np.random.choice(new_pop, size=2, replace=False) # tournament selection
            winner = max(competitors, key=attrgetter('fitness'))
            
            self.models[-i].copy_model(winner)
            self.models[-i].mutate()
            
            i+=1


        self.rnn_size *= 2
        self.controller_size *= 2
        self.output_size *= 2

        return new_pop[0]


    def train(self, generations=1500, evolutionary_leap_gens=10, disp_best=True):
        env = gym.make('CarRacing-v0', verbose=0)
        env = wrap_env(env, W, H, gray_scale=False)
        latent_size = 32
        vae_model = VAE(latent_size)

        checkpoint_dir = './vae_ckpt/'
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        vae_model.load_weights(latest)
            
        for g in range(1, generations+1):
            print('Generation', g)
            st = time()

            if not g%evolutionary_leap_gens:
                print('Leap')
                best_model = self.evolutionary_leap()
            else:
                best_model = self.next_gen()
        
            print(best_model)

            if disp_best:
                best_model.evaluate(env, vae_model, disp=True, n=1)


            best_model.save_all(name='{generation:04d}'.format(generation=g) + 'result:' + str(best_model.mean_result))


            print('T:', time()-st)
        env.close()
   



class NewPopulation(Population):
    """
    Options:
        -update elite
        -increase size (and get update mask)
        -evolve masked from scratch
        -evolve masked with gaussian noise
        -get random mask


     # increase size
        # train from 0 or 
        # with normal random init weights (bigger pop?)
    # evolve dropout mask to find nodes to reset or delete

    # evolution options:
    """

    def __init__(self, pop_size=64, n_process=None, n_eval=1, p_keep=0.5, n_candidate_eval=8, n_candidates=3, no_rew_early_stop=20,
                    rnn_size=256, controller_size=128, output_size=3):

        self.pop_size = pop_size 
        self.n_process = n_process
        self.n_candidate_eval = n_candidate_eval
        self.n_candidates = n_candidates

        self.no_rew_early_stop = no_rew_early_stop
        if n_process is None:
            self.n_process = os.cpu_count()

        self.n_eval = n_eval
        self.p_keep = p_keep
        self.rnn_size = rnn_size
        self.controller_size = controller_size
        self.output_size = output_size

        self.elite = Model2(no_rew_early_stop, rnn_size=rnn_size, controller_size=controller_size)


        self.gen = 0

        self.results_queue = Queue()
        self.training_queue = Queue()
        self.pool = Pool(self.n_process, distribute, (self.training_queue, self.results_queue, True))    
    
    def elite_from_file(self, name, two=True):
        if two:
            self.elite.load_weights(name)


    def evolve(self, generations=1, pop_size=64, n_init_eval=1, n_candidates=3, n_survivors=16, n_candidate_eval=8,\
                n_end_eval=2, n_end_candidates=3, n_end_candidate_eval=16,\
                masks=None, mode=None, eps=None):

        self.update_no_rew_early_stop()

        st = time()
        if not eps:
            eps = self.elite.epsilon

        weights = self.elite.get_mutated_weights(n=pop_size, masks=masks, mode=mode, eps=eps)

        models = [DummyModel(self.no_rew_early_stop, rnn_size=self.elite.rnn_size, controller_size=self.elite.controller_size) for _ in range(pop_size)]
        for i, model in enumerate(models):
            model.set_weights(weights[i])

        for g in range(generations):
            st = time()

            self.evaluate(models, n=n_init_eval)

            models.sort(key=attrgetter('fitness'), reverse=True)
            survivors = models[:n_survivors]


            candidates = survivors[:n_candidates]

            self.evaluate(candidates, n=n_candidate_eval)

            candidates.sort(key=attrgetter('fitness'), reverse=True)
            candidates[0].fitness = 1000

            survivors.sort(key=attrgetter('fitness'), reverse=True)
            models.sort(key=attrgetter('fitness'), reverse=True)

            for i in range(len(survivors)):
                survivors[i].add_rank(i)

            i=1
            while len(survivors)+i < pop_size+1:
                if len(survivors) <= 2:
                    winner = candidates[0]
                else:
                    competitors = np.random.choice(survivors, size=2, replace=False) # tournament selection
                    winner = max(competitors, key=attrgetter('fitness'))
                
                models[-i].copy_model(winner)
                models[-i].next_gen(new_weights=self.elite.get_mutated_weights(n=1, weights=models[-i].get_weights(), masks=masks, mode=mode, eps=eps)[0])
                i+=1

            print(candidates[0]) 
            #yield candidates[0] #?
            print('T:', time()-st)

        print('End of Era')
        self.evaluate(models, n=n_end_eval)
        models.sort(key=attrgetter('fitness'), reverse=True)

        candidates = models[:n_end_candidates]

        self.evaluate(candidates, n=n_end_candidate_eval)

        candidates.sort(key=attrgetter('fitness'), reverse=True)

        self.elite.copy_model(candidates[0])
        print(candidates[0])
        print('T:', time()-st)

        return candidates[0]



    def train(self, disp_best=False, save_best=False):
        env = gym.make('CarRacing-v0', verbose=0)
        env = wrap_env(env, W, H, gray_scale=False)
        latent_size = 32
        vae_model = VAE(latent_size)
        checkpoint_dir = './vae_ckpt/'
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        vae_model.load_weights(latest)

        init = './saved/0297result_372.95431161691926'
        self.elite_from_file(init)

        #self.evolve(generations=0, pop_size=64, mode='init', n_end_eval=2, n_end_candidate_eval=8)
        #print(self.elite)

        #self.evolve(generations=1, pop_size=64, mode='normal')
        print(self.elite)


        if disp_best:
            self.elite.evaluate(env, vae_model, disp=True, n=1)

        for i in range(1, 1000):
            st = time()

            print('Era', i)

            #masks = self.elite.change_size(rnn_plus=0, controller_plus=16)            
            #self.evolve(generations=20, pop_size=64, n_end_eval=2, n_end_candidate_eval=16)
            #self.evolve(generations=4, pop_size=64, mode='normal', masks=masks, n_end_eval=3, n_end_candidate_eval=8, n_survivors=8, eps=0.01)
            #print('rand evolve 1')
            masks = self.elite.get_random_mask(p=0.08, mask_mask=[False, False, False, True, True, False, False], with_bias=True)[0]
            self.evolve(generations=0, pop_size=96, mode='normal', masks=masks, n_end_eval=3, n_end_candidate_eval=16, n_survivors=8, eps=0.025)
            #print('rand evolve 2')
            #masks = self.elite.get_random_mask(p=1, mask_mask=[False, False, False, False, False, True, True], with_bias=True)[0]
            #self.evolve(generations=4, pop_size=64, mode='normal', masks=masks, n_end_eval=3, n_end_candidate_eval=8, n_survivors=8, eps=0.01)
            
            print('elite:', self.elite)
            if disp_best:
                self.elite.evaluate(env, vae_model, disp=True, n=1)
            if save_best:
                self.elite.save_all(name='{era:04d}'.format(era=i) + 'result:' + str(self.elite.mean_result))
            print('TT:', time()-st)


        #print(masks)
        
        env.close()








def distribute(in_queue, out_queue, individual):
    print('starting', os.getpid())

    latent_size = 32
    vae_model = VAE(latent_size)

    checkpoint_dir = './vae_ckpt/'
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    vae_model.load_weights(latest)
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

        res = m.evaluate(env, vae=vae_model)

        ret = {'id': model_info['id'], 'result': res}
        out_queue.put(ret)
        tf.keras.backend.clear_session()

    env.close()