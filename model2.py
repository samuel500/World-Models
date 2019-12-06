import numpy as np 
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
import random
from copy import deepcopy
import pickle

from mutations import *

class empty_class:
    pass


class Model2(tf.keras.Model):

    def __init__(self, no_rew_early_stop=10, rnn_size=256, latent_space=32, controller_size=128, output_size=3, hada=True):
        super().__init__()
        self.rnn_size = rnn_size
        self.controller_size = controller_size
        self.latent_space = latent_space
        self.output_size = output_size

        
        self.rnn = tf.keras.layers.LSTM(rnn_size, return_state=True)
        self.rnn.build(input_shape=(1, 1, 32+3))
        self.hada = hada
        if hada:
            h_act = tf.nn.tanh
        else:
            h_act = tf.nn.relu
        h_act = tf.nn.relu
        self.controller_hidden = tf.keras.layers.Dense(units=controller_size, activation=h_act)
        self.controller_hidden.build(input_shape=(1, 1, rnn_size+latent_space))
        self.controller_output = tf.keras.layers.Dense(units=output_size, activation=None)
        self.controller_output.build(input_shape=(1, 1, controller_size))

        self._epsilon = 0.005

        self.generation = 0
        self.results = {}
        self.no_rew_early_stop = no_rew_early_stop
        self.rank_hist = []
        self.history = {}

        #self.training_mask = [1]*len(self.get_weights())

        self.initializers = [
            #LSTM
            tf.random_uniform_initializer(minval=-np.sqrt(6/(32+3+4*rnn_size)), maxval=np.sqrt(6/(32+3+4*rnn_size))),
            tf.keras.initializers.Orthogonal(),
            tf.zeros_initializer(),
            # Hidden controller
            tf.random_uniform_initializer(minval=-np.sqrt(6/(32+rnn_size)), maxval=np.sqrt(6/(32+rnn_size))),
            tf.zeros_initializer(),
            # Output
            tf.random_uniform_initializer(minval=-np.sqrt(6/(controller_size+output_size)), maxval=np.sqrt(6/(controller_size+output_size))),
            tf.zeros_initializer(),

        ]
        #tf.random_uniform_initializer(minval= , maxval= )   # sqrt(6 / (fan_in + fan_out))
        #tf.keras.initializers.Orthogonal()
        #tf.keras.initializers.GlorotUniform



    def __repr__(self):
        ret = ''.join(['Model(gen=', str(self.generation), ', eps=', str(self.epsilon), ', no_rew=', str(self.no_rew_early_stop), ', acc=',\
            str(self.mean_result) , ', rnn_size=', str(self.rnn_size), ', controller_size=',\
            str(self.controller_size), ', output_size=', str(self.output_size), ')'])
        return ret


    def forward(self, obs, vae):
        if self.prev_act is None:
            self.prev_act = np.array([0,0,0])

        obs = obs.astype(np.float32)
        obs /= 255
        obs = obs.reshape((1, *obs.shape))

        z = vae.latent(obs)

        rnn_in = tf.concat([z[0], self.prev_act], 0)
        rnn_in = tf.reshape(rnn_in, (1,1, *rnn_in.shape))
        if self.h is None:
            _, self.h, self.c = self.rnn(rnn_in)
        else:
            _, self.h, self.c = self.rnn(rnn_in, initial_state=[self.h, self.c])

        c_in = tf.concat([z[0],self.h[0]], 0) # batch D?
        c_in = tf.reshape(c_in, (1, *c_in.shape))
        hidd = self.controller_hidden(c_in)
        out = self.controller_output(hidd)
        acts = np.array(out)[0]

        acts = np.tanh(acts)
        act = np.array([0.,0.,0.])
        for i in range(0,self.output_size,3):
            act += acts[i:i+3]
        act /= int(self.output_size/3)

        if self.hada:
            act[1] = (act[1]+1)/2
            act[2] = np.clip(act[2], 0, 1)

        return act, out


    def forward2(self, obs, vae):
        if self.prev_act is None:
            self.prev_act = np.array([0,0,0])

        obs = obs.astype(np.float32)
        obs /= 255
        obs = obs.reshape((1, *obs.shape))

        z = vae.latent(obs)

        rnn_in = tf.concat([z[0], self.prev_act], 0)
        rnn_in = tf.reshape(rnn_in, (1,1, *rnn_in.shape))
        if self.h is None:
            _, self.h, self.c = self.rnn(rnn_in)
        else:
            _, self.h, self.c = self.rnn(rnn_in, initial_state=[self.h, self.c])

        c_in = tf.concat([z[0],self.h[0]], 0) # batch D?
        c_in = tf.reshape(c_in, (1, *c_in.shape))
        hidd = self.controller_hidden(c_in)
        acts = self.controller_output(hidd) 
        print(type(acts))
        #acts = np.array(acts)

        return acts


    def get_mutated_weights(self, weights=None, mode=None, masks=None, n=1, eps=None): 
        """
        mode: {None: normally mutated weights (+normal(eps)), 
                initialize: use layer's init to (re)initialize the weights in mask}
        masks: {None: array of size 7 (3 for LSTM + 2 for hidden layer + 2 for output)}
        """
        if not eps:
            eps = self.epsilon
        if not masks:
            masks = 7*[1]
        if not weights:
            weights = self.get_weights()

        all_new_weights = []
        for _ in range(n):
            new_weights = []
            if not mode or mode in {'normal'}:
                for i, w in enumerate(weights):
                    new_weights.append(w+masks[i]*np.random.normal(scale=eps, size=w.shape))
            elif mode in {'initialize', 'init'}:
                for i, w in enumerate(weights):
                    #print(w.shape)
                    rand_w = self.initializers[i](w.shape).numpy()
                    new_weights.append(w*(1-masks[i])+masks[i]*rand_w)
            all_new_weights.append(new_weights)

        return all_new_weights


    def get_random_mask(self, p=0.1, mask_mask=None, with_bias=False, n=1):
        """
        :p: share of mutable weights
        """
        all_masks = []
        for _ in range(n):
            masks = []
            for i, w in enumerate(self.get_weights()):
                if len(w.shape) == 1 and not with_bias:
                    masks.append(0)
                else:
                    if mask_mask:
                        if not mask_mask[i]:
                            masks.append(0)
                            continue
                    num = np.prod(w.shape)
                    mask = np.zeros(num)
                    mask[:int(num*p)] = 1
                    np.random.shuffle(mask)
                    mask = mask.reshape(w.shape)
                    masks.append(mask)
            all_masks.append(masks)
        return all_masks



    def change_size(self, new_rnn=None, new_controller=None, new_output=None,\
                    rnn_plus=0, controller_plus=0, output_plus=0): # change scale of weights?

        if rnn_plus and not new_rnn:
            new_rnn = self.rnn_size + rnn_plus
        if controller_plus and not new_controller:
            new_controller = self.controller_size + controller_plus
        if output_plus and not new_output:
            new_output = self.output_size + output_plus

        all_masks = []

        if new_rnn:
            assert new_rnn>=self.rnn_size
            self.rnn_size = new_rnn
            weights = self.rnn.get_weights()
            
            #tf.keras.backend.clear_session()
            del self.rnn
            #gc.collect()

            new_weights, masks = mutate_lstm(weights, self.rnn_size)
            all_masks += masks

            self.rnn = tf.keras.layers.LSTM(self.rnn_size, return_state=True)
            self.rnn.build(input_shape=(1, 1, 32+3))
            self.rnn.set_weights(new_weights)
        else:
            all_masks += [0,0,0]
            

        if new_controller or new_rnn:
            if new_controller:
                assert new_controller >= self.controller_size
                self.controller_size = new_controller
            weights = self.controller_hidden.get_weights()

            del self.controller_hidden

            new_weights, masks = mutate_controller(weights, self.rnn_size, self.controller_size)
            all_masks += masks

            self.controller_hidden = tf.keras.layers.Dense(units=self.controller_size, activation=tf.nn.relu)
            self.controller_hidden.build(input_shape=(1, 1, self.rnn_size+self.latent_space))
            self.controller_hidden.set_weights(new_weights)
        else:
            all_masks += [0,0]


        if new_controller or new_output:
            if new_output:
                assert new_output >= self.output_size
                self.output_size = new_output
            weights = self.controller_output.get_weights()

            del self.controller_output

            new_weights, masks = mutate_output(weights, self.controller_size, self.output_size)
            all_masks += masks

            self.controller_output = tf.keras.layers.Dense(units=self.output_size, activation=None)
            self.controller_output.build(input_shape=(1, 1, self.controller_size))
            self.controller_output.set_weights(new_weights)
        else:
            all_masks += [0,0]

        
        return all_masks # mutation masks to only train new weights


    def evaluate(self, env, vae, n=1, disp=False, get_mutation_gradient=False):
        
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
                with tf.GradientTape() as tape:
                    act, out = self.forward(obs, vae)
                if get_mutation_gradient:
                    gradients = tape.gradient(out, model.trainable_variables)


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

        self.rnn_size = model.rnn_size
        self.controller_size = model.controller_size
        self.latent_space = model.latent_space
        self.output_size = model.output_size

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


        to_pickle.rnn_size = self.rnn_size
        to_pickle.controller_size = self.controller_size
        to_pickle.latent_space = self.latent_space
        to_pickle.output_size = self.output_size

        to_pickle.generation = self.generation
        to_pickle.results = deepcopy(self.results)
        to_pickle.no_rew_early_stop = self.no_rew_early_stop
        to_pickle.rank_hist = deepcopy(self.rank_hist)
        to_pickle.history = deepcopy(self.history)
        return to_pickle


    def save_all(self, name=None, save_path='./ga_ckpt/'):
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


    def load_all(self, name=None, save_path='./ga_ckpt/', latest=False):
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


class DummyModel:

    def __init__(self, no_rew_early_stop=10, rnn_size=256, latent_space=32, controller_size=128, output_size=3):
        self.rnn_size = rnn_size
        self.controller_size = controller_size
        self.latent_space = latent_space
        self.output_size = output_size


        self._epsilon = 0.005

        self.generation = 0
        self.results = {}
        self.no_rew_early_stop = no_rew_early_stop
        self.rank_hist = []
        self.history = {}


    def __repr__(self):
        ret = ''.join(['Model(gen=', str(self.generation), ', eps=', str(self.epsilon), ', no_rew=', str(self.no_rew_early_stop), ', acc=',\
            str(self.mean_result) , ', rnn_size=', str(self.rnn_size), ', controller_size=',\
            str(self.controller_size), ', output_size=', str(self.output_size), ')'])
        return ret


    def add_results(self, rewards):
        if str(self.generation) not in self.results:
            self.results[str(self.generation)] = []
        self.results[str(self.generation)] += rewards
        self.fitness = self.mean_result


    def add_rank(self, r):
        self.rank_hist.append(r)

    def copy_model(self, model, from_pickle=False):
        self._epsilon = model._epsilon

        self.rnn_size = model.rnn_size
        self.controller_size = model.controller_size
        self.latent_space = model.latent_space
        self.output_size = model.output_size

        self.generation = model.generation
        self.results = deepcopy(model.results)
        self.no_rew_early_stop = model.no_rew_early_stop
        self.rank_hist = deepcopy(model.rank_hist) 
        self.history = deepcopy(model.history)

        if not from_pickle:
            self.weights = model.weights

    def get_pickle_obj(self):
        to_pickle = empty_class()
        to_pickle._epsilon = self._epsilon


        to_pickle.rnn_size = self.rnn_size
        to_pickle.controller_size = self.controller_size
        to_pickle.latent_space = self.latent_space
        to_pickle.output_size = self.output_size

        to_pickle.generation = self.generation
        to_pickle.results = deepcopy(self.results)
        to_pickle.no_rew_early_stop = self.no_rew_early_stop
        to_pickle.rank_hist = deepcopy(self.rank_hist)
        to_pickle.history = deepcopy(self.history)
        return to_pickle

    def set_weights(self, weights):
        self.weights = weights
    def get_weights(self):
        return self.weights

    def next_gen(self, new_weights):
        self.weights = new_weights
        self.generation += 1

    @property
    def mean_result(self):
        return sum(self.results[str(self.generation)])/len(self.results[str(self.generation)])

    @property    
    def epsilon(self):
        return self._epsilon
    @epsilon.setter
    def epsilon(self, val):
        self._epsilon = np.clip(val, 0.001, 0.04)