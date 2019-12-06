import numpy as np 
import tensorflow as tf
import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import random
from copy import deepcopy
import pickle

FRAME_SIZE = 64

class empty_class:
    pass



class RisiModel(tf.keras.Model):

    def __init__(self, no_rew_early_stop=10, rnn_size=256, latent_space=32, controller_size=128, output_size=3, hada=False):
        super().__init__()
        self.rnn_size = rnn_size
        self.controller_size = controller_size
        self.latent_space = latent_space
        self.output_size = output_size

        self.hada = hada
        
        self.rnn = tf.keras.layers.LSTM(rnn_size, return_state=True)
        self.rnn.build(input_shape=(1, 1, 32+3))
        self.controller = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(rnn_size+latent_space,)),
                tf.keras.layers.Dense(units=controller_size, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=output_size, activation=None)
            ],
            name='controller'
        )

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(FRAME_SIZE, FRAME_SIZE, 3)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=4, strides=2, activation='relu'), # 31x31
                #tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=4, strides=2, activation='relu'), # 14x14
                #tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=4, strides=2, activation='relu'), # 6x6
                #tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=4, strides=2, activation='relu'), # 2x2
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_space + latent_space),
            ],
            name = 'encoder'
        )

        self._epsilon = 0.012
        self._both_mutation_p = 0.9
        self._controller_mutation_p = 0.5
        self._encoder_mutation_p = 0.95

        self.generation = 0
        self.results = {}
        self.no_rew_early_stop = no_rew_early_stop
        self.rank_hist = []
        self.history = {}


    def __repr__(self):
        ret = ''.join(['Model(gen=', str(self.generation), ', eps=', str(self.epsilon), ', both_p=', str(self.both_mutation_p),\
            ', controller_p=', str(self.controller_mutation_p), ', no_rew=', str(self.no_rew_early_stop) ,', acc=',\
            str(self.mean_result) , ', rnn_size=', str(self.rnn_size), ', controller_size=',\
            str(self.controller_size), ', output_size=', str(self.output_size), ')'])
        return ret


    def encode(self, x, training=True):
        mu, logvar = tf.split(self.encoder(x, training=training), num_or_size_splits=2, axis=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        eps = tf.random.normal(shape=mu.shape)
        return eps * tf.exp(logvar * .5) + mu

    def latent(self, x):
        mu, logvar = self.encode(x, training=False)
        return self.reparameterize(mu, logvar)

    def forward(self, obs):
        if self.prev_act is None:
            self.prev_act = np.array([0,0,0])

        obs = obs.astype(np.float32)
        obs /= 255
        obs = obs.reshape((1, *obs.shape))

        z = self.latent(obs)

        rnn_in = tf.concat([z[0], self.prev_act], 0)
        rnn_in = tf.reshape(rnn_in, (1,1, *rnn_in.shape))
        if self.h is None:
            _, self.h, self.c = self.rnn(rnn_in)
        else:
            _, self.h, self.c = self.rnn(rnn_in, initial_state=[self.h, self.c])

        c_in = tf.concat([z[0],self.h[0]], 0) # batch D?
        c_in = tf.reshape(c_in, (1, *c_in.shape))
        out = self.controller(c_in)
        acts = np.array(out)[0]

        acts = np.tanh(acts)
        act = np.array([0.,0.,0.])
        for i in range(0,self.output_size,3):
            act += acts[i:i+3]
        act /= int(self.output_size/3)


        if self.hada:
            print('wowow')
            act[1] = (act[1]+1)/2
            act[2] = np.clip(act[2], 0, 1)

        return act, out


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
        self.encoder_mutation_p += np.random.choice([0.05,0.05,-0.05,-0.05] + 14*[0])


        if random.random() < self.both_mutation_p:
            self.controller.set_weights(mutate_weights(self.controller))
            #print('rnn', self.rnn.get_weights())
            self.rnn.set_weights(mutate_weights(self.rnn))
            self.encoder.set_weights(mutate_weights(self.encoder))
        else:
            if random.random() < self.controller_mutation_p:
                self.controller.set_weights(mutate_weights(self.controller))
            else:
                self.rnn.set_weights(mutate_weights(self.rnn))
            if random.random() < self.encoder_mutation_p:
            	self.encoder.set_weights(mutate_weights(self.encoder))


        self.generation += 1


    def evaluate(self, env, n=1, disp=False):
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

                act, _ = self.forward(obs)
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
        to_pickle._both_mutation_p = self._both_mutation_p
        to_pickle._controller_mutation_p = self._controller_mutation_p

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
    @property    
    def encoder_mutation_p(self):
        return self._encoder_mutation_p
    @encoder_mutation_p.setter
    def encoder_mutation_p(self, val):
        self._encoder_mutation_p = np.clip(val, 0.05, 0.95)

