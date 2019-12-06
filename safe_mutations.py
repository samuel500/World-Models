

import numpy as np 
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import random
from copy import deepcopy
import pickle
import gym
from model2 import Model2


from atari_wrappers import WarpFrame
import random

from vae import VAE
from copy import deepcopy
from operator import attrgetter

from time import time

from model import Model, empty_class
from utils import *

@tf.function
def compute_apply_gradients(model, x):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, beta)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))




def test():
    env = gym.make('CarRacing-v0', verbose=0)
    env = wrap_env(env, W, H, gray_scale=False)
    latent_size = 32
    vae = VAE(latent_size)
    checkpoint_dir = './vae_ckpt/'
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    vae.load_weights(latest)

    disp = False
    #model_dir = './ga_ckpt/'
    #latest = tf.train.latest_checkpoint(model_dir)
    latest = './saved/0236result_643.5598879468932'
    #latest = './saved/0179result_431.7866706964803'
    print(latest)
    #model = Model2(rnn_size=256, controller_size=128, output_size=3, hada=False)
    model = Model(rnn_size=256, controller_size=128, output_size=3)

    model.load_weights(latest).expect_partial()

    done = False
    obs = env.reset()
    last_rew = 0
    tot_rew = 0

    model.h = None
    model.c = None
    model.prev_act = None

    last_rew_all = []

    st = time()

    for i in range(100000):
        if done:
            print(i)
            break
        if disp:
            env.render()


        with tf.GradientTape() as tape:
            act, obs = model.forward(obs, vae)
        #act = env.action_space.sample() #[0,0,0]
        #print(act)
            #print(act)
        gradients = tape.gradient(obs, model.trainable_variables)
        #for g in gradients:
        #    print(g) #.numpy())


        #print(act)
        model.prev_act = act
        obs, rew, done, _ = env.step(act)
        tot_rew += rew
        if rew < 0:
            last_rew += 1
        else:
            last_rew_all.append(last_rew)
            last_rew = 0

    print(time()-st)
    env.close()
    raise




if __name__=='__main__':
    test()
