
"""
Run this with:
xvfb-run -a -s "-screen 0 1400x900x24" -- python3 extract.py
"""


import numpy as np
import gym
import os
from matplotlib import pyplot as plt
from uuid import uuid4
from multiprocessing import Pool
from atari_wrappers import WarpFrame
import math
from vae import VAE
import tensorflow as tf

from copy import deepcopy
from time import time

MAX_FRAMES = 10000
MAX_TRIALS = 250
DIR_NAME = 'record_car_racing'
DIR_NAME_DQN = 'record_car_racing_DQN'
W = 64
H = 64

if not os.path.exists(DIR_NAME):
    os.mkdir(DIR_NAME)


if not os.path.exists(DIR_NAME_DQN):
    os.mkdir(DIR_NAME_DQN)


def extract(process_n=None):

    if not process_n:
        process_n = os.cpu_count()

    print('Number of parallel processes:', process_n)

    with Pool(process_n) as pool:
        arg = range(process_n)
        pool.map(extract_episodes, arg)

def wrap_env(env, width=84, height=84, gray_scale=False):
    env = WarpFrame(env, width, height, gray_scale)
    return env


def extract_episodes(id, skip_frame=1):

    def episode(env):
        filename = DIR_NAME+"/"+str(uuid4())[:8]+".npz"
        recording_obs = []
        recording_action = []


        num_actions = env.action_space.shape

        obs_dim = env.observation_space.shape

        done = False

        actions = sample_continuous_policy(env.action_space, MAX_FRAMES, 1. / 50)

        frames = 0
        obs = env.reset()
        for i in range(MAX_FRAMES):
            if done:
                frames += i
                break

            if i%skip_frame:
                act = np.random.randint(num_actions)
                obs, rew, done, _ = env.step(act)
                continue

            recording_obs.append(obs)

            act = actions[i]
            recording_action.append(act)

            obs, rew, done, _ = env.step(act)

        recording_obs = np.array(recording_obs, dtype=np.uint8)
        recording_action = np.array(recording_action, dtype=np.uint8)
        np.savez_compressed(filename, obs=recording_obs, action=recording_action)

        return frames

    s = np.random.randint(1e8)
    np.random.seed(s+id) # different seed for each process
    env = gym.make('CarRacing-v0')
    env = wrap_env(env, W, H, gray_scale=False)

    tot_frames = 0
    for i in range(MAX_TRIALS):
        f = episode(env)
        tot_frames += f
    print("Worker", id, ':', tot_frames, " frames recorded")
    return tot_frames


def test_disp(env):
    done = False
    env.reset()
    actions = sample_continuous_policy(env.action_space, MAX_FRAMES, 1. / 50)
    for i in range(MAX_FRAMES):
        env.render()
        if done:
            print(i)
            print('DONE')
            break

        act = env.action_space.sample()
        act = actions[i]
        obs, rew, done, _ = env.step(act)

        if i == 200:
            #obs = obs.reshape((obs.shape[:-1]))
            print(obs.shape)
            #plt.gray()

            plt.imshow(obs, interpolation='nearest')
            plt.show()


def sample_continuous_policy(action_space, seq_len, dt):
    """ Sample a continuous policy.
    Atm, action_space is supposed to be a box environment. The policy is
    sampled as a brownian motion a_{t+1} = a_t + sqrt(dt) N(0, 1).
    :args action_space: gym action space
    :args seq_len: number of actions returned
    :args dt: temporal discretization
    :returns: sequence of seq_len actions
    """
    actions = [action_space.sample()]
    for _ in range(seq_len):
        daction_dt = np.random.randn(len(actions[0]))
        actions.append(
            np.clip(actions[-1] + math.sqrt(dt) * daction_dt,
                    action_space.low, action_space.high))
    return actions


def extract_DQN(process_n=None):

    if not process_n:
        process_n = os.cpu_count()

    print('Number of parallel processes:', process_n)

    with Pool(process_n) as pool:
        arg = range(process_n)
        pool.map(extract_episodes_DQN, arg)



def extract_episodes_DQN(id, skip_frame=1):

    def episode(env, vae_model):
        ACTIONS = [
            (0., 0., 0.),
            (0., 1., 0.),
            (0., 0.5, 0.),
            (1., 0., 0.),
            (-1., 0., 0.),
            (0., 0., 0.6)
        ]

        filename = DIR_NAME_DQN+"/"+str(uuid4())[:8]+".npz"
        recording = []

        num_actions = env.action_space.shape

        obs_dim = env.observation_space.shape


        done = True
        obs = None
        frames = 0
        #print('0')
        for i in range(MAX_FRAMES):
            #print('1')

            if done:
                obs = env.reset()
                #print('2')

                obs = get_latent(obs, vae_model) # batch processing?
                #print('3')

            init_obs = obs

            act = np.random.randint(len(ACTIONS)) #self.env.action_space.sample()
            action = ACTIONS[act]

            obs, rew, done, _ = env.step(action)
            obs = get_latent(obs, vae_model) # batch processing?
            recording.append([init_obs, act, rew, obs, not done])

            if done:
                frames += i
                break
        recording = np.array(recording)
        np.savez_compressed(filename, recording=recording)

        return frames

    latent_size = 32
    vae_model = VAE(latent_size)
    checkpoint_dir = './vae_ckpt/'
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    vae_model.load_weights(latest)


    s = np.random.randint(1e8)
    np.random.seed(s+id) # different seed for each process
    env = gym.make('CarRacing-v0')
    env = wrap_env(env, W, H, gray_scale=False)

    tot_frames = 0
    for i in range(MAX_TRIALS):
        print(id, i)
        f = episode(env, vae_model)
        tot_frames += f
    print("Worker", id, ':', tot_frames, " frames recorded")
    return tot_frames





def get_latent(obs, vae_model):
    obs = obs.astype(np.float32)
    obs /= 255
    obs = obs.reshape((1, *obs.shape))
    obs = vae_model.latent(obs) # batch processing?

    return obs





def test_disp(env):
    done = False
    env.reset()
    actions = sample_continuous_policy(env.action_space, MAX_FRAMES, 1. / 50)
    for i in range(MAX_FRAMES):
        env.render()
        if done:
            print(i)
            print('DONE')
            break

        act = env.action_space.sample()
        act = actions[i]
        obs, rew, done, _ = env.step(act)

        if i == 200:
            #obs = obs.reshape((obs.shape[:-1]))
            print(obs.shape)
            #plt.gray()

            plt.imshow(obs, interpolation='nearest')
            plt.show()







if __name__=='__main__':
    st = time()
    n = 4
    extract_DQN(n)
    print('T:', time()-st)
