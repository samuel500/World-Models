import numpy as np
import gym
import os
from matplotlib import pyplot as plt
from uuid import uuid4
from multiprocessing import Pool, Process, Pipe

from atari_wrappers import FireResetEnv, WarpFrame

from copy import deepcopy


MAX_FRAMES = 10000
MAX_TRIALS = 100
DIR_NAME = 'record_car_racing'

if not os.path.isdir('./'+DIR_NAME):
    os.mkdir('./'+DIR_NAME)



def wrap_env(env, width=84, height=84, gray_scale=True):
    if hasattr(env.unwrapped, 'get_action_meanings'):
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
    env = WarpFrame(env, width, height, gray_scale)
    return env


def extract(env):

    cpu_c = os.cpu_count()
    print('cpu_c:', cpu_c)
    cpu_c = 2
    with Pool(cpu_c) as pool:
        arg = [(i, env) for i in range(cpu_c)]
        pool.starmap(extract_episodes, arg)

def extract_episodes(id, env, skip=4):

    s = np.random.randint(1e8) 
    np.random.seed(s+id) # or the workers are all choosing the same "random" actions

    def episode(env):
        filename = DIR_NAME+"/"+str(uuid4())[:8]+".npz"
        recording_obs = []
        recording_action = []
        num_actions = env.action_space.n
        
        obs_dim = env.observation_space.shape

        done = False
        frames = 0
        obs = env.reset()
        for i in range(MAX_FRAMES):
            if done:
                #print(id, i)
                frames += i
                break


            if i%skip:
                act = np.random.randint(num_actions)
                obs, rew, done, _ = env.step(act)
                continue

            recording_obs.append(obs)
            act = np.random.randint(num_actions) #env.action_space.sample()

            act_rec = np.eye(num_actions)[act]
            recording_action.append(act_rec)

            obs, rew, done, _ = env.step(act)

        recording_obs = np.array(recording_obs, dtype=np.uint8)
        recording_action = np.array(recording_action, dtype=np.uint8)
        np.savez_compressed(filename, obs=recording_obs, action=recording_action)

        return frames

    tot_frames = 0
    for i in range(MAX_TRIALS):
        f = episode(env)
        tot_frames += f
    print("Worker", id, ':', tot_frames, " frames recorded")
    return tot_frames



def test_disp(env):
    done = False
    env.reset()
    for i in range(MAX_FRAMES):
        if done:
            break

        act = env.action_space.sample()
        obs, rew, done, _ = env.step(act)

        if i == 1000:
            obs = obs.reshape((obs.shape[:-1]))
            print(obs.shape)
            plt.gray()

            plt.imshow(obs, interpolation='nearest')
            plt.show()


if __name__=='__main__':

    #env = gym.make('Pong-v0')
    #env = gym.make('PongNoFrameskip-v4')
    #env = gym.make('PongDeterministic-v4')
    #env = gym.make('Breakout-v0')
    #env = gym.make('BreakoutDeterministic-v4')
    env = gym.make('CarRacing-v0')

    w = 64
    h = 64
    env = wrap_env(env, w, h, gray_scale=False)

    extract(env)

    #test_disp(env)

