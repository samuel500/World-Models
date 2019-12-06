

import numpy as np 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool


from time import time
import gym
from gym import wrappers
from atari_wrappers import WarpFrame
import random

from vae import VAE
from copy import deepcopy
from operator import attrgetter


from model import Model, empty_class
from ga import wrap_env, W, H

from merger import merge_all



def test(env, models, vae, disp=True, n=1, avg=True):
    no_rew_early_stop = 20
    rewards = []
    for r in range(n):
        #self.rnn.reset_states()
        done = False
        obs = env.reset()
        last_rew = 0
        tot_rew = 0
        for model in models:
            model.h = None
            model.c = None
            model.prev_act = None

        last_rew_all = []

        for i in range(100000):
            if done:
                break
            if disp:
                env.render()

            acts = np.array([0.,0.,0.])
            for model in models:
                act, _ = model.forward(obs, vae)
                acts += np.array(act)
            
            if avg:
                act = acts/len(models)
            #print(act)
            for model in models:
                model.prev_act = act
            obs, rew, done, _ = env.step(act)
            tot_rew += rew
            if rew < 0:
                last_rew += 1
            else:
                last_rew_all.append(last_rew)
                last_rew = 0

            #if last_rew > no_rew_early_stop:
                #print('break @', i)
            #    break
        print(tot_rew)
        rewards.append(tot_rew)
        #print(last_rew_all)
        #print('avg:', sum(last_rew_all)/len(last_rew_all))
    print(rewards)
    for model in models:
            model.h = None
            model.c = None
            model.prev_act = None

    return rewards

if __name__=='__main__':
    latent_size = 32
    vae_model = VAE(latent_size)

    checkpoint_dir = './vae_ckpt/'
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    vae_model.load_weights(latest)


    env = gym.make('CarRacing-v0', verbose=0)
    env = wrap_env(env, W, H, gray_scale=False)
    env = wrappers.Monitor(env, './disp_dir/', force = True)


    model_dir = './ga_ckpt/'
    latest = tf.train.latest_checkpoint(model_dir)
    #latest = model_dir + '0034result:508.422078980511'
    latest = './saved/0094result_533.6143111305319'
    latest = './saved/0132result_600.4132167051537'
    #latest = './saved/0133result_617.943684555891'
    latest = './saved/0134result_669.6144042101097'
    latest = './saved/0151result_689.5666320438731'
    #latest = './saved/0192result_748.0616309675287'
    #latest = './saved/0211result_477.64911357840657'
    latest = './saved/0200result_728.6064873166086'
    
    #latest = './saved/0236result_643.5598879468932'
    #model = Model()
    #model.load_weights(latest)
    model_0 = Model(rnn_size=416, controller_size=208, output_size=39)
    model_0.load_weights(latest)
    
    rew = test(env, [model_0], vae_model, disp=True, n=50, avg=False)
    print('avg:', np.mean(rew))
    print('std', np.std(rew))
    env.close()


    #raise

    #latest = './saved/0177result_421.18715488163565'
    #latest = './saved/0179result_431.7866706964803'
    #latest = './saved/0198result_466.7534881787524'
    #latest = './saved/0029result_224.353128915717'
    #latest = './saved/0042result_271.5974076446249'
    #latest = './saved/0055result_317.1206674220253'
    #latest = './saved/0016result_42.65352757793934'
    #latest = './saved/0041result_244.02468926077972'
    #latest = './saved/0049result_140.6199458263877'
    #latest = './saved/0051result_302.1311724949909'
    #latest = './saved/0056result_363.32324368269093'
    #latest = './saved/0081result_422.07389722683615'

    
    latest = '0086result_443.71515983197327'
    #latest = '0096result_138.407305135033'
    #latest = '0104result_202.55477209905018'
    #latest = '0124result_503.35098973857504'
    #latest = '0219result_485.08431904575394'
    latest = '0236result_643.5598879468932'
    #latest = '0254result_578.2577006758438'
    latest = './saved/'+latest



    m1 = './saved/0254result_578.2577006758438'
    m2 = './saved/0236result_643.5598879468932'


    #model3 = Model(rnn_size=512)
    #for w in model3.get_weights():
    #    print(w.shape)


    #print(latest)
    #model1 = Model()
    #model2 = Model()
    #model1.load_weights(m1)
    #model2.load_weights(m2)

    #for w in model1.get_weights():
    #    print(w.shape)

    #'./saved/0030result_488.1702374970796'

    list_models = [
        
        #'./saved/0254result_578.2577006758438',
        #'./saved/0297result_372.95431161691926',



        #'./saved/0236result_643.5598879468932',
        #'./saved/0081result_422.07389722683615',

        #'./saved/0177result_421.18715488163565',

        #'./saved/0198result_466.7534881787524',
        './saved/0179result_431.7866706964803',
        #'./saved/0293result_285.57063288357404',
        #'./saved/0124result_503.35098973857504',
        #'./saved/0301result_395.1973162218356',
        #'./saved/0060result_425.47869352868946',
        #'./saved/0072result_211.4064480425836',
        #'./saved/0088result_343.65973280883276'

        
    

        
    ]
    small_models = [
        #'./small_net/0024result_73.37881187283189',
        #'./small_net/0023result_68.10649474462564',
        #'./small_net/0030result_112.49440075016027',
        

        #'./small_net/0005result_53.89669391302787',
        #'./small_net/0006result_44.271681554860244',
        #'./small_net/0007result_51.929937191616',
        #'./small_net/0008result_53.590599733443554',

        #'./small_net/0031result_75.47013714182765',
        #'./small_net/0024result_143.51840085309573',
        #'./small_net/0033result_126.29295347816975',
        #'./small_net/0010result_105.4385020211593',
        
        #'./small_net/0013result_89.42065495569074',
        
        #'./small_net/0030result_58.72030268151148',
        #'./small_net/0031result_80.75889247937188',
        #'./small_net/0032result_59.99984922062008',


    ]
    tiny_models = [
        #'./tiny_net/0006result_41.194774974355475',
        #'./tiny_net/0007result_41.008968948883236',
        #'./tiny_net/0008result_44.64426830360883',
        #'./tiny_net/0028result_50.310277881290006',
        #'./tiny_net/0029result_44.55096876655588',
        #'./tiny_net/0030result_53.65870533691165',
        #'./tiny_net/0031result_51.903937738560536',
    ]

    #print(latest)
    s_models = [Model(rnn_size=32, controller_size=16) for m in small_models] 
    b_models = [Model(output_size=3) for m in list_models]
    t_models = [Model(rnn_size=16, controller_size=12) for m in tiny_models]

    for i in range(len(s_models)):
        s_models[i].load_weights(small_models[i])
    for i in range(len(b_models)):
        b_models[i].load_weights(list_models[i])
    for i in range(len(t_models)):
        t_models[i].load_weights(tiny_models[i])


    #for w in s_models[0].get_weights():
    #    print(w.shape)
    #    print(w)
    #    break
    #name = './saved/0132result_500.64363311437586'
    #new_model = Model(rnn_size=1024, controller_size=512, output_size=48)
    #new_model.load_weights(name)
    #models_new = [new_model]

    models = b_models + s_models + t_models #+ [model_0]

    tot_rnn = sum([m.rnn_size for m in models])
    tot_con = sum([m.controller_size for m in models])
    tot_out = sum([m.output_size for m in models])

    ensemble_model = Model(rnn_size=tot_rnn, controller_size=tot_con, output_size=tot_out)
    #for i, w in enumerate(ensemble_model.get_weights()):
    #    print(w)


    ensemble_model.set_weights(merge_all([m.get_weights() for m in models]))
    models = [ensemble_model, model_0]

    #name = './saved/0090result_680.0453514739107'
    
    

    rew = test(env, models, vae_model, disp=True, n=40, avg=True)

    print('avg:', np.mean(rew))
    print('std', np.std(rew))
    env.close()



    #model.load_all(latest=True)



    

