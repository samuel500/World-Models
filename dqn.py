
import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import random
import gym
from extract import wrap_env
from vae import VAE
import statistics

# high = (1,1,1)
# low = (-1, 0, 0)

ACTIONS = [
    (0., 0., 0.),
    (0., 1., 0.),
    (0., 0.5, 0.),
    (1., 0., 0.),
    (-1., 0., 0.),
    (0., 0., 0.6)
]


class ExperienceBuffer():

    def __init__(self, size=10000):
        self.size=size
        self.cursor = 0
        self.buffer = []

    def add(self, exp): # to update....???
        if len(self.buffer) < self.size:
            self.buffer.append(exp)
        else:
            self.buffer[self.cursor] = exp
        self.cursor += 1
        if self.cursor == self.size:
            self.cursor = 0

    def sample(self, sample_size, clip_reward=True, exp_stacked=1, with_action=False):

        ret = np.array(random.sample(self.buffer, k=sample_size))
        if clip_reward:
            ret[:,2] = np.clip(ret[:,2], -1, 1)
        return ret


class QNetwork(tf.keras.Model):

    def __init__(self, num_actions, obs_dim, scope='main'):
        super().__init__()

        self.scope = scope
        self.out_dim = num_actions
        self.obs_dim = obs_dim

        self.q = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(obs_dim)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(num_actions)
            ],
            name = 'QNetwork'
        )

    def copy_model_parameters(self, model):
        self.set_weights(model.get_weights())




class DQN():

    def __init__(self, env, vae_model, latent_size=32, lr=1e-3, gamma=0.99, buffer_size=100000,
                 device=None, epoch_steps=5e4, evaluation_runs=5, batch_size=64,
                 target_network_update_freq=2000, epsilon_decay_steps = 5e5,
                 start_buffer=20000,algo='DQ', eval_rate=1):

        self.env = env
        self.vae_model = vae_model

        self.epoch = 0
        self.losss = []
        self.eval_rate = eval_rate
        self.eval_rec = {}
        self.algo = algo

        self.exp_buf = ExperienceBuffer(buffer_size)

        self.gamma = gamma
        self.batch_size = batch_size

        self.evaluation_runs = evaluation_runs
        self.epoch_steps = epoch_steps
        self.num_actions = len(ACTIONS)
        self.obs_dim = latent_size
        self.epsilon_start = 1
        self.epsilon_end = 0.1
        self.epsilon_decay_steps = epsilon_decay_steps

        #self.dueling = False
        #if 'd' in algo:
        #    self.dueling = True

        self.qnet = QNetwork(self.num_actions, self.obs_dim, scope='main')

        self.target_network = QNetwork(self.num_actions, self.obs_dim, scope='target')

        self.target_network_update_freq = target_network_update_freq

        #self.qnet_loss = tf.reduce_mean(tf.losses.huber_loss(predictions = self.qnet.forward, labels=self.qnet.Y, delta=5.0))

        self.qnet_optimizer = tf.keras.optimizers.Adam(lr)

        self.steps = 0
        self.best_train_eval = -99999
        self.best_test_eval = -99999


        self.initialize_buffer_from_files(start_buffer)


    def save_model(self, info='', folder='dqn_ckpt'):
        if not os.path.exists(folder):
            os.mkdir(folder)

        self.qnet.save_weights(os.path.join(folder,'DQN_Epoch{epoch:04d}'.format(epoch=self.epoch)+info))

    '''
    def initialize_buffer(self, steps=20000):
        done = True
        obs = None
        print("\nInitializing buffer:")
        for _ in tqdm(range(steps)):
            if done:
                obs = self.env.reset()
                obs = self.get_latent(obs) # batch processing?

            init_obs = obs

            act = np.random.randint(len(ACTIONS)) #self.env.action_space.sample()
            action = ACTIONS[act]

            obs, rew, done, _ = self.env.step(action)
            obs = self.get_latent(obs) # batch processing?
            self.exp_buf.add([init_obs, act, rew, obs, not done])
    '''


    def initialize_buffer_from_files(self, steps=50000):

        DATA_DIR_DQN = "record_car_racing_DQN"

        filelist = os.listdir(DATA_DIR_DQN)
        filelist.sort()
        #filelist = filelist #[0:1000]

        all_in_one = True

        idx = 0
        buff = []
        l = 0
        for i in range(len(filelist)):
            filename = filelist[i]
            raw_data = np.load(os.path.join(DATA_DIR_DQN, filename),allow_pickle=True)['recording']
            l += len(raw_data)
            if all_in_one:
                buff += list(raw_data)
            else:
                buff.append(list(raw_data))
            if l > steps:
                if all_in_one:
                    buff = buff[:steps]
                else:
                    buff[-1] = buff[-1][:l-steps]
                l = steps
                break
        self.exp_buf.buffer = buff
        #print(self.exp_buf.buffer[5])
        print('inited', l)
        print('exps', len(buff))


    def get_latent(self, obs):
        obs = obs.astype(np.float32)
        obs /= 255
        obs = obs.reshape((1, *obs.shape))
        obs = self.vae_model.latent(obs) # batch processing?

        return obs


    def choose_action(self, obs):
        if random.random() < self.epsilon:
            act = np.random.randint(len(ACTIONS)) #self.env.action_space.sample()
        else:
            action = self.qnet.q([obs])
            act = np.argmax(action[0])
        return act


    def sample_exp(self, n=1):
        act = self.choose_action(self.obs)
        action = ACTIONS[act]
        obs, rew, done, _ = self.env.step(action)
        obs = self.get_latent(obs) # batch processing?

        #obs = self.env.reset()
        #obs = self.get_latent(obs) # batch processing?

        for e in range(n):
            if done:
                obs = self.env.reset()
            init_obs = obs
            act = self.choose_action(obs)
            action = ACTIONS[act]
            obs, rew, done, _ = self.env.step(action)
            obs = self.get_latent(obs) # batch processing?

            #tot_rew += rew
            self.exp_buf.add([init_obs, act, rew, obs, not done])

            if done:
                obs = self.env.reset()



    def train_epoch(self):

        self.epoch += 1
        i = 0
        print("====Epoch:", self.epoch, "====")
        print('steps', self.steps)
        print('epsilon', self.epsilon)

        epoch_losss = []

        rew_list = []
        diff_list = []
        with tqdm(total=self.epoch_steps) as pbar:

            while i < self.epoch_steps:

                done = False
                tot_rew = 0

                #obs = self.env.reset()
                #obs = self.get_latent(obs) # batch processing?


                while (not done) and (i < self.epoch_steps):

                    step_num = i + (self.epoch-1)*self.epoch_steps


                    '''
                    init_obs = obs
                    act = self.choose_action(obs)
                    action = ACTIONS[act]
                    obs, rew, done, _ = self.env.step(action)
                    obs = self.get_latent(obs) # batch processing?

                    #tot_rew += rew
                    self.exp_buf.add([init_obs, act, rew, obs, not done])

                    if done:
                        obs = self.env.reset()
                    '''


                    if not (self.steps)%self.target_network_update_freq:
                        self.target_network.copy_model_parameters(self.qnet)

                    # training

                    sample = self.exp_buf.sample(self.batch_size)

                    DDQN = False
                    if 'D' in self.algo:
                        DDQN = True

                    verb = False

                    if DDQN:
                        nn_input = np.stack(sample[:,3])
                        nextqs = self.qnet.q(nn_input, training=False)
                        nextqs = np.array(nextqs)

                        maxq_index = np.argmax(nextqs, axis=1)



                    if np.isnan(nextqs).any():
                            print('nan at', step_num)
                            print(nextqs)
                            print(dqn.qnet.q.layers[1].weights)
                            print(dqn.qnet.q.layers[2].weights)
                            raise


                    nextqs = self.target_network.q(np.stack(sample[:,3]), training=False)
                    nextqs = np.array(nextqs)

                    if DDQN:
                        target_q = nextqs[range(len(nextqs)), maxq_index]
                    else:
                        target_q = np.amax(nextqs, 1)

                    ys = sample[:,2] + sample[:,4] * self.gamma * target_q


                    action_mask = np.eye(self.num_actions)[sample[:,1].astype(int)]

                    ys = np.multiply(ys, action_mask.T).T
                    ys = ys.astype(np.float32)
                    action_mask = action_mask.astype(np.float32)
                    action_mask = tf.constant(action_mask)
                    #print('ys', ys)
                    #print(action_mask.dtype)
                    compute_apply_gradients(self.qnet, np.stack(sample[:,0]), ys, self.qnet_optimizer, action_mask)

                    #epoch_losss.append(loss)

                    self.steps += 1
                    i += 1
                    pbar.update(1)

        #mean_acc = statistics.mean(rew_list)
        #print("\nAvg rew:", mean_acc)
        #if self.best_train_eval < mean_acc:
        #    self.best_train_eval = mean_acc
        #print("losss:", statistics.mean(list(map(float, epoch_losss))))
        #self.losss += epoch_losss
        if not self.epoch%self.eval_rate:
            results = self.run_evaluation()
            self.eval_rec[self.epoch] = results
            mean_acc = statistics.mean(results)
            if self.best_test_eval < mean_acc:
                self.best_test_eval = mean_acc
                info = 'acc:' + str(mean_acc)
                self.save_model(info)


    def run_evaluation(self, evaluation_runs=5):
        done = False
        obs = None

        rs = []
        for r in tqdm(range(evaluation_runs)):
            tot_rew = 0
            obs = self.env.reset()
            obs = self.get_latent(obs) # batch processing?

            done = False
            noopAct = random.randint(0,30)
            for i in range(len(self.qnet.layers)):
                print(self.qnet.q.layers[i])

            for i in range(100000):
                self.env.render()
                if done:
                    break

                action = self.qnet.q([obs], training=False)

                #print('obs:', obs)
                #print(action)


                act = np.argmax(action[0])
                action = ACTIONS[act]

                obs, rew, done, _ = self.env.step(action)
                obs = self.get_latent(obs) 

                tot_rew += rew
            rs.append(tot_rew)
        print("test rewards:", rs)
        return rs


    def display_agent(self):
        import io
        import base64
        from IPython.display import HTML
        uid = self.env.unwrapped.spec.id + '-' + 'Epoch' + str(self.epoch)
        env = gym.wrappers.Monitor(self.env, "./gym-results", force=True, uid=uid)

        obs = env.reset()
        done = False
        noopAct = random.randint(0,30)
        for i in range(50000):
            if done:
                break

            if i < noopAct: # https://arxiv.org/pdf/1511.06581.pdf
                    act = 0 # env.action_space.sample()
            else:
                action = self.qnet.q([obs])
                act = ACTIONS[np.argmax(action[0])]

            obs, rew, done, _ = env.step(act)

        env.close()

        return env


    @property
    def epsilon(self):
        eps = self.epsilon_start - (self.epsilon_start-self.epsilon_end) * self.steps / self.epsilon_decay_steps
        eps = max(self.epsilon_end, eps)
        return eps

W = 64
H = 64

env = gym.make('CarRacing-v0')
env = wrap_env(env, W, H, gray_scale=False)

@tf.function
def compute_loss(model, x, yh, mask):
    err = tf.keras.losses.Huber(delta=1.0)

    y = model.q(x, training=True)
    #print(y.shape)
    #print(x.shape)
    y *= mask
    #rec_loss = tf.reduce_mean(tf.math.square(yh - y))
    rec_loss = err(yh, y)
    return rec_loss

@tf.function
def compute_apply_gradients(model, x, yh, optimizer, mask):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, yh, mask)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


if __name__=='__main__':
    latent_size = 32
    vae_model = VAE(latent_size)
    checkpoint_dir = './vae_ckpt/'
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    vae_model.load_weights(latest)
    #for i in range(len(vae_model.encoder.layers)):
        #print(dir(vae_model.encoder.layers[i]))
    #    vae_model.encoder.layers[i].trainable = False


    dqn_args = {
        "env": env,
        'vae_model': vae_model,
        'lr':8e-4,
        'buffer_size':2500000,
        'epoch_steps':10e4, #5e4,
        'gamma': 0.99,
        'target_network_update_freq': 1000,
        'epsilon_decay_steps': 5e5,
        'start_buffer': 1800000,
        'batch_size': 32,

    }


    dqn = DQN(**dqn_args)
    #dqn.run_evaluation()
    epochs = 100
    for e in range(epochs):
        print(e)
        #dqn.run_evaluation(1)
        dqn.train_epoch()
