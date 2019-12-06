import numpy as np 
import tensorflow as tf
import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt


class VAE(tf.keras.Model):

    def __init__(self, latent_dim):
    
        super().__init__()
    
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(FRAME_SIZE, FRAME_SIZE, 3)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=4, strides=2, activation='relu'), # 31x31
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=4, strides=2, activation='relu'), # 14x14
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=4, strides=2, activation='relu'), # 6x6
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=4, strides=2, activation='relu'), # 2x2
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ],
            name = 'encoder'
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=2**2*256, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(-1, 1, 2**2*256)),
                tf.keras.layers.Conv2DTranspose(
                    filters=128,
                    kernel_size=5,
                    strides=2,
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=5,
                    strides=2,
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=6,
                    strides=2,
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=3,
                    kernel_size=6,
                    strides=2,
                    activation='sigmoid'),
               
            ],
            name = 'decoder'
        )

    def call(self, x, training=True):
        mu, logvar = self.encode(x, training)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)


    def encode(self, x, training=True):
        mu, logvar = tf.split(self.encoder(x, training=training), num_or_size_splits=2, axis=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        eps = tf.random.normal(shape=mu.shape)
        return eps * tf.exp(logvar * .5) + mu

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        return logits # not logits

    def latent(self, x):
        mu, logvar = self.encode(x, training=False)
        return self.reparameterize(mu, logvar)

    def test(self, x):
        n = 5
        samples = np.random.choice(len(x), n**2, replace=False)
        samples.sort()
        x = x[samples,:]
        preds = self.call(x, training=False)
        preds = np.array(preds)
        #print(preds)
        imd = FRAME_SIZE
        canvas_orig = np.empty((imd*n , 2*imd * n+1, 3))
        for i in range(n):
            batch_x = x[i*n:i*n+n]
            g = preds[i*n:i*n+n]

            for j in range(n):
                canvas_orig[i * imd:(i + 1) * imd, j * imd:(j + 1) * imd] = \
                    batch_x[j].reshape([imd, imd, 3])
                canvas_orig[i * imd :(i + 1) * imd, j * imd + n*imd+1:(j + 1) * imd + n*imd+1] = \
                    g[j].reshape([imd, imd, 3])
        canvas_orig[:, n*imd:n*imd+1] = 1

        print("Original Images")
        plt.figure(figsize=(n*2+1, n))
        plt.imshow(canvas_orig, origin="upper")
        plt.draw()
        plt.show()


def create_dataset(filelist, N=100, M=10000, T=1000): # N is 10000 episodes, M is number of timesteps

    test_data = np.zeros((T, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    filename = filelist[-1]

    raw_data = np.load(os.path.join(DATA_DIR, filename))['obs']
    if T > len(raw_data):
        print('T too large, check test dataset creation function')
        T = len(raw_data)
    test_data = raw_data[:T]

    data = np.zeros((M*N, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    idx = 0
    for i in range(N):
        filename = filelist[i]
        raw_data = np.load(os.path.join(DATA_DIR, filename))['obs']
        l = len(raw_data)
        if (idx+l) > (M*N):
            data = data[:idx]
            print('premature break')
            break
        l = min(l, M)
        data[idx:idx+l] = raw_data[:l]
        
        idx += l
        if ((i+1) % 100 == 0):
            print("loading file", i+1)
    data = data[:idx]

    return data, test_data


@tf.function
def compute_loss(model, x, beta):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    y = model.decode(z)

    #rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=x) # MSE/L2?
    #rec_loss = -tf.math.reduce_sum(rec_loss) # , axis=[1, 2, 3])

    rec_loss = tf.reduce_sum(tf.math.square(x - y))
    rec_loss = tf.reduce_mean(rec_loss)

    kl_loss =  -0.5*tf.math.reduce_sum((1 + logvar - tf.square(mean) - tf.exp(logvar)))
    # https://openreview.net/forum?id=Sy2fzU9gl
    kl_loss *= beta[0] 
    kl_loss = tf.reduce_mean(kl_loss)    

    return tf.reduce_mean(kl_loss+rec_loss)

@tf.function
def compute_apply_gradients(model, x, optimizer, beta):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, beta)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

FRAME_SIZE = 64



if __name__=='__main__':
    #model = VAE(32)
    #model.summary()
    #raise
    print(tf.__version__)

    NUM_EPOCH = 10
    DATA_DIR = "record_car_racing"


    model_save_path = "vae_ckpt"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)


    batch_size = 128
    filelist = os.listdir(DATA_DIR)
    filelist.sort()
    filelist = filelist #[0:1000]



    dataset, test_data = create_dataset(filelist, N=5)

    total_length = len(dataset)
    print(total_length)
    num_batches = int(np.floor(total_length/batch_size))
    print("num_batches", num_batches)


    optimizer = tf.keras.optimizers.Adam()

    model = VAE(32)
    model.summary()

    test_data = test_data.astype(np.float32)
    test_data /= 255
    model.test(test_data)

    epochs = 10

    kl_warm_up_epochs = 1 


    for epoch in range(1, epochs + 1):
        beta = np.array([1], dtype=np.float32)
        print(epoch)
        st = time.time()
        for batch in tqdm(range(num_batches)):

            if epoch <= kl_warm_up_epochs:
                # KL Warm Up
                # https://arxiv.org/abs/1602.02282
                beta = np.array([(batch + (epoch-1)*num_batches) / (num_batches*kl_warm_up_epochs)], dtype=np.float32)

            train_x = dataset[batch*batch_size:(batch+1)*batch_size]
            train_x = train_x.astype(np.float32)
            train_x /= 255

            compute_apply_gradients(model, train_x, optimizer, beta)

        model.test(test_data)

        model.save_weights(os.path.join(model_save_path,'VAE_Epoch{epoch:04d}'.format(epoch=epoch)))

        print('T:', time.time()-st)

