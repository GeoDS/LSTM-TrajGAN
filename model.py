import tensorflow as tf
import keras
import numpy as np
import random

random.seed(2020)
np.random.seed(2020)
tf.random.set_random_seed(2020)

from keras.layers import Input, Add, Average, Dense, LSTM, Lambda, TimeDistributed, Concatenate, Embedding
from keras.initializers import he_uniform
from keras.regularizers import l1

from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

from losses import d_bce_loss, trajLoss

class LSTM_TrajGAN():
    def __init__(self, latent_dim, keys, vocab_size, max_length, lat_centroid, lon_centroid, scale_factor):
        self.latent_dim = latent_dim
        self.max_length = max_length
        
        self.keys = keys
        self.vocab_size = vocab_size
        
        self.lat_centroid = lat_centroid
        self.lon_centroid = lon_centroid
        self.scale_factor = scale_factor
        
        self.x_train = None
        
        # Define the optimizer
        self.optimizer = Adam(0.001, 0.5)

        # Build the trajectory generator
        self.generator = self.build_generator()

        # The trajectory generator takes real trajectories and noise as inputs
        noise = Input(shape=(self.latent_dim,), name='input_noise')
        inputs = []
        for idx, key in enumerate(self.keys):
            i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
            inputs.append(i)
        inputs.append(noise)
        
        # The trajectory generator generates synthetic trajectories
        gen_trajs = self.generator(inputs)
        
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=d_bce_loss(gen_trajs[4]), optimizer=self.optimizer, metrics=['accuracy'])

        # The combined model only trains the trajectory generator
        self.discriminator.trainable = False

        # The discriminator takes generated trajectories as input and makes predictions
        pred = self.discriminator(gen_trajs[:4])

        # The combined model (combining the generator and the discriminator)
        self.combined = Model(inputs, pred)
        self.combined.compile(loss=trajLoss(inputs, gen_trajs), optimizer=self.optimizer)
        
        C_model_json = self.combined.to_json()
        with open("params/C_model.json", "w") as json_file:
            json_file.write(C_model_json)
            
        G_model_json = self.generator.to_json()
        with open("params/G_model.json", "w") as json_file:
            json_file.write(G_model_json)
        
        D_model_json = self.discriminator.to_json()
        with open("params/D_model.json", "w") as json_file:
            json_file.write(D_model_json)

    def build_discriminator(self):
        
        # Input Layer
        inputs = []
        
        # Embedding Layer
        embeddings = []
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                continue
            if key == 'lat_lon':
                i = Input(shape=(self.max_length, self.vocab_size[key]),
                          name='input_' + key)

                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=64, use_bias=True, activation='relu', kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_latlon = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_latlon)

            else:
                i = Input(shape=(self.max_length,self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=self.vocab_size[key], use_bias=True, activation='relu', kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_attr = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_attr)
            inputs.append(i)
            embeddings.append(e)
            
        # Feature Fusion Layer
        concat_input = Concatenate(axis=2)(embeddings)
        unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(concat_input)
        d = Dense(units=100, use_bias=True, activation='relu', kernel_initializer=he_uniform(seed=1), name='emb_trajpoint')
        dense_outputs = [d(x) for x in unstacked]
        emb_traj = Lambda(lambda x: tf.stack(x, axis=1))(dense_outputs)
        
        # LSTM Modeling Layer (many-to-one)
        lstm_cell = LSTM(units=100, recurrent_regularizer=l1(0.02))(emb_traj)
        
        # Output
        sigmoid = Dense(1, activation='sigmoid')(lstm_cell)

        return Model(inputs=inputs, outputs=sigmoid)

    def build_generator(self):
        
        # Input Layer
        inputs = []
        
        # Embedding Layer
        embeddings = []
        noise = Input(shape=(self.latent_dim,), name='input_noise')
        mask = Input(shape=(self.max_length, 1), name='input_mask')
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                inputs.append(mask)
                continue
            elif key == 'lat_lon':
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=64, activation='relu', use_bias=True, kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_latlon = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_latlon)
            else:
                i = Input(shape=(self.max_length,self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=self.vocab_size[key], activation='relu', use_bias=True, kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_attr = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_attr)
            inputs.append(i)
            embeddings.append(e)
        inputs.append(noise)
        
        # Feature Fusion Layer
        concat_input = Concatenate(axis=2)(embeddings)
        unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(concat_input)
        d = Dense(units=100, use_bias=True, activation='relu', kernel_initializer=he_uniform(seed=1), name='emb_trajpoint')
        dense_outputs = [d(Concatenate(axis=1)([x, noise])) for x in unstacked]
        emb_traj = Lambda(lambda x: tf.stack(x, axis=1))(dense_outputs)
        
        # LSTM Modeling Layer (many-to-many)
        lstm_cell = LSTM(units=100,
                        batch_input_shape=(None, self.max_length, 100),
                        return_sequences=True,
                        recurrent_regularizer=l1(0.02))(emb_traj)
        
        # Outputs
        outputs = []
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                output_mask = Lambda(lambda x: x)(mask)
                outputs.append(output_mask)
            elif key == 'lat_lon':
                output = TimeDistributed(Dense(2, activation='tanh'), name='output_latlon')(lstm_cell)
                scale_factor = self.scale_factor
                output_stratched = Lambda(lambda x: x * scale_factor)(output)
                outputs.append(output_stratched)
            else:
                output = TimeDistributed(Dense(self.vocab_size[key], activation='softmax'), name='output_' + key)(lstm_cell)
                outputs.append(output)
                
        return Model(inputs=inputs, outputs=outputs)

    def train(self, epochs=200, batch_size=256, sample_interval=10):
        
        # Training data
        x_train = np.load('data/final_train.npy',allow_pickle=True)
        self.x_train = x_train

        # Padding zero to reach the maxlength
        X_train = [pad_sequences(f, self.max_length, padding='pre', dtype='float64') for f in x_train]
        self.X_train = X_train
        
        
        
        # Originally, we refer to a Keras implementation of GAN that trains on one random batch per epoch.
        # This is a bit confusing as an epoch here is more like an iteration rather than going through all the samples.
        # Doing this helps converge the model too, but the number of epochs it takes would be naturally large.
        # More details and explanations regarding this implementation:
        # https://github.com/eriklindernoren/Keras-GAN/issues/9
        # https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py
        
        # We also implement a regular training loop that trains on all samples (multiple batches) per epoch.
        # Please commment the first for-loop and uncomment the second for-loop to switch to the regular training loop.
        # The regular one usually takes fewer epochs to converge (around 200 when the batch size is 256).
        
        # Training the model
        # The original training loop that trains on one random batch per epoch
        for epoch in range(1,epochs+1):

            # Select a random batch of real trajectories
            idx = np.random.randint(0, X_train[0].shape[0], batch_size)
            
            # Ground truths for real trajectories and synthetic trajectories
            real_bc = np.ones((batch_size, 1))
            syn_bc = np.zeros((batch_size, 1))

            real_trajs_bc = []
            real_trajs_bc.append(X_train[0][idx]) # latlon
            real_trajs_bc.append(X_train[1][idx]) # day
            real_trajs_bc.append(X_train[2][idx]) # hour
            real_trajs_bc.append(X_train[3][idx]) # category
            real_trajs_bc.append(X_train[4][idx]) # mask
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            real_trajs_bc.append(noise) # noise

            # Generate a batch of synthetic trajectories
            gen_trajs_bc = self.generator.predict(real_trajs_bc)

            # Train the discriminator
            # No mask and noise are used
            d_loss_real = self.discriminator.train_on_batch(real_trajs_bc[:4], real_bc)
            d_loss_syn = self.discriminator.train_on_batch(gen_trajs_bc[:4], syn_bc)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_syn)

            # Train the generator
            # Mask and noise are used
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            real_trajs_bc[5] = noise
            g_loss = self.combined.train_on_batch(real_trajs_bc, real_bc)
            
            print("[%d/%d] D Loss: %f | G Loss: %f" % (epoch, epochs, d_loss[0], g_loss))

            # Print and save the losses/params
            if epoch % sample_interval == 0:
                self.save_checkpoint(epoch)
                print('Model params saved to the disk.')
        
        # Training the model
        # The regular training loop that trains on all samples (multiple batches) per epoch
#         for epoch in range(1,epochs+1):
            
#             random_indices = np.random.permutation(X_train[0].shape[0])
            
#             num_batches = np.ceil(random_indices.shape[0]/batch_size).astype(np.int)
            
#             for i in range(num_batches):
            
#                 # Select a random batch of real trajectories
#                 idx = random_indices[batch_size*i:batch_size*(i+1)]
                
#                 # Ground truths for real trajectories and synthetic trajectories
#                 real_bc = np.ones((idx.shape[0], 1))
#                 syn_bc = np.zeros((idx.shape[0], 1))
                
#                 # Random noise
#                 noise = np.random.normal(0, 1, (idx.shape[0], self.latent_dim))

#                 real_trajs_bc = []
#                 real_trajs_bc.append(X_train[0][idx]) # latlon
#                 real_trajs_bc.append(X_train[1][idx]) # day
#                 real_trajs_bc.append(X_train[2][idx]) # hour
#                 real_trajs_bc.append(X_train[3][idx]) # category
#                 real_trajs_bc.append(X_train[4][idx]) # mask
#                 real_trajs_bc.append(noise) # random noise

#                 # Generate a batch of synthetic trajectories
#                 gen_trajs_bc = self.generator.predict(real_trajs_bc)

#                 # Train the discriminator
#                 # No mask and noise are used
#                 d_loss_real = self.discriminator.train_on_batch(real_trajs_bc[:4], real_bc)
#                 d_loss_syn = self.discriminator.train_on_batch(gen_trajs_bc[:4], syn_bc)
#                 d_loss = 0.5 * np.add(d_loss_real, d_loss_syn)

#                 # Train the generator
#                 # Mask and noise are used
#                 noise = np.random.normal(0, 1, (idx.shape[0], self.latent_dim))
#                 real_trajs_bc[5] = noise
#                 g_loss = self.combined.train_on_batch(real_trajs_bc, real_bc)

#                 # Print the losses
#                 print("[Epoch %d/%d] [Batch %d/%d] D Loss: %f | G Loss: %f" % (epoch, epochs, i+1, num_batches, d_loss[0], g_loss))
#             # Save the params
#             if epoch % sample_interval == 0:
#                 self.save_checkpoint(epoch)
#                 print('Model params saved to the disk.')
    
    def save_checkpoint(self, epoch):
        self.combined.save_weights("training_params/C_model_"+str(epoch)+".h5")
        self.generator.save_weights("training_params/G_model_"+str(epoch)+".h5")
        self.discriminator.save_weights("training_params/D_model_"+str(epoch)+".h5")
        print("Training Params Saved")