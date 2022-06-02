import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.io import imread
import sys, os

from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

# Loading and scaling the MNIST Data

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0 * 2 - 1, x_test / 255.0 * 2 - 1

N, H, W = x_train.shape
D = H * W
x_train = x_train.reshape(-1, D)
x_test = x_test.reshape(-1, D)
latent_dim = 100

# The generator model

def generator_model(latent_dim):
  i = Input(shape=(latent_dim,))
  x = Dense(256, activation=LeakyReLU(alpha=0.2))(i)
  x = BatchNormalization(momentum=0.7)(x)
  x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
  x = BatchNormalization(momentum=0.7)(x)
  x = Dense(1024, activation=LeakyReLU(alpha=0.2))(x)
  x = BatchNormalization(momentum=0.7)(x)
  x = Dense(D, activation='tanh')(x)

  model = Model(i, x)
  return model

# The discriminator model

def discriminator_model(img_size):
  i = Input(shape=(img_size,))
  x = Dense(512, activation=LeakyReLU(alpha=0.2))(i)
  x = Dense(256, activation=LeakyReLU(alpha=0.2))(x)
  x = Dense(1, activation='sigmoid')(x)

  
  model = Model(i, x)
  return model

# Instanciating the discriminator
discriminator = discriminator_model(D)
discriminator.compile( loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# Instanciating the combined model
generator = generator_model(latent_dim)

generator.summary()

discriminator.summary()

# Create an input to represent noise sample from latent space
# Pass noise through generator to get an image

z = Input(shape=(latent_dim,))
img = generator(z)

# Discriminator shouldn't be trained so we set its trainable property to false
discriminator.trainable = False

# The true output is fake, but we label them real!
# Passing the output of Generator to the Discriminator
fake_pred = discriminator(img)

# Create the combined model object
combined_model_gen = Model(z, fake_pred)

# Compile the combined model
combined_model_gen.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Config

batch_size = 32
epochs = 40000
sample_period = 200 # every `sample_period` steps generate and save some data"

ones = np.ones(batch_size)
zeros = np.zeros(batch_size)

discrim_losses = []
gen_losses = []

# A function to generate a grid of random samples from the generator and save them to a file

def sample_images(epoch):
  rows, cols = 5, 5
  noise = np.random.randn(rows * cols, latent_dim)
  imgs = generator.predict(noise)

  imgs = 0.5 * imgs + 0.5

  fig, axs = plt.subplots(rows, cols)
  idx = 0
  for i in range(rows):
    for j in range(cols):
      axs[i,j].imshow(imgs[idx].reshape(H, W), cmap='gray')
      axs[i,j].axis('off')
      idx += 1
  fig.savefig("gan_images/%d.png" % epoch)
  plt.close()

# Main training loop
for epoch in range(epochs):
  # Train the discriminator 

  # Select a random batch of images
  idx = np.random.randint(0, x_train.shape[0], batch_size)
  real_imgs = x_train[idx]
  
  # Generate fake images
  noise = np.random.randn(batch_size, latent_dim)
  fake_imgs = generator.predict(noise)
  
  discrim_loss_real, discrim_acc_real = discriminator.train_on_batch(real_imgs, ones)
  discrim_loss_fake, discrim_acc_fake = discriminator.train_on_batch(fake_imgs, zeros)
  discrim_loss = 0.5 * (discrim_loss_real + discrim_loss_fake)
  discrim_acc  = 0.5 * (discrim_acc_real + discrim_acc_fake)
  
    ### Train generator ###

  
  noise = np.random.randn(batch_size, latent_dim)
  gen_loss = combined_model_gen.train_on_batch(noise, ones)
  
  # do it again!
  noise = np.random.randn(batch_size, latent_dim)
  gen_loss = combined_model_gen.train_on_batch(noise, ones)
  
  # Save the losses
  discrim_losses.append(discrim_loss)
  gen_losses.append(gen_loss)
  
  if epoch % 100 == 0:
    print(f"epoch: {epoch+1}/{epochs}, discrim_loss: {discrim_loss:.2f}, \
      discrim_acc: {discrim_acc:.2f}, gen_loss: {gen_loss:.2f}")
  
  if epoch % sample_period == 0:
    sample_images(epoch)

plt.plot(gen_losses, label='gen_losses')
plt.plot(discrim_losses, label='discrim_losses')
plt.legend()

a = imread('gan_images/0.png')
plt.imshow(a)

a = imread('gan_images/1000.png')
plt.imshow(a)

a = imread('gan_images/30000.png')
plt.imshow(a)

a = imread('gan_images/39800.png')
plt.imshow(a)