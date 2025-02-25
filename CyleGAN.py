import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, UpSampling2D, Flatten, Dense
from tensorflow.keras.models import Model
import librosa
import numpy as np
import matplotlib.pyplot as plt

class EmotionCycleGAN:
    def __init__(self, input_shape=(128, 128, 1)):
        self.input_shape = input_shape
        self.build_model()

    def build_generator(self):
        """Creates the Generator model."""
        inputs = Input(shape=self.input_shape)
        
        # Encoder
        x = Conv2D(64, (7, 7), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Bottleneck
        x = Conv2D(128, (3, 3), padding='same', strides=2)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Decoder
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        outputs = Conv2D(1, (7, 7), padding='same', activation='tanh')(x)
        return Model(inputs, outputs, name="Generator")

    def build_discriminator(self):
        """Creates the Discriminator model."""
        inputs = Input(shape=self.input_shape)

        x = Conv2D(64, (3, 3), padding='same', strides=2)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(128, (3, 3), padding='same', strides=2)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Flatten()(x)
        outputs = Dense(1, activation='sigmoid')(x)

        return Model(inputs, outputs, name="Discriminator")

    def build_model(self):
        """Builds the full CycleGAN model with loss functions."""
        self.generator_A2B = self.build_generator()
        self.generator_B2A = self.build_generator()
        self.discriminator_A = self.build_discriminator()
        self.discriminator_B = self.build_discriminator()

        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

    @tf.function
    def train_step(self, real_A, real_B):
        """Performs a single training step."""
        with tf.GradientTape(persistent=True) as tape:
            fake_B = self.generator_A2B(real_A, training=True)
            cycle_A = self.generator_B2A(fake_B, training=True)

            fake_A = self.generator_B2A(real_B, training=True)
            cycle_B = self.generator_A2B(fake_A, training=True)

            discriminator_A_fake = self.discriminator_A(fake_A, training=True)
            discriminator_B_fake = self.discriminator_B(fake_B, training=True)

            cycle_loss = tf.reduce_mean(tf.abs(real_A - cycle_A)) + tf.reduce_mean(tf.abs(real_B - cycle_B))
            generator_loss = tf.reduce_mean(tf.losses.mse(tf.ones_like(discriminator_B_fake), discriminator_B_fake)) + \
                             tf.reduce_mean(tf.losses.mse(tf.ones_like(discriminator_A_fake), discriminator_A_fake)) + \
                             cycle_loss

            discriminator_loss_A = tf.reduce_mean(tf.losses.mse(tf.ones_like(self.discriminator_A(real_A, training=True)), self.discriminator_A(real_A, training=True))) + \
                                   tf.reduce_mean(tf.losses.mse(tf.zeros_like(self.discriminator_A(fake_A, training=True)), self.discriminator_A(fake_A, training=True)))

            discriminator_loss_B = tf.reduce_mean(tf.losses.mse(tf.ones_like(self.discriminator_B(real_B, training=True)), self.discriminator_B(real_B, training=True))) + \
                                   tf.reduce_mean(tf.losses.mse(tf.zeros_like(self.discriminator_B(fake_B, training=True)), self.discriminator_B(fake_B, training=True)))

        generator_grads = tape.gradient(generator_loss, self.generator_A2B.trainable_variables + self.generator_B2A.trainable_variables)
        discriminator_grads_A = tape.gradient(discriminator_loss_A, self.discriminator_A.trainable_variables)
        discriminator_grads_B = tape.gradient(discriminator_loss_B, self.discriminator_B.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_grads, self.generator_A2B.trainable_variables + self.generator_B2A.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_grads_A, self.discriminator_A.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_grads_B, self.discriminator_B.trainable_variables))

        return generator_loss, discriminator_loss_A, discriminator_loss_B

    def save_model(self, directory):
        """Saves the model weights."""
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.generator_A2B.save_weights(os.path.join(directory, 'generator_A2B.h5'))
        self.generator_B2A.save_weights(os.path.join(directory, 'generator_B2A.h5'))
        self.discriminator_A.save_weights(os.path.join(directory, 'discriminator_A.h5'))
        self.discriminator_B.save_weights(os.path.join(directory, 'discriminator_B.h5'))

    def load_model(self, directory):
        """Loads the model weights."""
        self.generator_A2B.load_weights(os.path.join(directory, 'generator_A2B.h5'))
        self.generator_B2A.load_weights(os.path.join(directory, 'generator_B2A.h5'))
        self.discriminator_A.load_weights(os.path.join(directory, 'discriminator_A.h5'))
        self.discriminator_B.load_weights(os.path.join(directory, 'discriminator_B.h5'))


if __name__ == '__main__':
    model = EmotionCycleGAN(input_shape=(128, 128, 1))
    print("🚀 Model compiled successfully!")
