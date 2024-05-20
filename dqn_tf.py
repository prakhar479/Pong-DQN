"""
A DQN model implemented in TensorFlow with convolutional layers to learn to play the game of Pong.
"""

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DQNAgent:
    def __init__(self, state_shape, n_actions):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.model = self.create_model()
    
    def create_model(self):
        model = keras.Sequential()
        model.add(layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_shape))
        model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(self.n_actions))
        model.compile(optimizer=keras.optimizers.Adam(), loss='mse')
        return model
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.n_actions)
        return np.argmax(self.model.predict(state)[0])
    
    def train(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + 0.95 * np.amax(self.model.predict(next_state)[0])
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filename):
        self.model.save_weights(filename)
    
    def load(self, filename):
        self.model.load_weights(filename)
    