"""
Main file for running the training and testing of the model for the game of Pong.
"""

import gymnasium as gym
import numpy as np
import tensorflow as tf

from dqn_tf import DQNAgent
from pong import PongEnv

def preprocess_frame(frame):
    frame = frame[35:195, 15:145]
    frame = frame[::2, ::2, 0]
    frame[frame == 144] = 0
    frame[frame == 109] = 0
    frame[frame != 0] = 1
    return np.expand_dims(frame, axis=2)

def main():
    env = PongEnv()
    agent = DQNAgent((80, 65, 1), 3)
    episodes = 1000
    for e in range(episodes):
        state = preprocess_frame(env.reset())
        state = np.expand_dims(state, axis=0)
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_frame(next_state)
            next_state = np.expand_dims(next_state, axis=0)
            agent.train(state, action, reward, next_state, done)
            state = next_state
        print(f"episode: {e}/{episodes}, score: {env.score}, e: {agent.epsilon}")
        if e % 50 == 0:
            agent.save(f"models/model_{e}")
        
if __name__ == "__main__":
    main()