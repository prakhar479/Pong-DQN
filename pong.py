"""
Custom pong game environement for OpenAI gym using pygame
"""

import gymnasium as gym
import numpy as np
import pygame as pg
import random as rnd
from gymnasium import spaces

class Ball:
    def __init__(self, width, height, speed):
        self.width = width
        self.height = height
        self.speed = speed
        self.x = width // 2
        self.y = height // 2
        self.direction = rnd.choice([1, -1]), rnd.choice([1, -1])
    
    def move(self):
        self.x += self.speed * self.direction[0]
        self.y += self.speed * self.direction[1]
    
    def bounce(self):
        self.direction = -self.direction[0], self.direction[1]
    
    def reset(self):
        self.x = self.width // 2
        self.y = self.height // 2
        self.direction = rnd.choice([1, -1]), rnd.choice([1, -1])

class Paddle:
    def __init__(self, x, height, speed, width):
        self.x = x
        self.y = height // 2
        self.speed = speed
        self.width = width
        self.height = height
    
    def move(self, action):
        if action == 0:
            self.y -= self.speed
        elif action == 2:
            self.y += self.speed
        self.y = max(0, min(self.y, self.height - self.height))
    
    def reset(self):
        self.y = self.height // 2

class PongEnv(gym.Env):
    def __init__(self, width=600, height=400, ball_speed=5, paddle_speed=5):
        self.width = width
        self.height = height
        self.ball_speed = ball_speed
        self.paddle_speed = paddle_speed
        self.ball = Ball(width, height, ball_speed)
        self.paddle = Paddle(width - 10, height, paddle_speed, width)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=255, shape=(width, height, 3), dtype=np.uint8)
        self.screen = pg.display.set_mode((width, height))
        self.clock = pg.time.Clock()
        self.done = False
        self.reward = 0
        self.reset()
    
    def reset(self):
        self.ball.reset()
        self.paddle.reset()
        self.done = False
        self.reward = 0
        return self.get_observation()

    def get_observation(self):
        self.screen.fill((0, 0, 0))
        pg.draw.rect(self.screen, (255, 255, 255), (self.paddle.x, self.paddle.y, 10, self.paddle.height))
        pg.draw.ellipse(self.screen, (255, 255, 255), (self.ball.x, self.ball.y, 10, 10))
        return pg.surfarray.array3d(pg.display.get_surface())
    
    def step(self, action):
        self.reward = 0
        self.paddle.move(action)
        self.ball.move()
        if self.ball.y <= 0 or self.ball.y >= self.height:
            self.ball.bounce()
        if self.ball.x >= self.width - 10:
            if self.paddle.y <= self.ball.y <= self.paddle.y + self.paddle.height:
                self.ball.bounce()
                self.reward = 1
            else:
                self.done = True
                self.reward = -1
        return self.get_observation(), self.reward, self.done, {}
    
    def render(self):
        pg.display.flip()
        self.clock.tick(60)
    
    def close(self):
        pg.quit()