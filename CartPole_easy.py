#!/usr/bin/env python
# encoding=utf-8

import gym
import numpy as np


def action_linear(env, weight):
    observation = env.reset()
    total_reward = 0
    for i in range(1000):
        env.render()
        action = 1 if np.dot(weight, observation) >= 0 else 0
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


def random_weight():
    env = gym.make('CartPole-v0')
    env.reset()
    np.random.seed(10)
    best_reward = -100.0
    best_weight = np.random.rand(4)

    for _ in range(1000):
        weight = np.random.rand(4)
        cur_reward = action_linear(env, weight)

        if cur_reward > best_reward:
            best_reward = cur_reward
            best_weight = weight

        if best_reward == 1000:
            break

    print('best random reward: ', best_reward)
    print('best random weight: ', best_weight)


def hill_climbing_weight():
    env = gym.make('CartPole-v0')
    env.reset()
    np.random.seed(10)
    best_reward = -100.0
    best_weight = np.random.rand(4)

    for _ in range(1000):
        weight = best_weight + np.random.normal(0, 0.01, 4)
        cur_reward = action_linear(env, weight)

        if cur_reward > best_reward:
            best_reward = cur_reward
            best_weight = weight

        if best_reward == 1000:
            break

    print('best hill climbing reward: ', best_reward)
    print('best hill climbing weight: ', best_weight)


random_weight()
hill_climbing_weight()