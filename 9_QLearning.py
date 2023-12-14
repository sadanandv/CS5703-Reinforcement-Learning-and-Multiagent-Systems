import gym
import numpy as np
import random

env = gym.make("FrozenLake-v1", is_slippery=True)

Q = np.zeros([env.observation_space.n, env.action_space.n])

learning_rate = 0.8
discount_factor = 0.95
num_episodes = 20000

epsilon = 0.96
max_epsilon = 1.00
min_epsilon = 0.21
decay_rate = 0.001

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:#Epsilon Greedy Action Selection
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        new_state, reward, done, info, deets = env.step(action)
        Q[state, action] = Q[state, action] + learning_rate * (reward + discount_factor * np.max(Q[new_state, :]) - Q[state, action])
        state = new_state

    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

print("Q-table:")
print(Q)

#Evaluate Agent
test_episodes = 100
successes = 0

for episode in range(test_episodes):
    state = env.reset()[0]
    done = False

    while not done:
        action = np.argmax(Q[state, :])
        new_state, reward, done, _, _ = env.step(action)
        state = new_state
        if done and reward == 1.0:
            successes += 1

#Calculate the success rate
success_rate = successes / test_episodes
print(f"Success Rate: {success_rate:.2f}")

