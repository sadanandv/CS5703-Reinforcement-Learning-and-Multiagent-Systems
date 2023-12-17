import gym
import numpy as np

# Initialize environment
env = gym.make('MountainCar-v0')  # Example environment

def td_learning(env, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    # Initialize weights for linear approximation
    w = np.zeros(env.observation_space.shape[0])

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            # Implement epsilon-greedy policy
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax([np.dot(w, state) for a in range(env.action_space.n)])  # Exploit

            new_state, reward, done, _, _ = env.step(action)

            # Calculate TD error
            td_target = reward + gamma * np.dot(w, new_state) if not done else reward
            td_error = td_target - np.dot(w, state)

            # Update weights
            w += alpha * td_error * state

            state = new_state

    return w

# Run TD Learning
weights = td_learning(env)
