import numpy as np

def generate_episode(policy, grid_size):
    episode = []
    state = (0, 0)  # Starting state
    while state not in [(grid_size - 1, grid_size - 1)]:
        action = policy[state]
        next_state, reward = take_action(state, action, grid_size)
        episode.append((state, action, reward))
        state = next_state
    return episode

def take_action(state, action, grid_size):

    row, col = state
    if action == "up" and row > 0:
        next_state = (row - 1, col)
    elif action == "down" and row < grid_size - 1:
        next_state = (row + 1, col)
    elif action == "left" and col > 0:
        next_state = (row, col - 1)
    elif action == "right" and col < grid_size - 1:
        next_state = (row, col + 1)
    else:
        next_state = state

    if next_state == (grid_size - 1, grid_size - 1):
        reward = 1
    else:
        reward = 0

    return next_state, reward

def monte_carlo_prediction(policy, grid_size, num_episodes):

    state_values = np.zeros((grid_size, grid_size))
    state_counts = np.zeros((grid_size, grid_size))

    for _ in range(num_episodes):
        episode = generate_episode(policy, grid_size)
        G = 0
        for t in reversed(range(len(episode))):
            state, _, reward = episode[t]
            G = reward + G  # Update the return
            state_counts[state] += 1
            state_values[state] += (G - state_values[state]) / state_counts[state]

    return state_values

# Example usage:
grid_size = 4
policy = {(i, j): "right" if (i + j) % 2 == 0 else "down" for i in range(grid_size) for j in range(grid_size)}
num_episodes = 1000

estimated_state_values = monte_carlo_prediction(policy, grid_size, num_episodes)
print("Estimated State Values:")
print(estimated_state_values)
