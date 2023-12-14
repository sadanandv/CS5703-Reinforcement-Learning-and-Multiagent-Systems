import numpy as np

class Gridworld:
    def __init__(self, rows, cols, start, goal, terminal_states, rewards, discount=0.9):
        self.rows = rows
        self.cols = cols
        self.curr_state = start
        self.goal = goal
        self.terminal_states = terminal_states
        self.rewards = rewards
        self.discount = discount
        self.actions = ['U', 'D', 'L', 'R']
        self.action_mapping = {
            'U': (-1, 0),
            'D': (1, 0),
            'L': (0, -1),
            'R': (0, 1)
        }
        self.values = np.zeros((rows, cols))
        
    def set_state(self, state):
        self.curr_state = state
    
    def get_curr_state(self):
        return self.curr_state
    
    def is_terminal(self, state):
        return state in self.terminal_states or state == self.goal
    
    def is_out_of_bounds(self, state):
        return state[0] < 0 or state[0] >= self.rows or state[1] < 0 or state[1] >= self.cols
    
    def get_next_state(self, state, action):
        action_offset = self.action_mapping[action]
        next_state = (state[0] + action_offset[0], state[1] + action_offset[1])
        
        if self.is_out_of_bounds(next_state) or self.is_terminal(state):
            return state
        
        return next_state
    
    def move(self, action):
        self.curr_state = self.get_next_state(self.curr_state, action)
        return self.curr_state
    
    def bellman(self, tolerance=1e-6):
        policy = {}
        while True:
            delta = 0
            for row in range(self.rows):
                for col in range(self.cols):
                    state = (row, col)
                    if self.is_terminal(state):
                        continue
                    
                    v = self.values[state]
                    best_value = float('-inf')
                    for action in self.actions:
                        next_state = self.get_next_state(state, action)
                        reward = self.rewards.get(state, {}).get(action, 0)
                        value = reward + self.discount * self.values[next_state]
                        
                        if value > best_value:
                            best_value = value
                            policy[state] = action
                            
                    self.values[state] = best_value
                    delta = max(delta, abs(v - self.values[state]))
            
            if delta < tolerance:
                break
        
        return policy, self.values


rows, cols = 3, 3
start = (2, 0)
goal = (0, 2)
terminal_states = [(1, 2),(0,2)]
rewards = {
    (0, 1): {'R': 5}, 
    (1, 1): {'R': -5}, 
}

gridworld = Gridworld(rows, cols, start, goal, terminal_states, rewards)
policy, values = gridworld.bellman()
print("Optimal Policy:", policy)
print("Optimal Values:", values)