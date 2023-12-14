import matplotlib.pyplot as plt
import numpy as np

n_trials = 10000
eps = 0.1
bandit_prob = [0.2,0.5,0.75]

class Bandit:
	def __init__(self,p):
		self.p = p
		self.p_estimate = 0.
		self.N = 0.
	
	def pull(self):
		return np.random.random() < self.p

	def update(self,x):
		self.N += 1.
		self.p_estimate = ((self.N-1)*self.p_estimate+x)/ self.N

def experiment():
		bandits = [Bandit(p) for p in bandit_prob]
		rewards = np.zeros(n_trials)
		n_explored = 0
		n_exploited = 0
		n_optimal = 0
		optimal_j = np.argmax([b.p for b in bandits])
		print(optimal_j)


		for i in range(n_trials):
			if np.random.random() < eps:
				n_explored += 1
				j = np.random.randint(len(bandits))
			else:
				n_exploited +=1
				j= np.argmax([b.p_estimate for b in bandits])
			if j==optimal_j:
				n_optimal +=1

			x = bandits[j].pull()
			rewards[i] = x
			bandits[j].update(x)

		for b in bandits:
			print('mean estimate:', b.p_estimate)

		print('total reward:', rewards.sum())
		print('num of times explored:',n_explored)
		print('num of times exploited:',n_exploited)
		print('total:',n_exploited+n_explored)
		print('num of times optimal bandit selected:', n_optimal)

		cumulative_rewards = np.cumsum(rewards)
		win_rates = cumulative_rewards/(np.arange(n_trials)+1)
		plt.plot(win_rates)
		plt.plot(np.ones(n_trials)*np.max(bandit_prob))

if __name__ == "__main__":
	experiment()
	






