import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

data = pd.read_csv(r"C:\Users\sadur\OneDrive - SSN Trust\Semester 1\Reinforcement Learning\[23120023]InternalAssessMent1RLML\data_UCB.csv")

def thompson_sampling(dataset, N=10000, d=10):
    numbers_of_wins = [0] * d
    numbers_of_losses = [0] * d
    ads_selected = []
    total_reward = 0
    for n in range(N):
        ad = 0
        max_random = 0
        for i in range(d):
            random_beta = np.random.beta(numbers_of_wins[i] + 1, numbers_of_losses[i] + 1)
            if random_beta > max_random:
                max_random = random_beta
                ad = i
        ads_selected.append(ad)
        reward = dataset.values[n, ad]
        if reward == 1:
            numbers_of_wins[ad] += 1
        else:
            numbers_of_losses[ad] += 1
        total_reward += reward
    return ads_selected, total_reward

ads_selected_ts, total_reward_ts = thompson_sampling(data)
total_reward_ts, np.bincount(ads_selected_ts)
print(total_reward_ts)
print(ads_selected_ts)


hist_data, bins, patches = plt.hist(ads_selected_ts, bins=range(0, 11), align='left', rwidth=0.8) 
plt.title('Histogram of ads Selected via Thompson Sampling')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.xticks(range(0, 10))
plt.bar_label(patches)
plt.show()
