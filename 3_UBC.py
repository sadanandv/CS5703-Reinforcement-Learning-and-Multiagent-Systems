import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

data = pd.read_csv(r"C:\Users\snucse\Desktop\OneDrive_1_12-14-2023\Datasets\data_UCB.csv")
data.head()

data.describe()

N = 1000
d = 10
ads_selected = []
num_selection = [0]*d
sum_reward = [0]*d
total_reward = 0

# Experiment
for n in range(0,N):
    ad = 0
    max_bound = 0
    for i in range(0,d):
        if(num_selection[i] >0):
            avg_reward = (sum_reward[i]/num_selection[i])
            UB = avg_reward + (0.99 * math.sqrt((3/2)*(math.log(n+1)/num_selection[i])))
            LB = avg_reward - (0.99 * math.sqrt((3/2)*(math.log(n+1)/num_selection[i])))
        else:
            UB = 1e400
            LB = 1e400
        if UB > max_bound:
            max_bound = UB
            ad = i
    ads_selected.append(ad) 
    num_selection[ad] = num_selection[ad]+1
    reward = data.values[n,ad]
    sum_reward[ad] = sum_reward[ad] + reward
    total_reward += total_reward + reward

plt.hist(ads_selected)