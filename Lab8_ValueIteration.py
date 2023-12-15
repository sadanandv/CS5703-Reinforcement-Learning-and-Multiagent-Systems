import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

env  = gym.make("FrozenLake-v1",
                render_mode="human")
env.reset()
env.render()
env.close()
#%%
print(env.observation_space)
print(env.action_space)
print(env.P[14][2])
#%%
df = 0.9
valueFunctionVector = np.zeros(
    env.observation_space.n)
maxIterations=30
tolerance = 10**(-5)
convergence = []
#%%

for i in range(maxIterations):
    convergence.append(np.linalg.norm(
        valueFunctionVector,2))
    vFNext = np.zeros(
        env.observation_space.n)
    for state in env.P:
        outerSum = 0
        for action in env.P[state]:
            innerSum=0
            for p,nS,r,T in env.P[state][action]:
                innerSum+= p*(r+df*valueFunctionVector[nS])
            outerSum+= 0.25*innerSum
        vFNext[state] = outerSum
    if(np.max(np.abs(vFNext - valueFunctionVector))< tolerance):
        valueFunctionVector = vFNext
        print('converged')
        break
    valueFunctionVector = vFNext

def grid_print(valueFunction):
    ax = sns.heatmap(valueFunction.reshape(4,4),
                     annot=True,
                     cbar=False, cmap='Blues')
    plt.show()
    
grid_print(valueFunctionVector)

plt.plot(convergence)
plt.show()
