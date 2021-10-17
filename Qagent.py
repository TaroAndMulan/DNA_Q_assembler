import numpy as np
from collections import defaultdict
import random
import gym
import pickle
import copy

class Agent:

    def __init__(self, nA=10,gamma=0.99,alpha=0.85,epsilon=1.0):

        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
    def set_epsilon(self,num,i):
        self.epsilon = 1-(1/num)*i

    def select_action(self, state):

        if random.random()<self.epsilon:
            return random.choice(range(self.nA))

        return np.argmax(self.Q[state]) 

    def select_action_pruned(self,state,not_allowed):
        prune_Q = self.Q[state]
        prune_Q[not_allowed] = -1000
        if random.random()<self.epsilon:
            return random.choice(np.delete(np.array(range(self.nA)),not_allowed))
        return np.argmax(prune_Q)

    def step(self, state, action, reward, next_state, done):
        target = reward+self.gamma*np.max(self.Q[next_state])
        old = self.Q[state][action]
        self.Q[state][action] += self.alpha*(target-old)

    def save_model(self):
        with open('Q.pickle', 'wb') as handle:
            pickle.dump(self.Q, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self):
        with open('Q.pickle', 'rb') as handle:
            self.Q = pickle.load(handle)