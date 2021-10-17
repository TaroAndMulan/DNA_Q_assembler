import numpy as np
from collections import defaultdict
import random
import gym
import pickle
import copy
from Qagent import Agent


def transform(state):
    true_state = 0
    for i,d in enumerate(state):
        true_state += (d+1)*(10**i) 
    return true_state

def main():
    # CHANGE ENVIRONMENT HERE
    env = gym.make('gymnome_assembly:GymnomeAssembly_25_10_8-v1')
    # env = gym.make('gymnome_assembly:GymnomeAssembly_25_10_10-v1')
    print ("---------------------------")
    print ("genome: ",env.microgenome)
    print ("reads: ", env.reads)
    print ("solution: ",env.getOptimalPermutation())
    print ("---------------------------")

    # create Q learning agent
    agent = Agent(nA=env.action_space.n,gamma=0.99,alpha=0.8,epsilon=1.0)
    state = env.reset()
    state = transform(state)
    done = False
    num_episodes = 3000000

    for i in range(num_episodes):

        state = env.reset()
        state = transform(state)
        episode_reward = 0
        done = False
        agent.set_epsilon(num_episodes,i)
        not_allowed_action = []

        while not done:

            #action = agent.select_action(state)
            action = agent.select_action_pruned(state,not_allowed_action)
            not_allowed_action.append(action)
            next_state,reward,done,_ = env.step(action)
            temp = next_state
            next_state = transform(next_state)
            agent.step(state,action,reward,next_state,done)
            state = next_state
            episode_reward += reward

        if i%10000 ==0:
            print ("episode",i," solution: ",temp, "score = ", round(episode_reward,2))


if __name__ == "__main__":
    main()
   