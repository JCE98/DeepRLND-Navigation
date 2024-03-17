'''
Udacity Deep Reinforcement Learning Nanodegree
Navigation Project
Author: Josiah Everhart
Objective: Train a deep reinforcement learning agent to navigate a world, collecting yellow bananas and avoiding blue bananas, using value based optimization methods
'''

#Import Libraries
from unityagents import UnityEnvironment
import numpy as np
import sys, os, torch, time
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent import Agent

def dqn(env, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    '''
    Deep Q-Learning
    
    Params
    ======
        env (user-defined Unity-ML environment class): Unity-ML environment in which the agent is to be trained
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    '''
    scores = []                                            # list containing scores from each episode
    scores_window = deque(maxlen=100)                      # last 100 scores
    eps = eps_start                                        # initialize epsilon
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]
    action_size = brain.vector_action_space_size
    state_size = len(env_info.vector_observations[0])
    agent = Agent(state_size, action_size,seed=0)
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment in training mode
        state = env_info.vector_observations[0]            # get the current state
        score = 0                                          # initialize score
        for t in range(max_t):
            action = agent.act(state, eps)                 # agent takes action
            action = action.astype(np.int32)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            if done:                                       # exit loop if episode finished
                break
        scores_window.append(score)                        # save most recent score
        scores.append(score)                               # save most recent score
        eps = max(eps_end, eps_decay*eps)                  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.\
              format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:                           #Taking an average score every 100 episodes
            print('\rEpisode {}\tAverage Score: {:.2f}'.\
                  format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0:                  #Checking if environment has been solved
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            if not os.path.exists(projectFilesPath+"/results/"+time.strftime("%Y%m%d-%H%M%S")): os.makedirs(projectFilesPath+"/results/"+time.strftime("%Y%m%d-%H%M%S"))
            torch.save(agent.qnetwork_local.state_dict(), projectFilesPath+"/results/"+time.strftime("%Y%m%d-%H%M%S")+'/checkpoint.pth')
            break
    return scores

def DemoTrainedAgent(env, agentFile, demoEpisodes=3, max_t=200):
    '''
    Demonstrate Trained Agent Performance
    
    Params
    ======
        env (user-defined Unity-ML environment class): Unity-ML environment in which the agent is to be trained
        agentFile (str): path to the trained agent checkpoint .pth file that contains the trained weights
        demoEpisodes (int): maximum number of demonstration episodes
        max_t (int): maximum number of time steps per episode
    '''
    scores = []                                            # list containing scores from each episode
    scores_window = deque(maxlen=100)                      # last 100 scores
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]
    action_size = brain.vector_action_space_size
    state_size = len(env_info.vector_observations[0])
    state_dict = torch.load(agentFile)
    agent = Agent(state_size, action_size,seed=0)
    agent.setFClayerNeurons(fc1_units=state_dict['fc1.weight'].shape[0], fc2_units=state_dict['fc2.weight'].shape[0])
    agent.qnetwork_local.load_state_dict(torch.load(agentFile))
    for i_episode in range(1,demoEpisodes+1):
        env_info = env.reset(train_mode=False)[brain_name] # reset the environment in demonstration mode
        state = env_info.vector_observations[0]            # get the current state
        score = 0                                          # initialize score
        for t in range(max_t):
            action = agent.act(state)                      # trained agent takes action
            action = action.astype(np.int32)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            if done:                                       # exit loop if episode finished
                break
        scores_window.append(score)                        # save most recent score
        scores.append(score)                               # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f}'.\
              format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:                           #Taking an average score every 100 episodes
            print('\rEpisode {}\tAverage Score: {:.2f}'.\
                  format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0:                   #Checking if environment has been solved
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            break
    return scores


if __name__ == "__main__":
    #Initialize environment
    projectFilesPath = os.path.dirname(sys.argv[0])
    env = UnityEnvironment(file_name = projectFilesPath + "/Banana_Windows_x86_64/Banana.exe")

    #Training or Demonstration
    while True:
        response = input("Would you like to train a new agent from scratch, or demonstrate a trained agent? (train or demo): ")
        if response == "train":
            #Train Agent
            scores = dqn(env)
            filename = "training"
            break
        elif response == "demo":
            #Demonstrate Trained Agent
            while True:
                agentFile = input("Enter the path to the trained agent checkpoint .pth file: ").replace("\\","/")
                if not os.path.exists(agentFile):
                    print("The file path you entered could not be found. Please check the path and try again.")
                else:
                    print("Loading agent checkpoint at %s..." % agentFile)
                    break
            filename = "demo"
            scores = DemoTrainedAgent(env, agentFile)
            break
        else:
            response = input("Please enter a valid response (demo or train): ")
    env.close()
    #Plotting Training/Demo Results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    if not os.path.exists(projectFilesPath+"/results/"+time.strftime("%Y%m%d-%H%M%S")): os.makedirs(projectFilesPath+"/results/"+time.strftime("%Y%m%d-%H%M%S"))
    plt.savefig(projectFilesPath+"/results/"+time.strftime("%Y%m%d-%H%M%S")+"/"+filename+".png")