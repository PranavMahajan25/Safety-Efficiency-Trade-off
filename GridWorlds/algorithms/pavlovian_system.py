import numpy as np
import random
import sys
from collections import deque
import matplotlib.pyplot as plt
from utils.helper_functions import seq_to_col_row,row_col_to_seq


def tdlearning(model, mode, pav_mode, random_restarts=True, alpha=0.1, gamma=1.0, mu = 50, tau_0=1.0, tau_k=0.0, maxiter=None, maxeps=2000):
    """
    Solves the supplied environment using temporal difference methods.

    Parameters
    ----------
    model : python object
        Holds information about the environment to solve
        such as the reward structure and the transition dynamics.
    
    mode: string
        Which TD method to use - Qlearning (sarsamax), or sarsa or expected sarsa

    pav_mode: string
        Which pavlovian policy to learn - pavlovian_pain or pavlovian_reward

    random_restarts : bool
        Whether starting position should be random.

    alpha : float
        Algorithm learning rate. Defaults to 0.01

    gamma : float
        Discount rate

    tau_0 : float
        Initial temperature parameter
    
    tau_k : float
        Controls the rate of annealing

    maxiter : int
        The maximum number of iterations to perform per episode.

    maxeps : int
        The number of episodes to run the algorithm for.

    Returns
    -------
    q : numpy array of shape (N, 1)
        The state-action value for the environment where N is the
        total number of states

    pi : numpy array of shape (N, 1)
        Optimal policy for the environment where N is the total
        number of states.

    state_counts : numpy array of shape (N, 1)
        Counts of the number of times each state is visited
    """
    scores = []                        # list containing scores from each episode
    score_means = []
    score_stds = []
    scores_window = deque(maxlen=30)  
    ep_pains = []
    ep_pain_means = []
    ep_pain_stds = []
    ep_pain_window = deque(maxlen=30)  
    ep_steps = []
    ep_step_means = []
    ep_step_stds = []
    ep_step_window = deque(maxlen=30) 

    init_sigma = 0
    # initialize the state-action value function and the state counts
    Q = np.random.randn(model.num_states, model.num_actions) * init_sigma
    V = np.random.randn(model.num_states, 1) * init_sigma
    state_counts = np.zeros((model.num_states, 1))

    for i_episode in range(maxeps):
        tau = tau_0 / (1 + tau_k * i_episode)

        if i_episode % 1 == 0:
            print("\rEpisode {}/{}, 1/tau: {}, mu: {}".format(i_episode, maxeps, 1.0/tau, mu), end="")
            sys.stdout.flush() 

        # for each new episode, start at a random start state or the given start state 
        if random_restarts:
            is_start_state_obstructed = True
            while is_start_state_obstructed:
                model.start_state = np.array([[np.random.randint(1,model.num_rows),np.random.randint(1,model.num_cols)]])
                model.start_state_seq = row_col_to_seq(model.start_state, model.num_cols)
                found_match = False
                for obs_st in model.goal_states:
                    if (model.start_state[0] == obs_st).all():
                        found_match = True
                for obs_st in model.obs_states:
                    if (model.start_state[0] == obs_st).all():
                        found_match = True
                if found_match:
                    is_start_state_obstructed = True
                else:
                    is_start_state_obstructed = False

            is_goal_state_obstructed = True
            while is_goal_state_obstructed:
                model.goal_state = np.array([[np.random.randint(1,model.num_rows),np.random.randint(1,model.num_cols)]])
                model.goal_states_seq = row_col_to_seq(model.goal_state, model.num_cols)
                found_match = False
                for obs_st in model.start_state:
                    if (model.goal_state[0] == obs_st).all():
                        found_match = True
                for obs_st in model.obs_states:
                    if (model.goal_state[0] == obs_st).all():
                        found_match = True
                if found_match:
                    is_goal_state_obstructed = True
                else:
                    is_goal_state_obstructed = False   
        state = int(model.start_state_seq)
        # print(state, seq_to_col_row(state, model.num_cols))
        # print(model.goal_state, model.goal_states_seq)

        ep_score = 0
        ep_pain = 0

        j_step = 0
        while True:
            if maxiter != None and j_step == maxiter:
                # print("Reached max steps {}, terminating episode.".format(maxiter))
                break
            # sample action probabilistically using softmax 
            # temperature parameter varies the noise
            action,_ = sample_action(Q, V, state, model.num_actions, tau, mu)
            # print("Action:", action)

            # initialize p and r
            p, r = 0, np.random.random()

            # sample the next state according to the action and the
            # probability of the transition

            for next_state in range(model.num_states):
                p += model.P[state, next_state, action]
                if r <= p:
                    break
            # print(next_state, seq_to_col_row(next_state, model.num_cols))
        
            if model.R[state, action] < 0:
                ep_pain+= abs(model.R[state, action])
            ep_score+= model.R[state, action] 

            reward = model.R[state, action]

            if pav_mode == 'pavlovian_pain':
                # Calculate the temporal difference and update V function
                V[state] += alpha * (min(reward,0) + gamma * V[next_state] - V[state])
                # Calculate the temporal difference and update Q function
                if mode == 'sarsamax':
                    Q[state, action] += alpha * (min(reward,0) + gamma * np.max(Q[next_state, :]) - Q[state, action])
                elif mode == 'sarsa':
                    next_action,_ = sample_action(Q, V, next_state, model.num_actions, tau, mu)
                    Q[state, action] += alpha * (min(reward,0) + gamma * Q[next_state, next_action] - Q[state, action])
                else:
                    print("Please choose mode out of - ['sarsa', 'sarsamax']")

            elif pav_mode == 'pavlovian_reward':
                # Calculate the temporal difference and update V function
                V[state] += alpha * (max(reward,0) + gamma * V[next_state] - V[state])
                # Calculate the temporal difference and update Q function
                if mode == 'sarsamax':
                    Q[state, action] += alpha * (max(reward,0) + gamma * np.max(Q[next_state, :]) - Q[state, action])
                elif mode == 'sarsa':
                    next_action,_ = sample_action(Q, V, next_state, model.num_actions, tau, mu)
                    Q[state, action] += alpha * (max(reward,0) + gamma * Q[next_state, next_action] - Q[state, action])
                else:
                    print("Please choose mode out of - ['sarsa', 'sarsamax']")

           
            # count the state visits
            state_counts[state] += 1
            j_step +=1

            #Store the previous state
            state = next_state
            # End episode if state is a terminal state
            if np.any(state == model.goal_states_seq):
                # print(state)
                ep_score+= reward
                if pav_mode == 'pavlovian_pain':
                    V[state] += alpha * (min(reward,0) - V[state])
                    Q[state, action] += alpha * (min(reward,0) - Q[state, action])
                elif pav_mode == 'pavlovian_reward':
                    V[state] += alpha * (max(reward,0) - V[state])
                    Q[state, action] += alpha * (max(reward,0) - Q[state, action])
                # print("we won mr. stark! in steps:", j_step)
                break

        scores_window.append(ep_score)       # save most recent score
        scores.append(ep_score)              # save most recent score
        score_means.append(np.mean(scores_window))
        score_stds.append(np.std(scores_window))

        ep_pain_window.append(ep_pain)
        ep_pains.append(ep_pain)
        ep_pain_means.append(np.mean(ep_pain_window))
        ep_pain_stds.append(np.std(ep_pain_window))

        ep_step_window.append(j_step)
        ep_steps.append(j_step)
        ep_step_means.append(np.mean(ep_step_window))
        ep_step_stds.append(np.std(ep_step_window))

    # determine the q function and policy
    q = np.max(Q, axis=1).reshape(-1,1)
    pi = np.zeros_like(q)
    for state in range(model.num_states):
        best_actions = np.argwhere(Q[state, :] == np.amax(Q[state, :]))
        best_actions = best_actions.flatten().tolist()
        action = random.choice(best_actions)
        pi[state] = action
    pi = pi.astype(int)

    spi = np.zeros((model.num_states, model.num_actions))
    for state in range(model.num_states):
        _, action_probs = sample_action(Q, V, state, model.num_actions, tau, mu)
        spi[state, :] = action_probs

    scores = np.array(scores)
    score_means = np.array(score_means)
    score_stds = np.array(score_stds)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plt.plot(np.arange(len(scores)), scores)
    plt.plot(np.arange(len(score_means)), score_means)
    plt.fill_between(np.arange(len(score_means)),score_means-score_stds,score_means+score_stds,alpha=.4)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    ep_pains = np.array(ep_pains, dtype=float)
    ep_pain_means = np.array(ep_pain_means, dtype=float)
    ep_pain_stds = np.array(ep_pain_stds, dtype=float)

    # plot the pains
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plt.plot(np.arange(len(ep_pains)), ep_pains)
    plt.plot(np.arange(len(ep_pain_means)), ep_pain_means)
    plt.fill_between(np.arange(len(ep_pain_means)),ep_pain_means-ep_pain_stds,ep_pain_means+ep_pain_stds,alpha=.4)
    plt.ylabel('Pain accrued in an episode')
    plt.xlabel('Episode #')
    plt.show()

    ep_steps = np.array(ep_steps)
    ep_step_means = np.array(ep_step_means)
    ep_step_stds = np.array(ep_step_stds)

    # plot the steps
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plt.plot(np.arange(len(ep_steps)), ep_steps)
    plt.plot(np.arange(len(ep_step_means)), ep_step_means)
    plt.fill_between(np.arange(len(ep_step_means)),ep_step_means-ep_step_stds,ep_step_means+ep_step_stds,alpha=.4)
    plt.ylabel('Steps to goal')
    plt.xlabel('Episode #')
    plt.show()

    with open('pavlovian_tdlearning_scores.npy', 'wb') as f:
        np.save(f, np.array(scores))
    with open('pavlovian_tdlearning_pains.npy', 'wb') as f:
        np.save(f, np.array(ep_pains))

    return V, Q, pi, state_counts, spi


def sample_action(Q, V, state, num_actions, tau, mu):
    """
    Probabilistic action selection using softmax.

    Parameters
    ----------
    Q : numpy array of shape (N, num_actions)
        Q function for the environment where N is the total number of states.
    
    V : numpy array of shape (N, 1)
        V function for the environment where N is the total number of states.
        
    state : int
        The current state.

    num_actions : int
        The number of actions.

    tau : float
        Temperature parameter

    Returns
    -------
    action : int
        Number representing the selected action between 0 and num_actions.
    """
    advantages = Q[state, :] - V[state]
    transformed_advantages_clipped = np.clip(mu * advantages/ tau, 0, 709)
    Q_num = np.exp(transformed_advantages_clipped)
    Q_denom = np.sum(Q_num)
    Q_dist = Q_num / Q_denom

    action_list = np.arange(num_actions)
    action = random.choices(action_list, weights = Q_dist, k=1)

    return action, Q_dist