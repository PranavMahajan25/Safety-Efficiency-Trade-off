## Since the latest commit -  this code implements ...colearn_correctedQp.py
## Corrected the Qp as per the Overleaf document. (minor correction)
## instead of the Q+ and Q- we now have Qr and Qp
## a' is still sampled seperately here, like in split Q,
## ..._correctedpolicy.py will combine it like maxpain.

from math import e
from posixpath import join
import numpy as np
import random
import sys
import os
from collections import deque
import matplotlib.pyplot as plt
from utils.helper_functions import seq_to_col_row,row_col_to_seq
# from utils.visualise import visualise_trajectory
from run_codes.dynamic_environment import move_catastrophic_death, move_goal, disappear_catastrophic_death


def tdlearning(model, mode, pav_pain = True, w=0, modulate_w=False, random_restarts=True, alpha=0.1, gamma=1.0, mu = 50, tau_0=1.0, tau_k=0.0, maxiter=None, maxeps=2000, dynobs=False, dynobs_vanish=False, goalchange=None, saverun_folder='./run_outputs/', run_num = 1, kappa = 2, beta = 0.5, two_goals=False):
    """
    Solves the supplied environment using temporal difference methods.

    Parameters
    ----------
    model : python object
        Holds information about the environment to solve
        such as the reward structure and the transition dynamics.

    mode : string
        Which TD method to use - Qlearning (sarsamax), or sarsa or expected sarsa

    pav_pain: bool
        Whether to learn the pavlovian pain policy

    w : float
        Linear weight [0,1] for Pavlovian-instrumental transfer

    modulate_w : bool
        Whether to keep w fixed or modulate it based on uncertainty

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

    dynobs : bool
        Only used to move the obstacle in environment movingbstacle, using dynamic_environment.py
    
    dynobs_vanish: bool
        If true, the obstacle will vanish half way through (500 episode). Default false.

    goalchange : numpy array
        Array of new goal states (default None)
        Only used to shift goal, using dynamic_environment.py

    saverun_folder: string
        String with path to folder where to save the info about this run -
        scores, ep_pains, ep_steps, cum_pains, cum_steps

    run_num: int
        run number to be included in save path

    kappa: float
        scaling factor for running average of TDE

    beta: float
        scaling factor for learning rate of running average of TDE

    two_goals: bool
        To decide if to use two different goal sequences

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
    cum_pains = []
    cum_pain_means = []
    cum_pain_stds = []
    cum_pain_window = deque(maxlen=30)
    ep_steps = []
    ep_step_means = []
    ep_step_stds = []
    ep_step_window = deque(maxlen=30)
    cum_steps = []
    cum_step_means = []
    cum_step_stds = []
    cum_step_window = deque(maxlen=30)
    w_list = []
    TDE_list = []
    modulate_w_step = False
    visualise = False
    plot_plots = False
    update_freq = 1
    print(plot_plots)
    init_sigma = 0
    # initialize the state-action value function and the state counts
    Q = np.random.randn(model.num_states, model.num_actions) * init_sigma
    V = np.random.randn(model.num_states, 1) * init_sigma
    Vr = np.random.randn(model.num_states, 1) * init_sigma
    Vp = np.random.randn(model.num_states, 1) * init_sigma
    absTDE = np.random.randn(model.num_states, 1) * init_sigma
    running_avg_absTDE = 0
    beta=beta # scale learning rate for absTDE

    #only used in two_goals setting
    cum_goal_one_reach = 0
    cum_goal_two_reach = 0
    cum_goal_one_reaches = []
    cum_goal_two_reaches = []

    if pav_pain:
        Qp = np.random.randn(model.num_states, model.num_actions) * init_sigma
    else:
        Qp = None

    state_counts = np.zeros((model.num_states, 1))

    kappa = kappa
    tau_min = 0.01
    cum_pain = 0
    cum_step = 0
    for i_episode in range(maxeps):
        tau = max(tau_0 / (1 + tau_k * i_episode), tau_min)

        if modulate_w_step and not modulate_w:
            if i_episode<300:
                w=0.5
            else:
                w=0.1
        # print(model.goal_states_seq)
        if goalchange is not None and i_episode==500:
            move_goal(model, goalchange)
            # print(model.goal_states_seq)

        if i_episode % update_freq == 0:
            print("\rEpisode {}/{}, 1/tau: {}, mu: {}, w: {}".format(i_episode+1, maxeps, 1.0/tau, mu, w), end="")
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

        state = int(model.start_state_seq)
        # print(state, seq_to_col_row(state, model.num_cols))

        ep_score = 0
        ep_pain = 0

        j_step = 0
        while True:
            if maxiter != None and j_step == maxiter:
                # print("Reached max steps {}, terminating episode.".format(maxiter))
                break
            if dynobs:
                if dynobs_vanish:
                    if i_episode<500:
                        move_catastrophic_death(model, j_step)
                    else:
                        disappear_catastrophic_death(model)
                else:
                    move_catastrophic_death(model, j_step)

            # sample action probabilistically using softmax
            # temperature parameter varies the noise
            if modulate_w:
                w = compute_w(state, running_avg_absTDE, kappa)
                # print(w)
            action,_ = sample_action(Q, (V,Vr,Vp), state, model.num_actions, tau, mu, Qp, w)
            # print("Action:", action)

            # initialize p and r
            p, r = 0, np.random.random()

            # sample the next state according to the action and the
            # probability of the transition

            # REVISIT this
            for next_state in range(model.num_states):
                p += model.P[state, next_state, action]
                if r <= p:
                    break
            # print(next_state, seq_to_col_row(next_state, model.num_cols))

            if model.R[state, action] < 0:
                ep_pain+= abs(model.R[state, action])
            ep_score+= model.R[state, action]

            # Calculate the temporal difference and update V function
            V[state] += alpha * (model.R[state, action] + gamma * V[next_state] - V[state])
            # print(V[state])
            Vr[state] += alpha * (max(model.R[state, action],0) + gamma * Vr[next_state] - Vr[state])
            Vp[state] += alpha * (-min(model.R[state, action],0) + gamma * Vp[next_state] - Vp[state])
            absTDE[state] = (1- beta*alpha) * absTDE[state] + beta*alpha * (abs(model.R[state, action] + gamma * np.max(Q[next_state, :]) - Q[state, action]))
            running_avg_absTDE = (1- beta*alpha) * running_avg_absTDE + beta*alpha * (abs(model.R[state, action] + gamma * np.max(Q[next_state, :]) - Q[state, action]))
            TDE_list.append(model.R[state, action] + gamma * np.max(Q[next_state, :]) - Q[state, action])

            # Calculate the temporal difference and update Q function
            if mode == 'sarsamax':
                Q[state, action] += alpha * (model.R[state, action] + gamma * np.max(Q[next_state, :]) - Q[state, action])
            elif mode == 'sarsa':
                if modulate_w:
                    w = compute_w(next_state, running_avg_absTDE, kappa)
                next_action, _ = sample_action(Q, (V,Vr,Vp), next_state, model.num_actions, tau, mu, Qp, w)
                Q[state, action] += alpha * (model.R[state, action] + gamma * Q[next_state, next_action] - Q[state, action])
            else:
                print("Please choose mode out of - ['sarsa', 'sarsamax', 'beta-pessimistic']")

            if pav_pain:
                if mode == 'sarsamax':
                    Qp[state, action] += alpha * (-min(model.R[state, action],0) + gamma * np.min(Qp[next_state, :]) - Qp[state, action])
                elif mode == 'sarsa':
                    if modulate_w:
                        w = compute_w(next_state, running_avg_absTDE, kappa)
                    next_action, _ = sample_action(Q, (V,Vr,Vp), next_state, model.num_actions, tau, mu, Qp, w)
                    Qp[state, action] += alpha * (-min(model.R[state, action],0) + gamma * Qp[next_state, next_action] - Qp[state, action])
                else:
                    print("Please choose mode out of - ['sarsa', 'sarsamax', 'beta-pessimistic']")

            # count the state visits
            state_counts[state] += 1
            j_step +=1

            #Store the previous state
            state = next_state
            # End episode if state is a terminal state
            if two_goals:
                if np.any(state == model.small_goal_states_seq) or np.any(state == model.big_goal_states_seq):
                    ep_score+= model.R[state, action]
                    V[state] += alpha * (model.R[state, action] - V[state])
                    Vr[state] += alpha * (max(model.R[state, action],0)  - Vr[state])
                    absTDE[state] = (1- beta*alpha) * absTDE[state] + beta*alpha * (abs(model.R[state, action] - Q[state, action]))
                    running_avg_absTDE = (1- beta*alpha) * running_avg_absTDE + beta*alpha * (abs(model.R[state, action] + gamma * np.max(Q[next_state, :]) - Q[state, action]))
                    TDE_list.append(model.R[state, action] - Q[state, action])
                    Q[state, action] += alpha * (model.R[state, action] - Q[state, action])
                    if pav_pain:
                        Qp[state, action] += alpha * (-min(model.R[state, action],0) - Qp[state, action])
                    # print("we won mr. stark! in steps:", j_step)
                    if np.any(state == model.small_goal_states_seq):
                        cum_goal_one_reach+=1
                    else:
                        cum_goal_two_reach+=1
                    break
            else:
                if np.any(state == model.goal_states_seq):
                    ep_score+= model.R[state, action]
                    V[state] += alpha * (model.R[state, action] - V[state])
                    Vr[state] += alpha * (max(model.R[state, action],0)  - Vr[state])
                    absTDE[state] = (1- beta*alpha) * absTDE[state] + beta*alpha * (abs(model.R[state, action] - Q[state, action]))
                    running_avg_absTDE = (1- beta*alpha) * running_avg_absTDE + beta*alpha * (abs(model.R[state, action] + gamma * np.max(Q[next_state, :]) - Q[state, action]))
                    TDE_list.append(model.R[state, action] - Q[state, action])
                    Q[state, action] += alpha * (model.R[state, action] - Q[state, action])
                    if pav_pain:
                        Qp[state, action] += alpha * (-min(model.R[state, action],0) - Qp[state, action])
                    # print("we won mr. stark! in steps:", j_step)
                    break
            if model.restart_states_seq is not None and np.any(state == model.restart_states_seq):
                ep_score+= model.R[state, action]
                V[state] += alpha * (model.R[state, action] - V[state])
                Vp[state] += alpha * (-min(model.R[state, action],0)  - Vp[state])
                absTDE[state] = (1-beta*alpha) * absTDE[state] + beta*alpha * (abs(model.R[state, action] - Q[state, action]))
                running_avg_absTDE = (1- beta*alpha) * running_avg_absTDE + beta*alpha * (abs(model.R[state, action] + gamma * np.max(Q[next_state, :]) - Q[state, action]))
                TDE_list.append(model.R[state, action] - Q[state, action])
                Q[state, action] += alpha * (model.R[state, action] - Q[state, action])
                if pav_pain:
                    Qp[state, action] += alpha * (-min(model.R[state, action],0) - Qp[state, action])
                # print("catastrophic death in steps:", j_step)
                break

        if modulate_w:
            w = compute_w(state, running_avg_absTDE, kappa)
        w_list.append(w)

        scores_window.append(ep_score)       # save most recent score
        scores.append(ep_score)              # save most recent score
        score_means.append(np.mean(scores_window))
        score_stds.append(np.std(scores_window))

        ep_pain_window.append(ep_pain)
        ep_pains.append(ep_pain)
        ep_pain_means.append(np.mean(ep_pain_window))
        ep_pain_stds.append(np.std(ep_pain_window))

        cum_pain += ep_pain
        cum_pain_window.append(cum_pain)
        cum_pains.append(cum_pain)
        cum_pain_means.append(np.mean(cum_pain_window))
        cum_pain_stds.append(np.std(cum_pain_window))

        ep_step_window.append(j_step)
        ep_steps.append(j_step)
        ep_step_means.append(np.mean(ep_step_window))
        ep_step_stds.append(np.std(ep_step_window))

        cum_step += j_step
        cum_step_window.append(cum_step)
        cum_steps.append(cum_step)
        cum_step_means.append(np.mean(cum_step_window))
        cum_step_stds.append(np.std(cum_step_window))

        cum_goal_one_reaches.append(cum_goal_one_reach)
        cum_goal_two_reaches.append(cum_goal_two_reach)

        # Add trajectory visualizing script
        if visualise:
            if i_episode == 1000:
                env_name = "hottiles"
                # visualise_trajectory(env_name, i_episode, model, maxiter, dynobs, modulate_w, running_avg_absTDE, kappa, Q, V,Vr,Vp, tau, mu, Qp, w, model.num_actions)

    ###############################################

    # determine the q function and policy
    q = np.max(Q, axis=1).reshape(-1,1) # of the instrumental policy
    # pi = np.argmax(Q, axis=1).reshape(-1,1) # this would be just the instrumental policy
    spi = np.zeros((model.num_states, model.num_actions))
    for state in range(model.num_states):
        if modulate_w:
            w = compute_w(state, running_avg_absTDE, kappa)
        _, action_probs = sample_action(Q, (V,Vr,Vp), state, model.num_actions, tau, mu, Qp, w)
        spi[state, :] = action_probs
    pi = np.argmax(spi, axis=1).reshape(-1,1) # combined policy

    Qp_spi = np.zeros((model.num_states, model.num_actions))
    for state in range(model.num_states):
        _, action_probs = sample_action(Qp, (V,Vr,Vp), state, model.num_actions, tau, mu, None, w=0)
        Qp_spi[state, :] = action_probs
    Qp_pi = np.argmax(Qp_spi, axis=1).reshape(-1,1) # just pav pain policy

    scores = np.array(scores)
    score_means = np.array(score_means)
    score_stds = np.array(score_stds)

    if plot_plots:
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

    if plot_plots:
        # plot the pains
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # plt.plot(np.arange(len(ep_pains)), ep_pains)
        plt.plot(np.arange(len(ep_pain_means)), ep_pain_means)
        plt.fill_between(np.arange(len(ep_pain_means)),ep_pain_means-ep_pain_stds,ep_pain_means+ep_pain_stds,alpha=.4)
        plt.ylabel('Pain accrued in an episode')
        plt.xlabel('Episode #')
        plt.show()

    cum_pains = np.array(cum_pains, dtype=float)
    cum_pain_means = np.array(cum_pain_means, dtype=float)
    cum_pain_stds = np.array(cum_pain_stds, dtype=float)

    if plot_plots:
        # plot the pains
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # plt.plot(np.arange(len(cum_pains)), cum_pains)
        plt.plot(np.arange(len(cum_pain_means)), cum_pain_means)
        plt.fill_between(np.arange(len(cum_pain_means)),cum_pain_means-cum_pain_stds,cum_pain_means+cum_pain_stds,alpha=.4)
        plt.ylabel('Cummulative pain accrued over episodes')
        plt.xlabel('Episode #')
        plt.show()

    ep_steps = np.array(ep_steps)
    ep_step_means = np.array(ep_step_means)
    ep_step_stds = np.array(ep_step_stds)

    if plot_plots:
        # plot the steps
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # plt.plot(np.arange(len(ep_steps)), ep_steps)
        plt.plot(np.arange(len(ep_step_means)), ep_step_means)
        plt.fill_between(np.arange(len(ep_step_means)),ep_step_means-ep_step_stds,ep_step_means+ep_step_stds,alpha=.4)
        plt.ylabel('Steps to goal')
        plt.xlabel('Episode #')
        plt.show()

    cum_steps = np.array(cum_steps, dtype=float)
    cum_step_means = np.array(cum_step_means, dtype=float)
    cum_step_stds = np.array(cum_step_stds, dtype=float)

    if plot_plots:
        # plot the cum steps
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # plt.plot(np.arange(len(cum_steps)), cum_steps)
        plt.plot(np.arange(len(cum_step_means)), cum_step_means)
        plt.fill_between(np.arange(len(cum_step_means)),cum_step_means-cum_step_stds,cum_step_means+cum_step_stds,alpha=.4)
        plt.ylabel('Cummulative steps')
        plt.xlabel('Episode #')
        plt.show()

        TDE_list = np.array(TDE_list)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.hist(TDE_list)
        # plt.ylabel('')
        plt.xlabel('TDE')
        plt.show()

        # plot the steps
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(w_list)), w_list)
        plt.ylabel('Flexible $\omega$ value')
        plt.xlabel('Episode count')
        plt.title('Uncertainty-based modulation of $\omega$')
        plt.show()
    
    if two_goals:
        cum_goal_one_reaches = np.array(cum_goal_one_reaches, dtype=float)
        cum_goal_two_reaches = np.array(cum_goal_two_reaches, dtype=float)
        if plot_plots:
            # plot the cum goal one reaches
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.plot(np.arange(len(cum_goal_one_reaches)), cum_goal_one_reaches)
            plt.ylabel('Cummulative goal one reaches')
            plt.xlabel('Episode #')
            plt.show()

            # plot the cum goal two reaches
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.plot(np.arange(len(cum_goal_two_reaches)), cum_goal_two_reaches)
            plt.ylabel('Cummulative goal two reaches')
            plt.xlabel('Episode #')
            plt.show()

    os.makedirs(saverun_folder, exist_ok=True)
    if modulate_w:
        os.makedirs(os.path.join(saverun_folder, 'modulate_omega_kappa='+str(kappa)+'_beta='+str(beta)), exist_ok=True)
        saverun_folder = os.path.join(saverun_folder, 'modulate_omega_kappa='+str(kappa)+'_beta='+str(beta))
    else:
        os.makedirs(os.path.join(saverun_folder, 'omega='+str(w)), exist_ok=True)
        saverun_folder = os.path.join(saverun_folder, 'omega='+str(w))
    saverun_folder_scores = os.path.join(saverun_folder, 'scores')
    saverun_folder_ep_pains = os.path.join(saverun_folder, 'ep_pains')
    saverun_folder_ep_steps = os.path.join(saverun_folder, 'ep_steps')
    saverun_folder_cum_pains = os.path.join(saverun_folder, 'cum_pains')
    saverun_folder_cum_steps = os.path.join(saverun_folder, 'cum_steps')
    if two_goals:
        saverun_folder_cum_goal_one_reaches = os.path.join(saverun_folder, 'cum_goal_one_reaches')
        saverun_folder_cum_goal_two_reaches = os.path.join(saverun_folder, 'cum_goal_two_reaches')
    os.makedirs(saverun_folder_scores, exist_ok=True)
    os.makedirs(saverun_folder_ep_pains, exist_ok=True)
    os.makedirs(saverun_folder_ep_steps, exist_ok=True)
    os.makedirs(saverun_folder_cum_pains, exist_ok=True)
    os.makedirs(saverun_folder_cum_steps, exist_ok=True)
    if two_goals:
        os.makedirs(saverun_folder_cum_goal_one_reaches, exist_ok=True)
        os.makedirs(saverun_folder_cum_goal_two_reaches, exist_ok=True)
        with open(os.path.join(saverun_folder_cum_goal_one_reaches, 'run_'+str(run_num)+'.npy'), 'wb') as f:
            np.save(f, np.array(cum_goal_one_reaches))
        with open(os.path.join(saverun_folder_cum_goal_two_reaches, 'run_'+str(run_num)+'.npy'), 'wb') as f:
            np.save(f, np.array(cum_goal_two_reaches))
    with open(os.path.join(saverun_folder_scores, 'run_'+str(run_num)+'.npy'), 'wb') as f:
        np.save(f, np.array(scores))
    with open(os.path.join(saverun_folder_ep_pains, 'run_'+str(run_num)+'.npy'), 'wb') as f:
        np.save(f, np.array(ep_pains))
    with open(os.path.join(saverun_folder_ep_steps, 'run_'+str(run_num)+'.npy'), 'wb') as f:
        np.save(f, np.array(ep_steps))
    with open(os.path.join(saverun_folder_cum_pains, 'run_'+str(run_num)+'.npy'), 'wb') as f:
        np.save(f, np.array(cum_pains))
    with open(os.path.join(saverun_folder_cum_steps, 'run_'+str(run_num)+'.npy'), 'wb') as f:
        np.save(f, np.array(cum_steps))

    return (V,Vr,Vp), q, pi, state_counts, spi, Qp, Qp_pi, -Qp_spi, absTDE


def compute_w(state, running_avg_absTDE, kappa):
    w = kappa * running_avg_absTDE
    w = min(w,1)
    return w


def sample_action(Q, Vs, state, num_actions, tau, mu, Qp = None, w=0):
    """
    Probabilistic action selection using softmax.

    Parameters
    ----------
    Q : numpy array of shape (N, num_actions)
        Q function for the environment where N is the total number of states.

    Vs : a tuple of numpy arrays of shape (N, 1)
        (V, Vr, Vp) function for the environment where N is the total number of states.

    state : int
        The current state.

    num_actions : int
        The number of actions.

    tau : float
        Temperature parameter

    Qp : np.array
        Q values of pavlovian pain policy

    w : float
        Linear weighting [0,1] for Pavlovian instrumental transfer

    Returns
    -------
    action : int
        Number representing the selected action between 0 and num_actions.
    """
    V, Vr, Vp = Vs

    # derive pavlovian policy from q values
    # use argwhere

    if Qp is not None:
        A_p = np.argwhere(Qp[state, :] == np.amin(Qp[state, :]))
        A_p = A_p.flatten().tolist()
        # print(Qp[state,:], A_p)

    if Qp is None:
        advantages = Q[state, :] - V[state]
    else:
        ## add later
        advantages = (Q[state, :] - V[state]) * (1- w)
        for a_p in A_p:
            advantages[a_p] += Vp[state] * w

    # A more safe and stable implementation of softmax
    advantages_shifted = advantages - np.max(advantages)
    transformed_advantages = mu * advantages_shifted /tau
    Q_num = np.exp(transformed_advantages)
    Q_denom = np.sum(Q_num)
    Q_dist = Q_num / Q_denom

    action_list = np.arange(num_actions)
    action = random.choices(action_list, weights = Q_dist, k=1)

    return action, Q_dist
