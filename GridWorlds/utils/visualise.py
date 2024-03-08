import numpy as np
import random
import sys
from collections import deque
import matplotlib.pyplot as plt
from utils.helper_functions import seq_to_col_row,row_col_to_seq
from utils.plots import plot_gridworld, plot_quiverpolicy
from run_codes.dynamic_environment import move_catastrophic_death, move_goal

import cv2




def visualise_trajectory(env_name, episode_num, model, maxiter, dynobs, modulate_w, running_avg_absTDE, kappa, Q, V,Vr,Vp, tau, mu, Qp, w, num_actions):
    frames = []

    state = int(model.start_state_seq)
    j_step = 0
    while True:

        data = plot_gridworld(model, visualise = True, state=seq_to_col_row(state, model.num_cols))
        # data = np.moveaxis(data, 2, 0)
        frames.append(data)


        if maxiter != None and j_step == maxiter:
            print("Reached max steps {}, terminating episode.".format(maxiter))
            break
        if dynobs:
            move_catastrophic_death(model, j_step)

        # sample action probabilistically using softmax
        # temperature parameter varies the noise
        if modulate_w:
            # w = compute_w(state, running_avg_absTDE, absTDE, TDEmax)
            w = compute_w(state, running_avg_absTDE, kappa)
            # print(w)
        action,_ = sample_action(Q, (V,Vr,Vp), state, num_actions, tau, mu, Qp, w)
        # print("Action:", action)
        j_step +=1
        # initialize p and r
        p, r = 0, np.random.random()

        # sample the next state according to the action and the
        # probability of the transition

        # REVISIT this
        if action[0] == 4: # freeze/immobility
                next_state = state
                print("Agent freezed for this timestep")
        else:
            for next_state in range(model.num_states):
                p += model.P[state, next_state, action]
                if r <= p:
                    break
            # print(next_state, seq_to_col_row(next_state, model.num_cols)

        #Store the previous state
        state = next_state
        if np.any(state == model.goal_states_seq):
            print("we won mr. stark! in steps:", j_step)
            break
        if model.restart_states_seq is not None and np.any(state == model.restart_states_seq):
            print("catastrophic death in steps:", j_step)
            break
        print(j_step)

    frames = np.array(frames)
    frames = np.flip(frames, axis=3)
    print(frames.shape, np.max(frames), np.min(frames))
    print("Saving video... ", end="")

    frameSize = (frames.shape[2], frames.shape[1])
    out = cv2.VideoWriter(env_name + "_episode_" + str(episode_num) +".avi",cv2.VideoWriter_fourcc(*'DIVX'), 16, frameSize)

    for i in range(frames.shape[0]):
        img = frames[i, :, :, :]
        out.write(img)

    out.release()

    print("Done.")
    sys.exit()
    return



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
        A_p = np.argwhere(Qp[state, :] == np.amax(Qp[state, :]))
        A_p = A_p.flatten().tolist()

    if Qp is None:
        advantages = Q[state, :] - V[state]
    else:
        ## add later
        advantages = (Q[state, :] - V[state]) * (1- w)
        for a_p in A_p:
            advantages[a_p] += Vp[state] * w

    # transformed_advantages_clipped = np.clip(mu * advantages/ tau, a_min=None, a_max=709)
    # Q_num = np.exp(transformed_advantages_clipped)

    # A more safe and stable implementation of softmax
    advantages_shifted = advantages - np.max(advantages)
    transformed_advantages = mu * advantages_shifted /tau
    Q_num = np.exp(transformed_advantages)
    Q_denom = np.sum(Q_num)
    Q_dist = Q_num / Q_denom

    action_list = np.arange(num_actions)
    action = random.choices(action_list, weights = Q_dist, k=1)

    return action, Q_dist
