import os
import sys
sys.path.append("..")
import numpy as np
from env.grid_world_with_two_rewards import GridWorld
from utils.plots import plot_gridworld, plot_quiverpolicy
from algorithms import pavlovian_system, pavlovian_instrumental_transfer_presave, pavlovian_instrumental_transfer_colearn
np.random.seed(1)



def run_Tmaze(saverun_folder, run_num, mode, w, modulate_w, kappa, beta, maxeps=1000):
        mode = 'pavlovian_instrumental_transfer_colearn'

        ###########################################################
        # Instrumental Q learning (with pain as -ve reward)       #
        # Pavlovian reward system                                 #
        # Pavlovian pain system                                   #
        ###########################################################
        # specify world parameters
        num_rows = 13
        num_cols = 15
        obstructed_states = np.array([[0,0], [0,1], [0,2], [0,3], [0,4], [0,5], 
                                        [0,6], [0,7], [0,8], [0,9], [0,10],
                                        [0,11], [0,12], [0,13], [0,14],
                                        [6,0], [6,1], [6,2], [6,3], [6,4], 
                                        [6,5], 
                                        # [6,6], [6,7], [6,8],
                                        [6,9], [6,10],
                                        [6,11], [6,12], [6,13], [6,14], 
                                        [1,0], [2,0], [3,0], [4,0], [5,0], 
                                        [6,0], 
                                        [1,14], [2,14], [3,14], [4,14], [5,14], 
                                        [6,14], 

                                        [7,5], [8,5], [9,5], [10,5], [11,5], [12,5],
                                        [7,9], [8,9], [9,9], [10,9], [11,9], [12,9],
                                        [12,6], [12,7], [12,8],
                                        ])

        bad_states = np.array([ 
                                # [1,2], [2,2], [3,2], [4,2], [5,2],
                                # [1,3], [2,3], [3,3], [4,3], [5,3],
                                # [1,4], [2,4], [3,4], [4,4], [5,4],
                                # [1,5], [2,5], [3,5], [4,5], [5,5],
                                # [1,6], [2,6], [3,6], [4,6], [5,6],

                                [1,8], [2,8], [3,8], [4,8], [5,8],
                                [1,9], [2,9], [3,9], [4,9], [5,9],
                                [1,10], [2,10], [3,10], [4,10], [5,10],
                                [1,11], [2,11], [3,11], [4,11], [5,11],
                                [1,12], [2,12], [3,12], [4,12], [5,12],

                                ])

        start_state = np.array([[11,7]])
        small_goal_states = np.array([[3,1]])
        big_goal_states= np.array([[3,13]])

        # create model
        gw = GridWorld(num_rows=num_rows,
                        num_cols=num_cols,
                        start_state=start_state,
                        small_goal_states=small_goal_states,
                        big_goal_states = big_goal_states
                        )
        gw.add_obstructions(obstructed_states=obstructed_states, bad_states=bad_states)
        gw.add_rewards(step_reward=0,
                        small_goal_reward=0.1,
                        big_goal_reward=1,
                        bad_state_reward=-0.1)

        gw.add_transition_probability(p_good_transition=0.9, 
                                        bias=0.5)
        model = gw.create_gridworld()


        # plot_gridworld(model, two_goals=True, title="Gridworld")

        if mode == 'pavlovian_instrumental_transfer_colearn':
                print("Learning the instrumental policy, colearning the pavlovian policy")

                # Learn the optimal path from Pavlovian systems 
                # while learning the instrumental system to act
                # as pavlovian propensities in pav-inst transfer

                # # solve with TD for the instrtumental system
                v_functions, q_function, pi, state_counts, spi, Qp, Qp_pi, Qp_spi, absTDE  = pavlovian_instrumental_transfer_colearn.tdlearning(model,\
                        'sarsamax', pav_pain=True, \
                        w = w, modulate_w=modulate_w,
                        random_restarts=False, alpha=0.1, gamma=0.99, mu=1,\
                        tau_0=1, tau_k=0.025, maxiter=None, maxeps=1000, two_goals=True,
                        saverun_folder = saverun_folder, run_num=run_num,
                        kappa = kappa, beta = beta)
                        # default tau_0=0.5, tau_k=0.05