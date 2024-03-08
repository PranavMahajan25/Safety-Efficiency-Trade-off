import os
import sys
sys.path.append("..")
import numpy as np
from env.grid_world import GridWorld
from utils.plots import plot_gridworld, plot_quiverpolicy
from algorithms import pavlovian_instrumental_transfer_colearn



def run_movingobsenv(saverun_folder, run_num, mode, w, modulate_w, kappa, beta, maxeps=1000):
# mode = 'pavlovian_instrumental_transfer_colearn'

        ###########################################################
        # Instrumental Q learning (with pain as -ve reward)       #
        # Pavlovian reward system                                 #
        # Pavlovian pain system                                   #
        ###########################################################

        # specify world parameters
        num_rows = 11
        num_cols = 17
        obstructed_states = np.array([[0,0], [0,1], [0,2], [0,3], [0,4], [0,5],
                                        [0,6], [0,7], [0,8], [0,9], [0,10],
                                        [0,11], [0,12], [0,13], [0,14], [0,15],
                                        [0,16],
                                        [10,1], [10,2], [10,3], [10,4], [10,5],
                                        [10,6], [10,7], [10,8], [10,9], [10,10],
                                        [10,11], [10,12], [10,13], [10,14], [10,15],
                                        [10,16],
                                        [1,0], [2,0], [3,0], [4,0], [5,0],
                                        [6,0], [7,0], [8,0], [9,0], [10,0],
                                        [1,16], [2,16], [3,16], [4,16], [5,16],
                                        [6,16], [7,16], [8,16], [9,16], [10,16],

                                        [7,2], [7,3], [7,4], [7,5], [7,6], [7,7],
                                        [7,9], [7,10], [7,11], [7,12], [7,13], [7,14],
                                        [3,2], [3,3], [3,4], [3,5], [3,6], [3,7],
                                        [3,9], [3,10], [3,11], [3,12], [3,13], [3,14],
                                        [5,7], [5,8], [5,9],
                                        ])

        # moving_obstacle =  np.array([[4,8], [4,7], [4,6], [5,6], [6,6], [6,7], [6,8],
        #                  [6,9], [6,10], [5,10], [4,10], [4,9]])
        moving_obstacle =  np.array([[4,8]])

        start_state = np.array([[9,9]])
        goal_states = np.array([[1,7]])

        # create model
        gw = GridWorld(num_rows=num_rows,
                        num_cols=num_cols,
                        start_state=start_state,
                        goal_states=goal_states)
        gw.add_obstructions(restart_states=moving_obstacle, obstructed_states=obstructed_states)
        gw.add_rewards(step_reward=0,
                        goal_reward=1,
                        restart_state_reward=-1,
                        wallbump_reward=-0.1)

        gw.add_transition_probability(p_good_transition=1,
                                        bias=0.5)
        model = gw.create_gridworld()


        # plot_gridworld(model)

        if mode == 'pavlovian_instrumental_transfer_colearn':
                print("Learning the instrumental policy, colearning the pavlovian policy")

                # Learn the optimal path from Pavlovian systems
                # while learning the instrumental system to act
                # as pavlovian propensities in pav-inst transfer

                # # solve with TD for the instrtumental system
                v_functions, q_function, pi, state_counts, spi, Qp, Qp_pi, Qp_spi, absTDE = pavlovian_instrumental_transfer_colearn.tdlearning(model,\
                        'sarsamax', pav_pain=True, \
                        w = w, modulate_w=modulate_w,
                        random_restarts=False, alpha=0.1, gamma=0.99, mu=1,\
                        tau_0=1, tau_k=0.025, maxiter=None, maxeps=maxeps, # default tau_0=0.5, tau_k=0.05
                        dynobs=True,
                        saverun_folder = saverun_folder, run_num=run_num,
                        kappa = kappa, beta = beta
                        )


        saverun_folder_figures = os.path.join(saverun_folder, 'figures')
        os.makedirs(saverun_folder_figures, exist_ok=True)
        if run_num==0:
                path = os.path.join(saverun_folder_figures, 'moving_obs_env_absTDEplot.png')
                plot_gridworld(model, value_function=absTDE,\
                        title="Pavlovian Instrumental system - Absolute TDE values", path=path, secondary_goal=None, plot_plots=False)
                path = os.path.join(saverun_folder_figures, 'moving_obs_env_1.png')
                plot_gridworld(model, policy=pi, state_counts=state_counts,\
                        title="Pavlovian Instrumental system - State counts", path=path, secondary_goal=None, plot_plots=False)
                path = os.path.join(saverun_folder_figures, 'moving_obs_env_2.png')
                plot_gridworld(model, value_function=v_functions[0], policy=pi,\
                        title="Pavlovian Instrumental system - Value function V(s)", path=path, secondary_goal=None, plot_plots=False)
                path = os.path.join(saverun_folder_figures, 'moving_obs_env_3.png')
                plot_gridworld(model, value_function=v_functions[1], policy=pi,\
                        title="Pavlovian Instrumental system - Vr(s) function", path=path, secondary_goal=None, plot_plots=False)
                path = os.path.join(saverun_folder_figures, 'moving_obs_env_4.png')
                plot_gridworld(model, value_function=v_functions[2], policy=pi,\
                        title="Pavlovian Instrumental system - Vp(s) function", path=path, secondary_goal=None, plot_plots=False )
                path = os.path.join(saverun_folder_figures, 'moving_obs_env_spi.png')
                plot_quiverpolicy(model, stochastic_policy=spi,\
                        title="Pavlovian Instrumental system - Stochastic policy", path=path, secondary_goal=None, plot_plots=False)
                path = os.path.join(saverun_folder_figures, 'moving_obs_env_spi_1.png')
                plot_quiverpolicy(model, stochastic_policy=Qp_spi,\
                        title="Pavlovian Pain system (co-learned)- Stochastic policy", path=path, secondary_goal=None, plot_plots=False)
