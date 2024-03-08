import os
import sys
sys.path.append("..")
import numpy as np
from env.grid_world import GridWorld
from utils.plots import plot_gridworld, plot_quiverpolicy
from algorithms import pavlovian_instrumental_transfer_colearn


def run_wallmazeenv(saverun_folder, run_num, mode, w, modulate_w, kappa, beta, maxeps=1000):
# mode = 'pavlovian_instrumental_transfer_colearn'

        ###########################################################
        # Instrumental Q learning (with pain as -ve reward)       #
        # Pavlovian reward system                                 #
        # Pavlovian pain system                                   #
        ###########################################################
        # specify world parameters
        num_rows = 28
        num_cols = 31
        obstructed_states = np.array([[0,0], [0,1], [0,2], [0,3], [0,4], [0,5],
                                        [0,6], [0,7], [0,8], [0,9], [0,10],
                                        [0,11], [0,12], [0,13], [0,14], [0,15],
                                        [0,16], [0,17], [0,18], [0,19], [0,20],
                                        [0,21], [0,22], [0,23], [0,24], [0,25],
                                        [0,26], [0,27], [0,28], [0,29], [0,30],
                                        [27,0], [27,1], [27,2], [27,3], [27,4], [27,5],
                                        [27,6], [27,7], [27,8], [27,9], [27,10],
                                        [27,11], [27,12], [27,13], [27,14], [27,15],
                                        [27,16], [27,17], [27,18], [27,19], [27,20],
                                        [27,21], [27,22], [27,23], [27,24], [27,25],
                                        [27,26], [27,27], [27,28], [27,29],

                                        [1,0], [2,0], [3,0], [4,0], [5,0],
                                        [6,0], [7,0], [8,0], [9,0], [10,0],
                                        [11,0], [12,0], [13,0], [14,0], [15,0],
                                        [16,0], [17,0], [18,0],[19,0], [20,0],
                                        [21,0], [22,0], [23,0], [24,0], [25,0],
                                        [26,0], [27,0],

                                        [1,30], [2,30], [3,30], [4,30], [5,30],
                                        [6,30], [7,30], [8,30], [9,30], [10,30],
                                        [11,30], [12,30], [13,30], [14,30], [15,30],
                                        [16,30], [17,30], [18,30], [19,30], [20,30],
                                        [21,30], [22,30], [23,30], [24,30], [25,30],
                                        [26,30], [27,30],

                                        [4,6], [5,6], [6,6], [7,6], [8,6],
                                        [10,6], [11,6], [12,6], [13,6], [14,6],

                                        [1,12], [2,12], [3,12],
                                        [4,12], [5,12], [13,12], [14,12],
                                        [15,12], [16,12], [17,12],
                                        [19,12], [20,12], [21,12], [22,12], [23,12],
                                        [4,18], [5,18], [6,18], [7,18], [8,18],
                                        [10,18], [11,18], [12,18], [13,18], [14,18],
                                        [22,18], [23,18], [24,18], [25,18], [26,18],
                                        [1,24], [2,24], [3,24], [4,24], [5,24],
                                        [13,24], [14,24], [15,24], [16,24], [17,24],
                                        [19,24], [20,24], [21,24], [22,24], [23,24],


                                        [9,6], [9,7], [9,8], [9,9], [9,10],
                                        [9,11], [9,12], [9,13], [9,14], [9,15],
                                        [9,16], [9,17], [9,18], [9,19], [9,20],
                                        [9,21], [9,22], [9,23], [9,24], [9,25],
                                        [9,26], [9,27], [9,28], [9,29],
                                        [18,6], [18,7], [18,8], [18,9], [18,10],
                                        [18,11], [18,12], [18,13], [18,14], [18,15],
                                        [18,16], [18,17], [18,18], [18,19], [18,20],
                                        [18,21], [18,22], [18,23], [18,24], [18,25],
                                        [18,26], [18,27], [18,28], [18,29]
                                        ])


        start_state = np.array([[1,1]])
        goal_states = np.array([[26,29]])

        # create model
        gw = GridWorld(num_rows=num_rows,
                        num_cols=num_cols,
                        start_state=start_state,
                        goal_states=goal_states)
        gw.add_obstructions(obstructed_states=obstructed_states)
        gw.add_rewards(step_reward=0,
                        goal_reward=1,
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
                v_functions, q_function, pi, state_counts, spi, Qp, Qp_pi, Qp_spi, absTDE  = pavlovian_instrumental_transfer_colearn.tdlearning(model,\
                        'sarsamax', pav_pain=True, \
                        w = w, modulate_w=modulate_w,
                        random_restarts=False, alpha=0.1, gamma=0.99, mu=1,\
                        tau_0=1, tau_k=0.025, maxiter=None, maxeps=maxeps, # default tau_0=0.5, tau_k=0.05
                        saverun_folder = saverun_folder, run_num=run_num,
                        kappa = kappa, beta = beta
                        )


        saverun_folder_figures = os.path.join(saverun_folder, 'figures')
        os.makedirs(saverun_folder_figures, exist_ok=True)
        if run_num==0:
                path = os.path.join(saverun_folder_figures, 'wall_maze_env_absTDEplot.png')
                plot_gridworld(model, value_function=absTDE,\
                        title="Pavlovian Instrumental system - Absolute TDE values", path=path, secondary_goal=None, plot_plots=False)
                path = os.path.join(saverun_folder_figures, 'wall_maze_env_1.png')
                plot_gridworld(model, policy=pi, state_counts=state_counts,\
                        title="Pavlovian Instrumental system - State counts", path=path, secondary_goal=None, plot_plots=False)
                path = os.path.join(saverun_folder_figures, 'wall_maze_env_2.png')
                plot_gridworld(model, value_function=v_functions[0], policy=pi,\
                        title="Pavlovian Instrumental system - Value function V(s)", path=path, secondary_goal=None, plot_plots=False)
                path = os.path.join(saverun_folder_figures, 'wall_maze_env_3.png')
                plot_gridworld(model, value_function=v_functions[1], policy=pi,\
                        title="Pavlovian Instrumental system - Vr(s) function", path=path, secondary_goal=None, plot_plots=False)
                path = os.path.join(saverun_folder_figures, 'wall_maze_env_4.png')
                plot_gridworld(model, value_function=v_functions[2], policy=pi,\
                        title="Pavlovian Instrumental system - Vp(s) function", path=path, secondary_goal=None, plot_plots=False )
                path = os.path.join(saverun_folder_figures, 'wall_maze_env_spi.png')
                plot_quiverpolicy(model, stochastic_policy=spi,\
                        title="Pavlovian Instrumental system - Stochastic policy", path=path, secondary_goal=None, plot_plots=False)
                path = os.path.join(saverun_folder_figures, 'wall_maze_env_spi_1.png')
                plot_quiverpolicy(model, stochastic_policy=Qp_spi,\
                        title="Pavlovian Pain system (co-learned)- Stochastic policy", path=path, secondary_goal=None, plot_plots=False)
