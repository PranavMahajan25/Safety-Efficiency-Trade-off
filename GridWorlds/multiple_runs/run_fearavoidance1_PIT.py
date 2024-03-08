import os
import sys
sys.path.append("..")
import numpy as np
from env.grid_world_with_immobility import GridWorld
from utils.plots import plot_gridworld, plot_quiverpolicy
from algorithms import pavlovian_instrumental_transfer_colearn_truncated


def run_FAenv(saverun_folder, run_num, mode, w, modulate_w, kappa, beta):
# mode = 'pavlovian_instrumental_transfer_colearn'
        np.random.seed(run_num) # use run_num as random seed

        ###########################################################
        # Instrumental Q learning (with pain as -ve reward)       #
        # No Pavlovian reward system                              #
        # Pavlovian pain system                                   #
        ###########################################################
        # specify world parameters
        num_rows = 7
        num_cols = 15
        obstructed_states = np.array([[0,0], [0,1], [0,2], [0,3], [0,4], [0,5], 
                                        [0,6], [0,7], [0,8], [0,9], [0,10],
                                        [0,11], [0,12], [0,13], [0,14],
                                        [6,0], [6,1], [6,2], [6,3], [6,4], [6,5], 
                                        [6,6], [6,7], [6,8], [6,9], [6,10],
                                        [6,11], [6,12], [6,13], [6,14], 
                                        [1,0], [2,0], [3,0], [4,0], [5,0], 
                                        [6,0], 
                                        [1,14], [2,14], [3,14], [4,14], [5,14], 
                                        [6,14], 
                                        
                                        ])

        bad_states = np.array([ 
                                [1,2], [2,2], [3,2], [4,2], [5,2],
                                [1,3], [2,3], [3,3], [4,3], [5,3],
                                [1,4], [2,4], [3,4], [4,4], [5,4],
                                [1,5], [2,5], [3,5], [4,5], [5,5],
                                [1,6], [2,6], [3,6], [4,6], [5,6],
                                ])

        start_state = np.array([[3,7]])
        goal_states = np.array([[3,1]])

        # create model
        gw = GridWorld(num_rows=num_rows,
                        num_cols=num_cols,
                        start_state=start_state,
                        goal_states=goal_states)
        gw.add_obstructions(obstructed_states=obstructed_states, bad_states=bad_states)
        gw.add_rewards(step_reward=0,
                        goal_reward=1,
                        bad_state_reward=-0.1)

        gw.add_transition_probability(p_good_transition=0.9, 
                                        bias=0.5)
        model = gw.create_gridworld()


        plot_gridworld(model)


        if mode == 'pavlovian_instrumental_transfer_colearn_truncated':
                print("Learning the instrumental policy, colearning the pavlovian policy")

                # Learn the optimal path from Pavlovian systems 
                # while learning the instrumental system to act
                # as pavlovian propensities in pav-inst transfer

                # # solve with TD for the instrtumental system
                v_functions, q_function, pi, state_counts, spi, Qp, Qp_pi, Qp_spi, absTDE  = pavlovian_instrumental_transfer_colearn_truncated.tdlearning(model,\
                        'sarsamax', pav_pain=True, \
                        w = w, modulate_w=modulate_w,
                        random_restarts=False, alpha=0.1, gamma=0.99, mu=1,\
                        tau_0=1, tau_k=0.025, maxiter=100, maxeps=1000,
                        saverun_folder = saverun_folder, run_num=run_num,
                        kappa=kappa, beta=beta, fix_Pav_action=False)
                        # default tau_0=0.5, tau_k=0.05
        
        saverun_folder_figures = os.path.join(saverun_folder, 'figures')
        os.makedirs(saverun_folder_figures, exist_ok=True)
        if run_num==0:    
                path = os.path.join(saverun_folder_figures, 'fear_avoidance_env_absTDEplot.png')
                plot_gridworld(model, value_function=absTDE,\
                        title="Pavlovian Instrumental system - Absolute TDE values", path=path, secondary_goal=None, plot_plots=False)
                path = os.path.join(saverun_folder_figures, 'fear_avoidance_env_1.png')
                plot_gridworld(model, policy=pi, state_counts=state_counts,\
                        title="Pavlovian Instrumental system - State counts", path=path, secondary_goal=None, plot_plots=False)
                path = os.path.join(saverun_folder_figures, 'fear_avoidance_env_2.png')
                plot_gridworld(model, value_function=v_functions[0], policy=pi,\
                        title="Pavlovian Instrumental system - Value function V(s)", path=path, secondary_goal=None, plot_plots=False)
                path = os.path.join(saverun_folder_figures, 'fear_avoidance_env_3.png')
                plot_gridworld(model, value_function=v_functions[1], policy=pi,\
                        title="Pavlovian Instrumental system - Vr(s) function", path=path, secondary_goal=None, plot_plots=False)
                path = os.path.join(saverun_folder_figures, 'fear_avoidance_env_4.png')
                plot_gridworld(model, value_function=v_functions[2], policy=pi,\
                        title="Pavlovian Instrumental system - Vp(s) function", path=path, secondary_goal=None, plot_plots=False )
                path = os.path.join(saverun_folder_figures, 'fear_avoidance_env_spi.png')
                plot_quiverpolicy(model, stochastic_policy=spi,\
                        title="Pavlovian Instrumental system - Stochastic policy", path=path, secondary_goal=None, plot_plots=False)
                path = os.path.join(saverun_folder_figures, 'fear_avoidance_env_spi_1.png')
                plot_quiverpolicy(model, stochastic_policy=Qp_spi,\
                        title="Pavlovian Pain system (co-learned)- Stochastic policy", path=path, secondary_goal=None, plot_plots=False)
                