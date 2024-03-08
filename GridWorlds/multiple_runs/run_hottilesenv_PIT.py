import os
import sys
sys.path.append("..")
import numpy as np
from env.grid_world import GridWorld
from utils.plots import plot_gridworld, plot_quiverpolicy
from algorithms import pavlovian_instrumental_transfer_presave, pavlovian_instrumental_transfer_colearn




def run_hottilesenv(saverun_folder, run_num, mode, w, modulate_w, kappa, beta, maxeps=1000):
        # mode = 'pavlovian_instrumental_transfer_presave'
        # mode = 'pavlovian_instrumental_transfer_presave_goal_change'
        # mode = 'pavlovian_instrumental_transfer_colearn'
        # mode = 'pavlovian_instrumental_transfer_colearn_goal_change'

        np.random.seed(run_num) # use run_num as random seed

        ###########################################################
        # Instrumental Q learning (with pain as -ve reward)       #
        # No Pavlovian reward system                              #
        # Pavlovian pain system                                   #
        ###########################################################
        # specify world parameters
        num_rows = 20
        num_cols = 20
        obstructed_states = np.array([[0,0], [0,1], [0,2], [0,3], [0,4], [0,5], 
                                        [0,6], [0,7], [0,8], [0,9], [0,10],
                                        [0,11], [0,12], [0,13], [0,14], [0,15], 
                                        [0,16], [0,17], [0,18], [0,19],
                                        [19,0], [19,1], [19,2], [19,3], [19,4], [19,5], 
                                        [19,6], [19,7], [19,8], [19,9], [19,10],
                                        [19,11], [19,12], [19,13], [19,14], [19,15], 
                                        [19,16], [19,17], [19,18], [19,19],
                                        [1,0], [2,0], [3,0], [4,0], [5,0], 
                                        [6,0], [7,0], [8,0], [9,0], [10,0],
                                        [11,0], [12,0], [13,0], [14,0], [15,0], 
                                        [16,0], [17,0], [18,0],
                                        [1,19], [2,19], [3,19], [4,19], [5,19], 
                                        [6,19], [7,19], [8,19], [9,19], [10,19],
                                        [11,19], [12,19], [13,19], [14,19], [15,19], 
                                        [16,19], [17,19], [18,19]])
        bad_states = np.array([[4,1],[4,2],[4,3],[4,4],[4,5],[4,6],
                                [5,1],[5,2],[5,3],[5,4],[5,5],[5,6],
                                [6,1],[6,2],[6,3],[6,4],[6,5],[6,6],
                                [7,1],[7,2],[7,3],[7,4],[7,5],[7,6],
                                [8,1],[8,2],[8,3],[8,4],[8,5],[8,6],
                                [9,1],[9,2],[9,3],[9,4],[9,5],[9,6],
                                [10,1],[10,2],[10,3],[10,4],[10,5],[10,6],
                                [11,1],[11,2],[11,3],[11,4],[11,5],[11,6],
                                [12,1],[12,2],[12,3],[12,4],[12,5],[12,6],
                                [13,1],[13,2],[13,3],[13,4],[13,5],[13,6],
                                [14,1],[14,2],[14,3],[14,4],[14,5],[14,6],
                                [15,1],[15,2],[15,3],[15,4],[15,5],[15,6],
                                [16,1],[16,2],[16,3],[16,4],[16,5],[16,6],
                                [3,15], [3,16], [3,17], [3,18],
                                [4,15], [4,16], [4,17], [4,18],
                                [5,15], [5,16], [5,17], [5,18],
                                [6,15], [6,16], [6,17], [6,18],
                                [13,15], [13,16], [13,17], [13,18],
                                [14,15], [14,16], [14,17], [14,18],
                                [15,15], [15,16], [15,17], [15,18],
                                [16,15], [16,16], [16,17], [16,18],
                                [9,10], [9,11], [9,12],
                                [10,10], [10,11], [10,12],
                                [11,10], [11,11], [11,12]])

        start_state = np.array([[1,18]])
        # start_state = np.array([[16,8]])
        goal_states = np.array([[18,1]])

        # create model
        gw = GridWorld(num_rows=num_rows,
                        num_cols=num_cols,
                        start_state=start_state,
                        goal_states=goal_states)
        gw.add_obstructions(bad_states=bad_states, obstructed_states=obstructed_states)
        gw.add_rewards(step_reward=0,
                        goal_reward=1,
                        bad_state_reward=-0.1)

        gw.add_transition_probability(p_good_transition=0.9, #0.9
                                        bias=0.5)
        model = gw.create_gridworld()


        # plot_gridworld(model)

        if mode == 'pavlovian_instrumental_transfer_presave':
                print("Learning the instrumental policy, using presaved pavlovian policy")

                # Use the optimal path from Pavlovian systems 
                # as pavlovian propensities in pav-inst transfer

                # with open('pavlovian_reward_qvalues_hot_tiles_env.npy', 'rb') as f:
                #         pavlovian_reward_qvalues = np.load(f)
                with open('pavlovian_pain_qvalues_hot_tiles_env.npy', 'rb') as f:
                        pavlovian_pain_qvalues = np.load(f)

                # # solve with TD for the instrtumental system
                v_functions, q_function, pi, state_counts, spi, absTDE = pavlovian_instrumental_transfer_presave.tdlearning(model,'sarsamax',\
                        pavlovian_reward_qvalues = None,\
                        pavlovian_pain_qvalues = pavlovian_pain_qvalues, \
                        w = w, modulate_w=modulate_w,\
                        random_restarts=False, alpha=0.1, gamma=0.99, mu=1,\
                        tau_0=1, tau_k=0.025, maxiter=None, maxeps=1000, 
                        kappa=kappa, beta=beta
                        )

        # if mode == 'pavlovian_instrumental_transfer_colearn':
        #         print("\nLearning the instrumental policy, colearning the pavlovian policy - Run #"+str(run_num))

        #         # Learn the optimal path from Pavlovian systems 
        #         # while learning the instrumental system to act
        #         # as pavlovian propensities in pav-inst transfer

        #         # # solve with TD for the instrtumental system
        #         v_functions, q_function, pi, state_counts, spi, Qp, Qp_pi, Qp_spi, absTDE = pavlovian_instrumental_transfer_colearn.tdlearning(model,\
        #                 'sarsamax', pav_pain=True, \
        #                 w =w, modulate_w=modulate_w,
        #                 random_restarts=False, alpha=0.1, gamma=0.99, mu=1,\
        #                 tau_0=1, tau_k=0.025, maxiter=None, maxeps=maxeps,  # default tau_0=0.5, tau_k=0.05
        #                 saverun_folder = saverun_folder, run_num=run_num,
        #                 kappa = kappa, beta = beta
        #                 )
                        
                
        # if mode == 'pavlovian_instrumental_transfer_colearn_goal_change':
        #         print("\nLearning the instrumental policy, colearning the pavlovian policy -- Goal change env variant- Run #"+str(run_num))

        #         # Learn the optimal path from Pavlovian systems 
        #         # while learning the instrumental system to act
        #         # as pavlovian propensities in pav-inst transfer

        #         new_goal_states = np.array([[18,18]])
        #         # # solve with TD for the instrtumental system
        #         v_functions, q_function, pi, state_counts, spi, Qp, Qp_pi, Qp_spi, absTDE = pavlovian_instrumental_transfer_colearn.tdlearning(model,\
        #                 'sarsamax', pav_pain=True, \
        #                 w = w, modulate_w=modulate_w,
        #                 random_restarts=False, alpha=0.1, gamma=0.99, mu=1,\
        #                 tau_0=1, tau_k=0.025, maxiter=None, maxeps=1000, # default tau_0=0.5, tau_k=0.05
        #                 goalchange=new_goal_states, # watch out!
        #                 saverun_folder = saverun_folder, run_num=run_num,
        #                 kappa = kappa, beta = beta
        #                 )
                        
        saverun_folder_figures = os.path.join(saverun_folder, 'figures')
        os.makedirs(saverun_folder_figures, exist_ok=True)
        if run_num==0:       
                path = os.path.join(saverun_folder_figures, 'hot_tiles_env_absTDEplot.png')
                plot_gridworld(model, value_function=absTDE,\
                        title="Pavlovian Instrumental system - Absolute TDE values", path=path, secondary_goal=None, plot_plots=False)
                path = os.path.join(saverun_folder_figures, 'hot_tiles_env_1.png')
                plot_gridworld(model, policy=pi, state_counts=state_counts,\
                        title="Pavlovian Instrumental system - State counts", path=path, secondary_goal=None, plot_plots=False)
                path = os.path.join(saverun_folder_figures, 'hot_tiles_env_2.png')
                plot_gridworld(model, value_function=v_functions[0], policy=pi,\
                        title="Pavlovian Instrumental system - Value function V(s)", path=path, secondary_goal=None, plot_plots=False)
                path = os.path.join(saverun_folder_figures, 'hot_tiles_env_3.png')
                plot_gridworld(model, value_function=v_functions[1], policy=pi,\
                        title="Pavlovian Instrumental system - Vr(s) function", path=path, secondary_goal=None, plot_plots=False)
                path = os.path.join(saverun_folder_figures, 'hot_tiles_env_4.png')
                plot_gridworld(model, value_function=v_functions[2], policy=pi,\
                        title="Pavlovian Instrumental system - Vp(s) function", path=path, secondary_goal=None, plot_plots=False )
                path = os.path.join(saverun_folder_figures, 'hot_tiles_env_spi.png')
                plot_quiverpolicy(model, stochastic_policy=spi,\
                        title="Pavlovian Instrumental system - Stochastic policy", path=path, secondary_goal=None, plot_plots=False)
                path = os.path.join(saverun_folder_figures, 'hot_tiles_env_spi_1.png')
                plot_quiverpolicy(model, stochastic_policy=Qp_spi,\
                        title="Pavlovian Pain system (co-learned)- Stochastic policy", path=path, secondary_goal=None, plot_plots=False)
                