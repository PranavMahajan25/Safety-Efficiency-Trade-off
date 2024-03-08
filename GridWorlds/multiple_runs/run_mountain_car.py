import sys
sys.path.append("../../rbf-approximator/")
import gym
import gym_minigrid
from algorithms import rbf_pavlovian_instrumental_transfer_colearn
import numpy as np
np.random.seed(1)

def run_mountaincarenv(saverun_folder, run_num, mode, w, modulate_w, kappa, beta, maxeps=300):
# mode  = 'pavlovian_instrumental_transfer_colearn'

    np.random.seed(run_num) # use run_num as random seed

    ###########################################################
    # Instrumental Q learning (with pain as -ve reward)       #
    # Pavlovian reward system                                 #
    # Pavlovian pain system                                   #
    ###########################################################

    env = gym.make('Modified-MountainCar-v1')


    num_episodes = maxeps
    discount_factor = 0.995
    alpha = 0.01
    mu=1
    tau_0=1
    tau_k=1 #0.025 

    if mode == 'pavlovian_instrumental_transfer_colearn':
        print("Learning the instrumental policy, colearning the pavlovian policy")

        # # solve with TD for the pavlovian system
        learned_weights = rbf_pavlovian_instrumental_transfer_colearn.tdlearning(env, 
                'sarsamax', pav_pain=True,\
                w=w, modulate_w=modulate_w,\
                alpha=alpha, gamma=discount_factor, mu=mu, tau_0=tau_0, tau_k=tau_k,\
                maxiter=None, maxeps=num_episodes,
                saverun_folder = saverun_folder, run_num=run_num,
                kappa = kappa, beta = beta
                )
    

    env.close()