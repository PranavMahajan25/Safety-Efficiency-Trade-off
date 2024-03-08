import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from run_hottilesenv_PIT import run_hottilesenv
from run_longroute_PIT import run_longrouteenv
from run_movingobstacle_PIT import run_movingobsenv
from run_wallmaze_PIT import run_wallmazeenv
from run_fearavoidance1_PIT import run_FAenv
# from run_mountain_car import run_mountaincarenv

# run_num = 0
# expt = 'all'

# # run_hottilesenv('./', run_num, 'pavlovian_instrumental_transfer_presave', w=0, modulate_w=True, kappa=6, beta=0.4, maxeps=1000)
# # run_hottilesenv('./', run_num, 'pavlovian_instrumental_transfer_colearn_goal_change', w=0, modulate_w=True, kappa=6, beta=0.4, maxeps=1000)
# # run_longrouteenv('./', run_num, 'pavlovian_instrumental_transfer_colearn', w=0, modulate_w=True, kappa=6.5, beta=0.8, maxeps=1000)
# # run_movingobsenv('./', run_num, 'pavlovian_instrumental_transfer_colearn', w=0, modulate_w=True, kappa=4, beta=0.05, maxeps=1000)


# beta_range = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# kappa_range = [0.5, 1, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
# # beta_range = [0.1, 0.2]
# # kappa_range = [2, 4]

# # Moderately Painful States Environments (Hot tiles env)
# if expt == 'expt1' or expt == 'all':
#     print("\nModerately Painful States Environments (Hot tiles env)")
#     mode = 'pavlovian_instrumental_transfer_presave'
#     os.makedirs('grid_search_outputs', exist_ok=True)
#     saverun_folder = os.path.join('grid_search_outputs','painful_states_gridworld')
#     for beta in beta_range:
#         for kappa in kappa_range:
#             run_hottilesenv(saverun_folder, run_num, mode, w=0, modulate_w=True, kappa=kappa, beta=beta, maxeps=500)

# saverun_folder = os.path.join('grid_search_outputs','painful_states_gridworld')
# grid_cum_steps = np.zeros([len(beta_range), len(kappa_range)])
# grid_cum_pains = np.zeros([len(beta_range), len(kappa_range)])
# i=0
# for beta in beta_range:
#     j=0
#     for kappa in kappa_range:
#         mw_config = 'modulate_omega_kappa='+str(kappa)+'_beta='+str(beta)+'/'
#         mw_1_steps = np.load(saverun_folder + '/' + mw_config + 'ep_steps/run_'+str(run_num)+'.npy')
#         mw_1_pains = np.squeeze(np.load(saverun_folder + '/' + mw_config + 'ep_pains/run_'+str(run_num)+'.npy'))
#         grid_cum_steps[i][j] = np.sum(mw_1_steps)
#         grid_cum_pains[i][j] = np.sum(mw_1_pains)
#         j+=1
#     i+=1

# with open('grid_cum_steps_expt1.npy', 'wb') as f:
#     np.save(f, grid_cum_steps)
# with open('grid_cum_pains_expt1.npy', 'wb') as f:
#     np.save(f, grid_cum_pains)

beta_range = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
kappa_range = [0.5, 1, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 5, 5.5, 6, 6.5, 7]

with open('grid_cum_steps_expt1.npy', 'rb') as f:
    grid_cum_steps = np.load(f)
with open('grid_cum_pains_expt1.npy', 'rb') as f:
    grid_cum_pains = np.load(f)

grid_cum_pains = grid_cum_pains[:11, :20]
grid_cum_steps = grid_cum_steps[:11, :20]


k=10
sortedbypain_indices = np.squeeze(np.dstack(np.unravel_index(np.argsort(grid_cum_pains.ravel()), (len(beta_range), len(kappa_range)))))
print(sortedbypain_indices[:k])

max_cum_pain = 8111.7
max_cum_step = 4221355.0
grid_cum_tradeoff = (grid_cum_pains/max_cum_pain) * (grid_cum_pains/max_cum_pain) + (grid_cum_steps/max_cum_step) * (grid_cum_steps/max_cum_step)
sortedbytradeoff_indices = np.squeeze(np.dstack(np.unravel_index(np.argsort(grid_cum_tradeoff.ravel()), (len(beta_range), len(kappa_range)))))
print(sortedbytradeoff_indices[:k])

for index in sortedbytradeoff_indices[:k]:
    beta = beta_range[index[0]]
    kappa = kappa_range[index[1]]
    print("beta=", beta, " kappa=", kappa," cum_pain=", grid_cum_pains[index[0], index[1]], " cum_step=", grid_cum_steps[index[0], index[1]])



im = plt.imshow(grid_cum_steps/max_cum_step, interpolation='none')
plt.colorbar(im)
plt.xticks(np.arange(len(kappa_range)), kappa_range)
plt.yticks(np.arange(len(beta_range)), beta_range)
plt.show()
im = plt.imshow(grid_cum_pains/max_cum_pain, interpolation='none') 
plt.colorbar(im)
plt.xticks(np.arange(len(kappa_range)), kappa_range)
plt.yticks(np.arange(len(beta_range)), beta_range)
plt.show()
im = plt.imshow(1/grid_cum_tradeoff, interpolation='none') 
plt.colorbar(im)
plt.xticks(np.arange(len(kappa_range)), kappa_range)
plt.yticks(np.arange(len(beta_range)), beta_range)
plt.show()

# '''
# Top candidates - hottiles environment:
# [[10 17]
#  [ 8  8]
#  [ 4 17]
#  [ 5 17]
#  [ 5 16]
#  [ 8 17]
#  [ 3 16]
#  [10  8]
#  [ 3  4]
#  [ 7  4]]
# beta= 0.9  kappa= 6  cum_pain= 4237.1999999999825  cum_step= 378078.0
# beta= 0.7  kappa= 3  cum_pain= 4247.399999999992  cum_step= 294653.0  (actually hit or miss, 5.5k cum pain in a run)
# beta= 0.3  kappa= 6  cum_pain= 4283.099999999987  cum_step= 380228.0
# beta= 0.4  kappa= 6  cum_pain= 4287.79999999998  cum_step= 355841.0 (pretty decent I guess)
# beta= 0.4  kappa= 5.5  cum_pain= 4294.2999999999965  cum_step= 341949.0
# beta= 0.7  kappa= 6  cum_pain= 4303.2999999999765  cum_step= 384838.0
# beta= 0.2  kappa= 5.5  cum_pain= 4378.699999999988  cum_step= 313995.0
# beta= 0.9  kappa= 3  cum_pain= 4456.999999999997  cum_step= 322144.0
# beta= 0.2  kappa= 2  cum_pain= 4464.599999999984  cum_step= 283825.0
# beta= 0.6  kappa= 2  cum_pain= 4473.19999999999  cum_step= 284722.0
# '''

# ############
# #
# # beta_range = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# # kappa_range = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5]
# #
# #
# # ## Grid world with short route vs long route
# # if expt == 'expt3' or expt == 'all':
# #     print("\nGrid world with short route vs long route")
# #     mode = 'pavlovian_instrumental_transfer_colearn'
# #     os.makedirs('grid_search_outputs', exist_ok=True)
# #     saverun_folder = os.path.join('grid_search_outputs','longroute_vs_shortroute_gridworld')
# #     kappa = 4
# #     beta = 0.1
# #     for beta in beta_range:
# #         for kappa in kappa_range:
# #             run_longrouteenv(saverun_folder, run_num, mode, w=0, modulate_w=True, kappa=kappa, beta=beta, maxeps=500)
# #
# #
# # saverun_folder = os.path.join('grid_search_outputs','longroute_vs_shortroute_gridworld')
# # grid_cum_steps = np.zeros([len(beta_range), len(kappa_range)])
# # grid_cum_pains = np.zeros([len(beta_range), len(kappa_range)])
# # i=0
# # for beta in beta_range:
# #     j=0
# #     for kappa in kappa_range:
# #         mw_config = 'modulate_omega_kappa='+str(kappa)+'_beta='+str(beta)+'/'
# #         mw_1_steps = np.load(saverun_folder + '/' + mw_config + 'ep_steps/run_'+str(run_num)+'.npy')
# #         mw_1_pains = np.squeeze(np.load(saverun_folder + '/' + mw_config + 'ep_pains/run_'+str(run_num)+'.npy'))
# #         grid_cum_steps[i][j] = np.sum(mw_1_steps)
# #         grid_cum_pains[i][j] = np.sum(mw_1_pains)
# #         j+=1
# #     i+=1
# #
# # k=20
# # # print(grid_cum_pains)
# # sortedbypain_indices = np.squeeze(np.dstack(np.unravel_index(np.argsort(grid_cum_pains.ravel()), (len(beta_range), len(kappa_range)))))
# # print(sortedbypain_indices[:k])
# # for index in sortedbypain_indices[:k]:
# #     beta = beta_range[index[0]]
# #     kappa = kappa_range[index[1]]
# #     print("beta=", beta, " kappa=", kappa," cum_pain=", grid_cum_pains[index[0], index[1]], " cum_step=", grid_cum_steps[index[0], index[1]])
# #
# #
# #
# # im = plt.imshow(-grid_cum_steps, interpolation='none')
# # plt.colorbar(im)
# # plt.show()
# # im = plt.imshow(-grid_cum_pains, interpolation='none')
# # plt.colorbar(im)
# # # plt.show()
# #
# # '''
# # Top candidates - longroute_vs_shortroute_gridworld environment:
# #
# #
# # '''

# ############
# #
# # beta_range = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# # kappa_range = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
# #
# #
# # ## Grid world with a dynamically moving obstacle
# # if expt == 'expt4' or expt == 'all':
# #     print("\nGrid world with a dynamically moving obstacle")
# #     mode = 'pavlovian_instrumental_transfer_colearn'
# #     os.makedirs('grid_search_outputs', exist_ok=True)
# #     saverun_folder = os.path.join('grid_search_outputs','moving_obstacle_gridworld')
# #     kappa = 4
# #     beta = 0.1
# #     for beta in beta_range:
# #         for kappa in kappa_range:
# #             run_movingobsenv(saverun_folder, run_num, mode, w=0, modulate_w=True, kappa=kappa, beta=beta, maxeps=500)
# #
# #
# # saverun_folder = os.path.join('grid_search_outputs','moving_obstacle_gridworld')
# # grid_cum_steps = np.zeros([len(beta_range), len(kappa_range)])
# # grid_cum_pains = np.zeros([len(beta_range), len(kappa_range)])
# # i=0
# # for beta in beta_range:
# #     j=0
# #     for kappa in kappa_range:
# #         mw_config = 'modulate_omega_kappa='+str(kappa)+'_beta='+str(beta)+'/'
# #         mw_1_steps = np.load(saverun_folder + '/' + mw_config + 'ep_steps/run_'+str(run_num)+'.npy')
# #         mw_1_pains = np.squeeze(np.load(saverun_folder + '/' + mw_config + 'ep_pains/run_'+str(run_num)+'.npy'))
# #         grid_cum_steps[i][j] = np.sum(mw_1_steps)
# #         grid_cum_pains[i][j] = np.sum(mw_1_pains)
# #         j+=1
# #     i+=1
# #
# # k=10
# # # print(grid_cum_pains)
# # sortedbypain_indices = np.squeeze(np.dstack(np.unravel_index(np.argsort(grid_cum_pains.ravel()), (len(beta_range), len(kappa_range)))))
# # print(sortedbypain_indices[:k])
# # for index in sortedbypain_indices[:k]:
# #     beta = beta_range[index[0]]
# #     kappa = kappa_range[index[1]]
# #     print("beta=", beta, " kappa=", kappa," cum_pain=", grid_cum_pains[index[0], index[1]], " cum_step=", grid_cum_steps[index[0], index[1]])
# #
# #
# #
# # im = plt.imshow(-grid_cum_steps, interpolation='none')
# # plt.colorbar(im)
# # plt.show()
# # im = plt.imshow(-grid_cum_pains, interpolation='none')
# # plt.colorbar(im)
# # plt.show()
# #
# # '''
# # Top candidates - dyn moving obs environment:
# # '''
# #

# #################
# # beta_range = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# # kappa_range = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
# #
# #
# # ## Grid world with a dynamically moving obstacle
# # if expt == 'expt5' or expt == 'all':
# #     print("\nGrid world with a wall maze and moderate pain")
# #     mode = 'pavlovian_instrumental_transfer_colearn'
# #     os.makedirs('grid_search_outputs', exist_ok=True)
# #     saverun_folder = os.path.join('grid_search_outputs','wall_maze_gridworld')
# #     kappa = 3
# #     beta = 0.5
# #     for beta in beta_range:
# #         for kappa in kappa_range:
# #             run_wallmazeenv(saverun_folder, run_num, mode, w=0, modulate_w=True, kappa=kappa, beta=beta, maxeps=500)
# #
# #
# # saverun_folder = os.path.join('grid_search_outputs','wall_maze_gridworld')
# # grid_cum_steps = np.zeros([len(beta_range), len(kappa_range)])
# # grid_cum_pains = np.zeros([len(beta_range), len(kappa_range)])
# # i=0
# # for beta in beta_range:
# #     j=0
# #     for kappa in kappa_range:
# #         mw_config = 'modulate_omega_kappa='+str(kappa)+'_beta='+str(beta)+'/'
# #         mw_1_steps = np.load(saverun_folder + '/' + mw_config + 'ep_steps/run_'+str(run_num)+'.npy')
# #         mw_1_pains = np.squeeze(np.load(saverun_folder + '/' + mw_config + 'ep_pains/run_'+str(run_num)+'.npy'))
# #         grid_cum_steps[i][j] = np.sum(mw_1_steps)
# #         grid_cum_pains[i][j] = np.sum(mw_1_pains)
# #         j+=1
# #     i+=1
# #
# # k=10
# # # print(grid_cum_pains)
# # sortedbypain_indices = np.squeeze(np.dstack(np.unravel_index(np.argsort(grid_cum_pains.ravel()), (len(beta_range), len(kappa_range)))))
# # print(sortedbypain_indices[:k])
# # for index in sortedbypain_indices[:k]:
# #     beta = beta_range[index[0]]
# #     kappa = kappa_range[index[1]]
# #     print("beta=", beta, " kappa=", kappa," cum_pain=", grid_cum_pains[index[0], index[1]], " cum_step=", grid_cum_steps[index[0], index[1]])
# #
# #
# #
# # im = plt.imshow(-grid_cum_steps, interpolation='none')
# # plt.colorbar(im)
# # plt.show()
# # im = plt.imshow(-grid_cum_pains, interpolation='none')
# # plt.colorbar(im)
# # plt.show()
# #
# # '''
# # Top candidates - wallmaze environment:
# # '''

# # shell 1 - meta_param_grid_search longroute_vs_shortroute_gridworld
# # shell 2 - meta_param_grid_search dynobs
# # shell 3 - meta_param_grid_search wallmaze
