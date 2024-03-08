import os
import sys
from run_hottilesenv_PIT import run_hottilesenv
from run_longroute_PIT import run_longrouteenv
from run_movingobstacle_PIT import run_movingobsenv
from run_wallmaze_PIT import run_wallmazeenv
from run_fearavoidance1_PIT import run_FAenv
# from run_mountain_car import run_mountaincarenv
from run_two_rewards_PIT import run_Tmaze
number_of_runs = 10
expt = 'all'


## Moderately Painful States Environments (Hot tiles env)
if expt == 'expt1' or expt == 'all':
    print("\nModerately Painful States Environments (Hot tiles env)")
    mode = 'pavlovian_instrumental_transfer_presave'
    os.makedirs('multiple_run_outputs', exist_ok=True)
    saverun_folder = os.path.join('multiple_run_outputs','painful_states_gridworld')
    kappa = 6
    beta = 0.4
    for run_num in range(number_of_runs):
        run_hottilesenv(saverun_folder, run_num, mode, w=0, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_hottilesenv(saverun_folder, run_num, mode, w=0.1, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_hottilesenv(saverun_folder, run_num, mode, w=0.5, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_hottilesenv(saverun_folder, run_num, mode, w=0.9, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_hottilesenv(saverun_folder, run_num, mode, w=0, modulate_w=True, kappa=kappa, beta=beta)


## Moderately Painful States Environments (Hot tiles env) with goal change
if expt == 'expt2' or expt == 'all':
    print("\nModerately Painful States Environments (Hot tiles env) with goal change")
    mode = 'pavlovian_instrumental_transfer_colearn_goal_change'
    os.makedirs('multiple_run_outputs', exist_ok=True)
    saverun_folder = os.path.join('multiple_run_outputs','painful_states_gridworld_goal_change')
    kappa = 2
    beta = 0.5
    for run_num in range(number_of_runs):
        run_hottilesenv(saverun_folder, run_num, mode, w=0, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_hottilesenv(saverun_folder, run_num, mode, w=0.1, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_hottilesenv(saverun_folder, run_num, mode, w=0.5, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_hottilesenv(saverun_folder, run_num, mode, w=0.9, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_hottilesenv(saverun_folder, run_num, mode, w=0, modulate_w=True, kappa=kappa, beta=beta)

## Grid world with short route vs long route
if expt == 'expt3' or expt == 'all':
    print("\nGrid world with short route vs long route")
    mode = 'pavlovian_instrumental_transfer_colearn'
    os.makedirs('multiple_run_outputs', exist_ok=True)
    saverun_folder = os.path.join('multiple_run_outputs','longroute_vs_shortroute_gridworld')
    kappa = 4
    beta = 0.1
    for run_num in range(number_of_runs):
        run_longrouteenv(saverun_folder, run_num, mode, w=0, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_longrouteenv(saverun_folder, run_num, mode, w=0.1, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_longrouteenv(saverun_folder, run_num, mode, w=0.5, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_longrouteenv(saverun_folder, run_num, mode, w=0.9, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_longrouteenv(saverun_folder, run_num, mode, w=0, modulate_w=True, kappa=kappa, beta=beta)

## Grid world with a dynamically moving obstacle
if expt == 'expt4' or expt == 'all':
    print("\nGrid world with a dynamically moving obstacle")
    mode = 'pavlovian_instrumental_transfer_colearn'
    os.makedirs('multiple_run_outputs', exist_ok=True)
    saverun_folder = os.path.join('multiple_run_outputs','moving_obstacle_gridworld')
    kappa = 3
    beta = 0.1
    for run_num in range(number_of_runs):
        run_movingobsenv(saverun_folder, run_num, mode, w=0, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_movingobsenv(saverun_folder, run_num, mode, w=0.1, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_movingobsenv(saverun_folder, run_num, mode, w=0.5, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_movingobsenv(saverun_folder, run_num, mode, w=0.9, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_movingobsenv(saverun_folder, run_num, mode, w=0, modulate_w=True, kappa=kappa, beta=beta)


## Grid world with a wall maze and moderate pain
if expt == 'expt5' or expt == 'all':
    print("\nGrid world with a wall maze and moderate pain")
    mode = 'pavlovian_instrumental_transfer_colearn'
    os.makedirs('multiple_run_outputs', exist_ok=True)
    saverun_folder = os.path.join('multiple_run_outputs','wall_maze_gridworld')
    kappa = 3
    beta = 0.5
    for run_num in range(number_of_runs):
        run_wallmazeenv(saverun_folder, run_num, mode, w=0, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_wallmazeenv(saverun_folder, run_num, mode, w=0.1, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_wallmazeenv(saverun_folder, run_num, mode, w=0.5, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_wallmazeenv(saverun_folder, run_num, mode, w=0.9, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_wallmazeenv(saverun_folder, run_num, mode, w=0, modulate_w=True, kappa=kappa, beta=beta)

## Fear Avoidance environment
if expt == 'exptFA' or expt == 'all':
    print("\nGrid world with fear avoidance clinical prediction")
    mode = 'pavlovian_instrumental_transfer_colearn_truncated'
    os.makedirs('multiple_run_outputs', exist_ok=True)
    saverun_folder = os.path.join('multiple_run_outputs','fear_avoidance_gridworld')
    kappa = 3
    beta = 0.01
    for run_num in range(number_of_runs):
        run_FAenv(saverun_folder, run_num, mode, w=0, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_FAenv(saverun_folder, run_num, mode, w=0.1, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_FAenv(saverun_folder, run_num, mode, w=0.5, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_FAenv(saverun_folder, run_num, mode, w=0.9, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_FAenv(saverun_folder, run_num, mode, w=0, modulate_w=True, kappa=kappa, beta=beta)


## Moderately Painful States Environments (Hot tiles env) with goal change
if expt == 'expt2' or expt == 'all':
    print("\nModerately Painful States Environments (Hot tiles env) with goal change")
    mode = 'pavlovian_instrumental_transfer_colearn_goal_change'
    os.makedirs('multiple_run_outputs', exist_ok=True)
    saverun_folder = os.path.join('multiple_run_outputs','painful_states_gridworld_goal_change')
    kappa = 6
    beta = 0.4
    # for run_num in range(number_of_runs):
    #     run_hottilesenv(saverun_folder, run_num, mode, w=0, modulate_w=False, kappa=kappa, beta=beta)
    # for run_num in range(number_of_runs):
    #     run_hottilesenv(saverun_folder, run_num, mode, w=0.1, modulate_w=False, kappa=kappa, beta=beta)
    # for run_num in range(number_of_runs):
    #     run_hottilesenv(saverun_folder, run_num, mode, w=0.5, modulate_w=False, kappa=kappa, beta=beta)
    # for run_num in range(number_of_runs):
    #     run_hottilesenv(saverun_folder, run_num, mode, w=0.9, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_hottilesenv(saverun_folder, run_num, mode, w=0, modulate_w=True, kappa=kappa, beta=beta)

## T-maze with two-rewards
if expt == 'expt2.5' or expt == 'all':
    print("\nT-maze with two-rewards")
    mode = 'pavlovian_instrumental_transfer_colearn_goal_change'
    os.makedirs('multiple_run_outputs', exist_ok=True)
    saverun_folder = os.path.join('multiple_run_outputs','tmaze_two_rewards')
    kappa = 6 # doesn't matter
    beta = 0.4 # doesn't matter
    for run_num in range(number_of_runs):
        run_Tmaze(saverun_folder, run_num, mode, w=0, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_Tmaze(saverun_folder, run_num, mode, w=0.1, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_Tmaze(saverun_folder, run_num, mode, w=0.5, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_Tmaze(saverun_folder, run_num, mode, w=0.9, modulate_w=False, kappa=kappa, beta=beta)


# Tiny grid world with moderately painful states 
if expt == 'expt12' or expt == 'all':
    print("\nTiny Grid world with a moderately painful states")
    mode = 'pavlovian_instrumental_transfer_colearn'
    os.makedirs('multiple_run_outputs', exist_ok=True)
    saverun_folder = os.path.join('multiple_run_outputs','tiny_painful_states_gridworld')
    kappa = 2
    beta = 0.5
    for run_num in range(number_of_runs):
        run_tiny_hottilesenv(saverun_folder, run_num, mode, w=0, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_tiny_hottilesenv(saverun_folder, run_num, mode, w=0.1, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_tiny_hottilesenv(saverun_folder, run_num, mode, w=0.5, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_tiny_hottilesenv(saverun_folder, run_num, mode, w=0.9, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_tiny_hottilesenv(saverun_folder, run_num, mode, w=0, modulate_w=True, kappa=kappa, beta=beta)

# Grid world with short route vs long route
if expt == 'expt3' or expt == 'all':
    print("\nGrid world with short route vs long route")
    mode = 'pavlovian_instrumental_transfer_colearn'
    os.makedirs('multiple_run_outputs', exist_ok=True)
    saverun_folder = os.path.join('multiple_run_outputs','longroute_vs_shortroute_gridworld')
    kappa = 6
    beta = 0.1
    # for run_num in range(number_of_runs):
    #     run_longrouteenv(saverun_folder, run_num, mode, w=0, modulate_w=False, kappa=kappa, beta=beta)
    # for run_num in range(number_of_runs):
    #     run_longrouteenv(saverun_folder, run_num, mode, w=0.1, modulate_w=False, kappa=kappa, beta=beta)
    # for run_num in range(number_of_runs):
    #     run_longrouteenv(saverun_folder, run_num, mode, w=0.5, modulate_w=False, kappa=kappa, beta=beta)
    # for run_num in range(number_of_runs):
    #     run_longrouteenv(saverun_folder, run_num, mode, w=0.9, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_longrouteenv(saverun_folder, run_num, mode, w=0, modulate_w=True, kappa=kappa, beta=beta)

## Grid world with a dynamically moving obstacle
if expt == 'expt4' or expt == 'all':
    print("\nGrid world with a dynamically moving obstacle")
    mode = 'pavlovian_instrumental_transfer_colearn'
    os.makedirs('multiple_run_outputs', exist_ok=True)
    saverun_folder = os.path.join('multiple_run_outputs','moving_obstacle_gridworld')
    kappa = 3
    beta = 0.1
    for run_num in range(number_of_runs):
        run_movingobsenv(saverun_folder, run_num, mode, w=0, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_movingobsenv(saverun_folder, run_num, mode, w=0.1, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_movingobsenv(saverun_folder, run_num, mode, w=0.5, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_movingobsenv(saverun_folder, run_num, mode, w=0.9, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_movingobsenv(saverun_folder, run_num, mode, w=0, modulate_w=True, kappa=kappa, beta=beta)


## Grid world with a wall maze and moderate pain 
if expt == 'expt5' or expt == 'all':
    print("\nGrid world with a wall maze and moderate pain")
    mode = 'pavlovian_instrumental_transfer_colearn'
    os.makedirs('multiple_run_outputs', exist_ok=True)
    saverun_folder = os.path.join('multiple_run_outputs','wall_maze_gridworld')
    kappa = 3
    beta = 0.5
    for run_num in range(number_of_runs):
        run_wallmazeenv(saverun_folder, run_num, mode, w=0, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_wallmazeenv(saverun_folder, run_num, mode, w=0.1, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_wallmazeenv(saverun_folder, run_num, mode, w=0.5, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_wallmazeenv(saverun_folder, run_num, mode, w=0.9, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_wallmazeenv(saverun_folder, run_num, mode, w=0, modulate_w=True, kappa=kappa, beta=beta)

## Fear Avoidance environment
if expt == 'exptFA' or expt == 'all':
    print("\nGrid world with fear avoidance clinical prediction")
    mode = 'pavlovian_instrumental_transfer_colearn_truncated'
    os.makedirs('multiple_run_outputs', exist_ok=True)
    saverun_folder = os.path.join('multiple_run_outputs','fear_avoidance_gridworld')
    kappa = 6
    beta = 0.4
    # for run_num in range(number_of_runs):
    #     run_FAenv(saverun_folder, run_num, mode, w=0, modulate_w=False, kappa=kappa, beta=beta)
    # for run_num in range(number_of_runs):
    #     run_FAenv(saverun_folder, run_num, mode, w=0.1, modulate_w=False, kappa=kappa, beta=beta)
    # for run_num in range(number_of_runs):
    #     run_FAenv(saverun_folder, run_num, mode, w=0.5, modulate_w=False, kappa=kappa, beta=beta)
    # for run_num in range(number_of_runs):
    #     run_FAenv(saverun_folder, run_num, mode, w=0.9, modulate_w=False, kappa=kappa, beta=beta)
    for run_num in range(number_of_runs):
        run_FAenv(saverun_folder, run_num, mode, w=0, modulate_w=True, kappa=kappa, beta=beta)


# Modified mountaincar environment
if expt == 'expt6' or expt == 'all':
    print("\nModified mountaincar environment")
    mode = 'pavlovian_instrumental_transfer_colearn'
    os.makedirs('multiple_run_outputs', exist_ok=True)
    saverun_folder = os.path.join('multiple_run_outputs','modified_mountaincar')
    kappa = 3
    beta = 0.5
    # for run_num in range(number_of_runs):
    #     run_mountaincarenv(saverun_folder, run_num, mode, w=0, modulate_w=False, kappa=kappa, beta=beta)
    # for run_num in range(number_of_runs):
    #     run_mountaincarenv(saverun_folder, run_num, mode, w=0.1, modulate_w=False, kappa=kappa, beta=beta)
    # for run_num in range(number_of_runs):
    #     run_mountaincarenv(saverun_folder, run_num, mode, w=0.5, modulate_w=False, kappa=kappa, beta=beta)
    # for run_num in range(number_of_runs):
    #     run_mountaincarenv(saverun_folder, run_num, mode, w=0.9, modulate_w=False, kappa=kappa, beta=beta)
    # for run_num in range(number_of_runs):
    #     run_mountaincarenv(saverun_folder, run_num, mode, w=0, modulate_w=True, kappa=kappa, beta=beta)
    
