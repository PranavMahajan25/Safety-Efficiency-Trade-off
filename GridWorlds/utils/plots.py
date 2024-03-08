import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.helper_functions import create_policy_direction_arrays, row_col_to_seq
import numpy as np

def plot_quiverpolicy(model,stochastic_policy=None, title=None, path=None, two_goals=False, secondary_goal=None, plot_plots=True):
    """
    Plots the quiver plot of the stochastic policy

    Parameters
    ----------
    model : python object
        Holds information about the environment to solve
        such as the reward structure and the transition dynamics.

    stochastic policy : numpy array of shape (N, num_actions)
        Optimal policy of the environment.

    title : string
        Title of the plot. Defaults to None.

    path : string
        Path to save image. Defaults to None.
    """

    fig, ax = plt.subplots()
    add_value_function(model, None, "")
    add_patches(model, ax, two_goals=two_goals)
    add_secondary_goal(model, ax, secondary_goal)
    add_stochastic_policy(model, stochastic_policy,two_goals=two_goals)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
               fancybox=True, shadow=True, ncol=3)
    if title is not None:
        plt.title(title, fontdict=None, loc='center')
    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    if plot_plots:
        plt.show()
    return

def plot_gridworld(model, value_function=None, policy=None, state_counts=None, title=None, path=None, secondary_goal=None, two_goals=False, visualise = False, state = None, plot_plots=True):
    matplotlib.rcParams['figure.dpi'] = 300
    """
    Plots the grid world solution.

    Parameters
    ----------
    model : python object
        Holds information about the environment to solve
        such as the reward structure and the transition dynamics.

    value_function : numpy array of shape (N, 1)
        Value function of the environment where N is the number
        of states in the environment.

    policy : numpy array of shape (N, 1)
        Optimal policy of the environment.

    title : string
        Title of the plot. Defaults to None.

    path : string
        Path to save image. Defaults to None.

    visualise : bool
        Whether visualising trajectory or not
    
    state : numpay array
        Current state of the agent
    """

    if value_function is not None and state_counts is not None:
        raise Exception("Must supple either value function or state_counts, not both!")

    fig, ax = plt.subplots()

    # add features to grid world
    if value_function is not None:
        add_value_function(model, value_function, "Value function")
    elif state_counts is not None:
        add_value_function(model, state_counts, "State counts")
    elif value_function is None and state_counts is None:
        add_value_function(model, value_function, "Value function")
    add_patches(model, ax, visualise, state, two_goals=two_goals)
    add_policy(model, policy, two_goals=two_goals)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
               fancybox=True, shadow=True, ncol=3)
    if title is not None:
        plt.title(title, fontdict=None, loc='center')
    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight')

    if visualise:
        fig.canvas.draw()
        # Now we can save it to a numpy array.
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close("all")
        return data
    else:
        if plot_plots:
            plt.show()
        return 0

def add_value_function(model, value_function, name):

    if value_function is not None:
        # colobar max and min
        vmin = np.min(value_function)
        vmax = np.max(value_function)
        # reshape and set obstructed states to low value
        val = value_function[:-1, 0].reshape(model.num_rows, model.num_cols)
        if model.obs_states is not None:
            index = model.obs_states
            val[index[:, 0], index[:, 1]] = -100
        plt.imshow(val, vmin=vmin, vmax=vmax, zorder=5)
        plt.colorbar(label=name)
    else:
        val = np.zeros((model.num_rows, model.num_cols))
        plt.imshow(val, zorder=0)
        plt.yticks(np.arange(-0.5, model.num_rows+0.5, step=1))
        plt.xticks(np.arange(-0.5, model.num_cols+0.5, step=1))
        plt.grid()
        plt.colorbar(label=name)

# not using
def add_secondary_goal(model, ax, secondary_goal):
    if secondary_goal is not None:
        light = patches.RegularPolygon(tuple(np.flip(secondary_goal[0])), numVertices=5,
                                        radius=0.25, orientation=np.pi, edgecolor='orange', zorder=1,
                                        facecolor='orange', label="Secondary Goal")
        ax.add_patch(light)

def add_patches(model, ax, visualise = False, state = None, two_goals=False):

    if visualise:
        state_pos = patches.Circle(tuple(np.flip(state[0])), 0.2, linewidth=1,
                            edgecolor='b', facecolor='b', zorder=2, label="Start")
        ax.add_patch(state_pos)
    else:
        start = patches.Circle(tuple(np.flip(model.start_state[0])), 0.2, linewidth=1,
                            edgecolor='b', facecolor='b', zorder=1, label="Start")
        ax.add_patch(start)

    if two_goals:
        for i in range(model.small_goal_states.shape[0]):
            end = patches.RegularPolygon(tuple(np.flip(model.small_goal_states[i, :])), numVertices=5,
                                        radius=0.25, orientation=np.pi, edgecolor='g', zorder=1,
                                        facecolor='g', label="Goal" if i == 0 else None)
            ax.add_patch(end)

        for i in range(model.big_goal_states.shape[0]):
            end = patches.RegularPolygon(tuple(np.flip(model.big_goal_states[i, :])), numVertices=5,
                                        radius=0.25, orientation=np.pi, edgecolor='orange', zorder=1,
                                        facecolor='g', label="Goal" if i == 0 else None)
            ax.add_patch(end)
    else:
        for i in range(model.goal_states.shape[0]):
            end = patches.RegularPolygon(tuple(np.flip(model.goal_states[i, :])), numVertices=5,
                                        radius=0.25, orientation=np.pi, edgecolor='g', zorder=1,
                                        facecolor='g', label="Goal" if i == 0 else None)
            ax.add_patch(end)
    

    # obstructed states patches
    if model.obs_states is not None:
        for i in range(model.obs_states.shape[0]):
            obstructed = patches.Rectangle(tuple(np.flip(model.obs_states[i, :]) - 0.45), 0.9, 0.9,
                                           linewidth=1, edgecolor='dimgrey', facecolor='dimgrey', zorder=1,
                                           label="Obstructed" if i == 0 else None)
            ax.add_patch(obstructed)

    if model.bad_states is not None:
        for i in range(model.bad_states.shape[0]):
            bad = patches.Rectangle(tuple(np.flip(model.bad_states[i, :])-0.45), 0.9, 0.9,
                                linewidth=1, edgecolor='tomato', facecolor='tomato', zorder=1,
                                label="Bad state" if i == 0 else None)
            ax.add_patch(bad)

    if model.restart_states is not None:
        for i in range(model.restart_states.shape[0]):
            restart = patches.Rectangle(tuple(np.flip(model.restart_states[i, :]) -0.45), 0.9, 0.9,
                                    linewidth=1, edgecolor='darkorange', facecolor='darkorange', zorder=1,
                                    label="Restart state" if i == 0 else None)
            ax.add_patch(restart)
    
    if model.surprise_states is not None:
        for i in range(model.surprise_states.shape[0]):
            # surprise = patches.Wedge(tuple(np.flip(model.surprise_states[i, :])), 0.2, 40, -40,
            #                         linewidth=1, edgecolor='purple', facecolor='purple', zorder=1,
            #                         label="Surprise state" if i == 0 else None)
            surprise = patches.Rectangle(tuple(np.flip(model.surprise_states[i, :]) -0.45), 0.9, 0.9,
                                    linewidth=1, edgecolor='mediumorchid', facecolor='mediumorchid', zorder=1,
                                    label="Restart state" if i == 0 else None)
            ax.add_patch(surprise)

def add_policy(model, policy, two_goals=False):

    if policy is not None:
        # define the gridworld
        X = np.arange(0, model.num_cols, 1)
        Y = np.arange(0, model.num_rows, 1)

        # define the policy direction arrows
        U, V = create_policy_direction_arrays(model, policy)
        # remove the obstructions and final state arrows
        if two_goals:
            ra = model.small_goal_states
            U[ra[:, 0], ra[:, 1]] = np.nan
            V[ra[:, 0], ra[:, 1]] = np.nan

            ra = model.big_goal_states
            U[ra[:, 0], ra[:, 1]] = np.nan
            V[ra[:, 0], ra[:, 1]] = np.nan
        else:
            ra = model.goal_states
            U[ra[:, 0], ra[:, 1]] = np.nan
            V[ra[:, 0], ra[:, 1]] = np.nan
        if model.obs_states is not None:
            ra = model.obs_states
            U[ra[:, 0], ra[:, 1]] = np.nan
            V[ra[:, 0], ra[:, 1]] = np.nan
        if model.restart_states is not None:
            ra = model.restart_states
            U[ra[:, 0], ra[:, 1]] = np.nan
            V[ra[:, 0], ra[:, 1]] = np.nan

        plt.quiver(X, Y, U, V, zorder=10, label="Policy")

def add_stochastic_policy(model, spi, two_goals=False):

    if spi is not None:
        # define the gridworld
        X = np.arange(0, model.num_cols, 1)
        Y = np.arange(0, model.num_rows, 1)

        # define the policy direction arrows
        # intitialize direction arrays
        U = np.zeros((model.num_rows, model.num_cols))
        V = np.zeros((model.num_rows, model.num_cols))

        for row in range(model.num_rows):
            for col in range(model.num_cols):
                state = np.array([[row,col]])
                state_seq = row_col_to_seq(state, model.num_cols)
                V[row, col] = np.round(0.5*(spi[state_seq, 0] - spi[state_seq, 1]),4)
                U[row, col] = np.round(0.5*(spi[state_seq, 3] - spi[state_seq, 2]),4)

        # remove the obstructions and final state arrows
        if two_goals:
            ra = model.small_goal_states
            U[ra[:, 0], ra[:, 1]] = np.nan
            V[ra[:, 0], ra[:, 1]] = np.nan

            ra = model.big_goal_states
            U[ra[:, 0], ra[:, 1]] = np.nan
            V[ra[:, 0], ra[:, 1]] = np.nan
        else:
            ra = model.goal_states
            U[ra[:, 0], ra[:, 1]] = np.nan
            V[ra[:, 0], ra[:, 1]] = np.nan

        if model.obs_states is not None:    
            ra = model.obs_states
            U[ra[:, 0], ra[:, 1]] = np.nan
            V[ra[:, 0], ra[:, 1]] = np.nan
        if model.restart_states is not None:
            ra = model.restart_states
            U[ra[:, 0], ra[:, 1]] = np.nan
            V[ra[:, 0], ra[:, 1]] = np.nan

        plt.quiver(X, Y, U, V, zorder=10, label="Policy")