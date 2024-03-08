import numpy as np
import matplotlib.pyplot as plt

number_of_runs = 10
zoom200ep = False

SMALL_SIZE = 8
MEDIUM_SIZE = 10
# BIGGER_SIZE = 12
BIGGER_SIZE = 18

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)     # fontsize of the x and y labels
# plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title


env_folder = './multiple_run_outputs/painful_states_gridworld/'
# env_folder = './multiple_run_outputs/painful_states_gridworld_goal_change/'
# env_folder = './multiple_run_outputs/longroute_vs_shortroute_gridworld/'
# env_folder = './multiple_run_outputs/moving_obstacle_gridworld/'
# env_folder = './multiple_run_outputs/wall_maze_gridworld/'
# env_folder = './multiple_run_outputs/modified_mountaincar/'
# env_folder = './multiple_run_outputs/tiny_painful_states_gridworld/'
# env_folder = './multiple_run_outputs/tmaze_two_rewards/'

# mw_config = 'modulate_omega_kappa=2_beta=0.5/'
# mw_config = 'modulate_omega_kappa=6_beta=0.4/'
mw_config = 'modulate_omega_kappa=6.5_beta=0.6/'
# mw_config = 'modulate_omega_kappa=4_beta=0.1/'
# mw_config = 'modulate_omega_kappa=6.5_beta=0.8/'
# mw_config = 'modulate_omega_kappa=3_beta=0.1/'
# mw_config = 'modulate_omega_kappa=4_beta=0.05/'
# mw_config = 'modulate_omega_kappa=3_beta=0.5/'
# mw_config = 'modulate_omega_kappa=6_beta=0.6/'

#### Co-learn pav plotting ####
w000_cum_pains_all = []
w010_cum_pains_all = []
w050_cum_pains_all = []
w090_cum_pains_all = []
mw_1_cum_pains_all = []

w000_cum_steps_all = []
w010_cum_steps_all = []
w050_cum_steps_all = []
w090_cum_steps_all = []
mw_1_cum_steps_all = []

for run_num in range(number_of_runs):
    w000_cum_steps = np.load(env_folder + 'omega=0/cum_steps/run_'+str(run_num)+'.npy')
    w010_cum_steps = np.load(env_folder + 'omega=0.1/cum_steps/run_'+str(run_num)+'.npy')
    w050_cum_steps = np.load(env_folder + 'omega=0.5/cum_steps/run_'+str(run_num)+'.npy')
    w090_cum_steps = np.load(env_folder + 'omega=0.9/cum_steps/run_'+str(run_num)+'.npy')
    mw_1_cum_steps = np.load(env_folder + mw_config + 'cum_steps/run_'+str(run_num)+'.npy')

    w000_pains = np.squeeze(np.load(env_folder + 'omega=0/ep_pains/run_'+str(run_num)+'.npy'))
    w010_pains = np.squeeze(np.load(env_folder + 'omega=0.1/ep_pains/run_'+str(run_num)+'.npy'))
    w050_pains = np.squeeze(np.load(env_folder + 'omega=0.5/ep_pains/run_'+str(run_num)+'.npy'))
    w090_pains = np.squeeze(np.load(env_folder + 'omega=0.9/ep_pains/run_'+str(run_num)+'.npy'))
    mw_1_pains = np.squeeze(np.load(env_folder + mw_config + 'ep_pains/run_'+str(run_num)+'.npy'))

    w000_cum_pains = []
    w010_cum_pains = []
    w050_cum_pains = []
    w090_cum_pains = []
    mw_1_cum_pains = []

    cum_pain = 0
    for i in range(len(w000_pains)):
        cum_pain += w000_pains[i]
        w000_cum_pains.append(cum_pain)
    cum_pain = 0
    for i in range(len(w010_pains)):
        cum_pain += w010_pains[i]
        w010_cum_pains.append(cum_pain)
    cum_pain = 0
    for i in range(len(w050_pains)):
        cum_pain += w050_pains[i]
        w050_cum_pains.append(cum_pain)
    cum_pain = 0
    for i in range(len(w090_pains)):
        cum_pain += w090_pains[i]
        w090_cum_pains.append(cum_pain)
    cum_pain = 0
    for i in range(len(mw_1_pains)):
        cum_pain += mw_1_pains[i]
        mw_1_cum_pains.append(cum_pain)

    w000_cum_pains = np.array(w000_cum_pains)
    w010_cum_pains = np.array(w010_cum_pains)
    w050_cum_pains = np.array(w050_cum_pains)
    w090_cum_pains = np.array(w090_cum_pains)
    mw_1_cum_pains = np.array(mw_1_cum_pains)

    if zoom200ep:
        w000_cum_pains = w000_cum_pains[:200]
        w010_cum_pains = w010_cum_pains[:200]
        w050_cum_pains = w050_cum_pains[:200]
        w090_cum_pains = w090_cum_pains[:200]
        mw_1_cum_pains = mw_1_cum_pains[:200]

        w000_cum_steps = w000_cum_steps[:200]
        w010_cum_steps = w010_cum_steps[:200]
        w050_cum_steps = w050_cum_steps[:200]
        w090_cum_steps = w090_cum_steps[:200]
        mw_1_cum_steps = mw_1_cum_steps[:200]

    w000_cum_pains_all.append(w000_cum_pains)
    w010_cum_pains_all.append(w010_cum_pains)
    w050_cum_pains_all.append(w050_cum_pains)
    w090_cum_pains_all.append(w090_cum_pains)
    mw_1_cum_pains_all.append(mw_1_cum_pains)

    w000_cum_steps_all.append(w000_cum_steps)
    w010_cum_steps_all.append(w010_cum_steps)
    w050_cum_steps_all.append(w050_cum_steps)
    w090_cum_steps_all.append(w090_cum_steps)
    mw_1_cum_steps_all.append(mw_1_cum_steps)
 
##########

w000_cum_pains_all = np.array(w000_cum_pains_all)
w010_cum_pains_all = np.array(w010_cum_pains_all)
w050_cum_pains_all = np.array(w050_cum_pains_all)
w090_cum_pains_all = np.array(w090_cum_pains_all)
mw_1_cum_pains_all = np.array(mw_1_cum_pains_all)

w000_cum_steps_all = np.array(w000_cum_steps_all)
w010_cum_steps_all = np.array(w010_cum_steps_all)
w050_cum_steps_all = np.array(w050_cum_steps_all)
w090_cum_steps_all = np.array(w090_cum_steps_all)
mw_1_cum_steps_all = np.array(mw_1_cum_steps_all)


# plot steps
m0, s0 = np.mean(w000_cum_steps_all, axis=0), np.std(w000_cum_steps_all, axis=0)
m1, s1 = np.mean(w010_cum_steps_all, axis=0), np.std(w010_cum_steps_all, axis=0)
m2, s2 = np.mean(w050_cum_steps_all, axis=0), np.std(w050_cum_steps_all, axis=0)
m3, s3 = np.mean(w090_cum_steps_all, axis=0), np.std(w090_cum_steps_all, axis=0)
m4, s4 = np.mean(mw_1_cum_steps_all, axis=0), np.std(mw_1_cum_steps_all, axis=0)


max_cum_step = np.max(w090_cum_steps_all[:, -1])
norm_cum_step_w000 = w000_cum_steps_all[:, -1] / max_cum_step
norm_cum_step_w010 = w010_cum_steps_all[:, -1] / max_cum_step
norm_cum_step_w050 = w050_cum_steps_all[:, -1] / max_cum_step
norm_cum_step_w090 = w090_cum_steps_all[:, -1] / max_cum_step
norm_cum_step_mw_1 = mw_1_cum_steps_all[:, -1] / max_cum_step

# max_cum_step = np.max(w090_cum_steps_all[:, -1])
# min_cum_step = np.min(w000_cum_steps_all[:, -1])
# norm_cum_step_w000 = (w000_cum_steps_all[:, -1] - min_cum_step)/ (max_cum_step - min_cum_step)
# norm_cum_step_w010 = (w010_cum_steps_all[:, -1] - min_cum_step)/ (max_cum_step - min_cum_step)
# norm_cum_step_w050 = (w050_cum_steps_all[:, -1] - min_cum_step)/ (max_cum_step - min_cum_step)
# norm_cum_step_w090 = (w090_cum_steps_all[:, -1] - min_cum_step)/ (max_cum_step - min_cum_step)
# norm_cum_step_mw_1 = (mw_1_cum_steps_all[:, -1] - min_cum_step)/ (max_cum_step - min_cum_step)


if zoom200ep:
    m0, s0 = m0[:200], s0[:200]
    m1, s1 = m1[:200], s1[:200]
    m2, s2 = m2[:200], s2[:200]
    m3, s3 = m3[:200], s3[:200]
    m4, s4 = m4[:200], s4[:200]

fig = plt.figure()
ax = fig.add_subplot(111)
k=ax.plot(np.arange(len(m0)), m0, 'k', linewidth=2, label='instrumental')
ax.fill_between(np.arange(len(m0)),m0-s0,m0+s0,alpha=.3, color='k')
coral=ax.plot(np.arange(len(m1)), m1, 'coral', linewidth=2, label='$\omega=0.1$')
ax.fill_between(np.arange(len(m1)),m1-s1,m1+s1,alpha=.3, color='coral')
red=ax.plot(np.arange(len(m2)), m2, 'red', linewidth=2, label='$\omega=0.5$')
ax.fill_between(np.arange(len(m2)),m2-s2,m2+s2,alpha=.3, color='red')
darkred=ax.plot(np.arange(len(m3)), m3, 'darkred', linewidth=2, label='$\omega=0.9$')
ax.fill_between(np.arange(len(m3)),m3-s3,m3+s3,alpha=.3, color='darkred')
blue=ax.plot(np.arange(len(m4)), m4, 'blue', linewidth=2, label='flexible $\omega$')
ax.fill_between(np.arange(len(m4)),m4-s4,m4+s4,alpha=.3, color='blue')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# plt.ylabel('Cumulative steps taken over episodes')
plt.xlabel('Episode count')
plt.legend()
# plt.title("Cumulative steps")
plt.show()

# plot pain
m0, s0 = np.mean(w000_cum_pains_all, axis=0), np.std(w000_cum_pains_all, axis=0)
m1, s1 = np.mean(w010_cum_pains_all, axis=0), np.std(w010_cum_pains_all, axis=0)
m2, s2 = np.mean(w050_cum_pains_all, axis=0), np.std(w050_cum_pains_all, axis=0)
m3, s3 = np.mean(w090_cum_pains_all, axis=0), np.std(w090_cum_pains_all, axis=0)
m4, s4 = np.mean(mw_1_cum_pains_all, axis=0), np.std(mw_1_cum_pains_all, axis=0)


max_cum_pain = np.max(w090_cum_pains_all[:, -1])
norm_cum_pain_w000 = w000_cum_pains_all[:, -1] / max_cum_pain
norm_cum_pain_w010 = w010_cum_pains_all[:, -1] / max_cum_pain
norm_cum_pain_w050 = w050_cum_pains_all[:, -1] / max_cum_pain
norm_cum_pain_w090 = w090_cum_pains_all[:, -1] / max_cum_pain
norm_cum_pain_mw_1 = mw_1_cum_pains_all[:, -1] / max_cum_pain


# max_cum_pain = np.max(w090_cum_pains_all[:, -1])
# min_cum_pain = np.min(w050_cum_pains_all[:, -1])
# norm_cum_pain_w000 = (w000_cum_pains_all[:, -1] - min_cum_pain)/ (max_cum_pain - min_cum_pain)
# norm_cum_pain_w010 = (w010_cum_pains_all[:, -1] - min_cum_pain)/ (max_cum_pain - min_cum_pain)
# norm_cum_pain_w050 = (w050_cum_pains_all[:, -1] - min_cum_pain)/ (max_cum_pain - min_cum_pain)
# norm_cum_pain_w090 = (w090_cum_pains_all[:, -1] - min_cum_pain)/ (max_cum_pain - min_cum_pain)
# norm_cum_pain_mw_1 = (mw_1_cum_pains_all[:, -1] - min_cum_pain)/ (max_cum_pain - min_cum_pain)

if zoom200ep:
    m0, s0 = m0[:200], s0[:200]
    m1, s1 = m1[:200], s1[:200]
    m2, s2 = m2[:200], s2[:200]
    m3, s3 = m3[:200], s3[:200]
    m4, s4 = m4[:200], s4[:200]

fig = plt.figure()
ax = fig.add_subplot(111)
k=ax.plot(np.arange(len(m0)), m0, 'k', linewidth=2, label='instrumental')
ax.fill_between(np.arange(len(m0)),m0-s0,m0+s0,alpha=.3, color='k')
coral=ax.plot(np.arange(len(m1)), m1, 'coral', linewidth=2, label='$\omega=0.1$')
ax.fill_between(np.arange(len(m1)),m1-s1,m1+s1,alpha=.3, color='coral')
red=ax.plot(np.arange(len(m2)), m2, 'red', linewidth=2, label='$\omega=0.5$')
ax.fill_between(np.arange(len(m2)),m2-s2,m2+s2,alpha=.3, color='red')
darkred=ax.plot(np.arange(len(m3)), m3, 'darkred', linewidth=2, label='$\omega=0.9$')
ax.fill_between(np.arange(len(m3)),m3-s3,m3+s3,alpha=.3, color='darkred')
blue=ax.plot(np.arange(len(m4)), m4, 'blue', linewidth=2, label='flexible $\omega$')
ax.fill_between(np.arange(len(m4)),m4-s4,m4+s4,alpha=.3, color='blue')
# plt.ylabel('Cumulative pain accrued over episodes')
plt.xlabel('Episode count')
ax.legend()
# plt.title("Cumulative pain")
plt.show()

print(max_cum_pain, max_cum_step)

metric_w000 = 1/(norm_cum_step_w000**2 + norm_cum_pain_w000**2)
metric_w010 = 1/(norm_cum_step_w010**2 + norm_cum_pain_w010**2)
metric_w050 = 1/(norm_cum_step_w050**2 + norm_cum_pain_w050**2)
metric_w090 = 1/(norm_cum_step_w090**2 + norm_cum_pain_w090**2)
metric_mw_1 = 1/(norm_cum_step_mw_1**2 + norm_cum_pain_mw_1**2)


# metric_w000 = 1/(norm_cum_step_w000*norm_cum_pain_w000)
# metric_w010 = 1/(norm_cum_step_w010*norm_cum_pain_w010)
# metric_w050 = 1/(norm_cum_step_w050*norm_cum_pain_w050)
# metric_w090 = 1/(norm_cum_step_w090*norm_cum_pain_w090)
# metric_mw_1 = 1/(norm_cum_step_mw_1*norm_cum_pain_mw_1)


fig = plt.figure()
ax = fig.add_subplot(111)
labels = ['instrumental', 'flexible $\omega$', '$\omega=0.1$','$\omega=0.5$', '$\omega=0.9$']
mean_metric = [np.mean(metric_w000), np.mean(metric_mw_1), np.mean(metric_w010), np.mean(metric_w050), np.mean(metric_w090)]
std_metric = [np.std(metric_w000), np.std(metric_mw_1), np.std(metric_w010), np.std(metric_w050), np.std(metric_w090)]
ax.bar(labels,mean_metric, yerr=std_metric, color=['grey', 'blue', 'coral', 'red', 'darkred'], alpha=0.75)
plt.ylabel('Trade-off metric')
# plt.title('Performance on safety-efficiency trade-off metric')
plt.show()


# ## Only for mountain car

# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # labels = ['instrumental', 'flexible $\omega$', '$\omega=0.1$','$\omega=0.5$', '$\omega=0.9$']
# # mean_metric = [np.mean(w000_cum_steps_all[:, -1]), np.mean(mw_1_cum_steps_all[:, -1]), np.mean(w010_cum_steps_all[:, -1]), np.mean(w050_cum_steps_all[:, -1]), np.mean(w090_cum_steps_all[:, -1])]
# # std_metric = [np.std(w000_cum_steps_all[:, -1]), np.std(mw_1_cum_steps_all[:, -1]), np.std(w010_cum_steps_all[:, -1]), np.std(w050_cum_steps_all[:, -1]), np.std(w090_cum_steps_all[:, -1])]
# # ax.bar(labels,mean_metric, yerr=std_metric, color=['grey', 'blue', 'coral', 'red', 'darkred'])
# # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# # plt.ylabel('Cumulative steps taken over all episodes')
# # plt.show()


# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # labels = ['instrumental', 'flexible $\omega$', '$\omega=0.1$','$\omega=0.5$', '$\omega=0.9$']
# # mean_metric = [np.mean(w000_cum_pains_all[:, -1]), np.mean(mw_1_cum_pains_all[:, -1]), np.mean(w010_cum_pains_all[:, -1]), np.mean(w050_cum_pains_all[:, -1]), np.mean(w090_cum_pains_all[:, -1])]
# # std_metric = [np.std(w000_cum_pains_all[:, -1]), np.std(mw_1_cum_pains_all[:, -1]), np.std(w010_cum_pains_all[:, -1]), np.std(w050_cum_pains_all[:, -1]), np.std(w090_cum_pains_all[:, -1])]
# # ax.bar(labels,mean_metric, yerr=std_metric, color=['grey', 'blue', 'coral', 'red', 'darkred'])
# # plt.ylabel('Cumulative pain accrued over all episodes')
# # plt.show()




# # Plotting choices for T-maze

# w000_choice1_all = []
# w010_choice1_all = []
# w050_choice1_all = []
# w090_choice1_all = []

# w000_choice2_all = []
# w010_choice2_all = []
# w050_choice2_all = []
# w090_choice2_all = []

# for run_num in range(number_of_runs):
#     w000_choice1 = np.load(env_folder + 'omega=0/cum_goal_one_reaches/run_'+str(run_num)+'.npy')
#     w010_choice1 = np.load(env_folder + 'omega=0.1/cum_goal_one_reaches/run_'+str(run_num)+'.npy')
#     w050_choice1 = np.load(env_folder + 'omega=0.5/cum_goal_one_reaches/run_'+str(run_num)+'.npy')
#     w090_choice1 = np.load(env_folder + 'omega=0.9/cum_goal_one_reaches/run_'+str(run_num)+'.npy')
    
#     w000_choice2 = np.load(env_folder + 'omega=0/cum_goal_two_reaches/run_'+str(run_num)+'.npy')
#     w010_choice2 = np.load(env_folder + 'omega=0.1/cum_goal_two_reaches/run_'+str(run_num)+'.npy')
#     w050_choice2 = np.load(env_folder + 'omega=0.5/cum_goal_two_reaches/run_'+str(run_num)+'.npy')
#     w090_choice2 = np.load(env_folder + 'omega=0.9/cum_goal_two_reaches/run_'+str(run_num)+'.npy')

#     w000_choice1_all.append(w000_choice1[-1]/1000)
#     w010_choice1_all.append(w010_choice1[-1]/1000)
#     w050_choice1_all.append(w050_choice1[-1]/1000)
#     w090_choice1_all.append(w090_choice1[-1]/1000)

#     w000_choice2_all.append(w000_choice2[-1]/1000)
#     w010_choice2_all.append(w010_choice2[-1]/1000)
#     w050_choice2_all.append(w050_choice2[-1]/1000)
#     w090_choice2_all.append(w090_choice2[-1]/1000)


# # print(w000_choice1_all, w000_choice2_all)
# # print(w090_choice1_all, w090_choice2_all)

# mean_choice1 = [np.mean(w000_choice1_all), np.mean(w010_choice1_all), np.mean(w050_choice1_all), np.mean(w090_choice1_all)]
# std_choice1 = [np.std(w000_choice1_all), np.std(w010_choice1_all), np.std(w050_choice1_all), np.std(w090_choice1_all)]

# mean_choice2 = [np.mean(w000_choice2_all), np.mean(w010_choice2_all), np.mean(w050_choice2_all), np.mean(w090_choice2_all)]
# std_choice2 = [np.std(w000_choice2_all), np.std(w010_choice2_all), np.std(w050_choice2_all), np.std(w090_choice2_all)]


# fig = plt.figure()
# ax = fig.add_subplot(111)
# labels = ['instrumental', '$\omega=0.1$','$\omega=0.5$', '$\omega=0.9$']
# x = np.arange(4)  # the label locations
# width = 0.35  # the width of the bars
# rects1 = ax.bar(x - width/2, mean_choice1, width, label='Left reward (R=+0.1)', yerr=std_choice1, color=['green'], alpha=0.75)
# rects2 = ax.bar(x + width/2, mean_choice2, width, label='Right reward (R=+1)', yerr=std_choice2, color=['orange'], alpha=0.75)
# plt.xticks(x, labels)
# ax.legend()
# plt.ylabel('Normalised reward aquisitions')
# # plt.title('Rewarding goal chosen by the agent')
# fig.tight_layout()
# plt.show()