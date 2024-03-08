import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy as sp

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


env_folder = './multiple_run_outputs/longroute_vs_shortroute_gridworld/'

#### Co-learn pav plotting ####
w000_cum_pains_all = []
w010_cum_pains_all = []
w020_cum_pains_all = []
w030_cum_pains_all = []
w040_cum_pains_all = []
w050_cum_pains_all = []
w060_cum_pains_all = []
w070_cum_pains_all = []
w080_cum_pains_all = []
w090_cum_pains_all = []


w000_cum_steps_all = []
w010_cum_steps_all = []
w020_cum_steps_all = []
w030_cum_steps_all = []
w040_cum_steps_all = []
w050_cum_steps_all = []
w060_cum_steps_all = []
w070_cum_steps_all = []
w080_cum_steps_all = []
w090_cum_steps_all = []

for run_num in range(number_of_runs):
    w000_cum_pains = np.load(env_folder + 'omega=0/cum_pains/run_'+str(run_num)+'.npy')
    w010_cum_pains = np.load(env_folder + 'omega=0.1/cum_pains/run_'+str(run_num)+'.npy')
    w020_cum_pains = np.load(env_folder + 'omega=0.2/cum_pains/run_'+str(run_num)+'.npy')
    w030_cum_pains = np.load(env_folder + 'omega=0.3/cum_pains/run_'+str(run_num)+'.npy')
    w040_cum_pains = np.load(env_folder + 'omega=0.4/cum_pains/run_'+str(run_num)+'.npy')
    w050_cum_pains = np.load(env_folder + 'omega=0.5/cum_pains/run_'+str(run_num)+'.npy')
    w060_cum_pains = np.load(env_folder + 'omega=0.6/cum_pains/run_'+str(run_num)+'.npy')
    w070_cum_pains = np.load(env_folder + 'omega=0.7/cum_pains/run_'+str(run_num)+'.npy')
    w080_cum_pains = np.load(env_folder + 'omega=0.8/cum_pains/run_'+str(run_num)+'.npy')
    w090_cum_pains = np.load(env_folder + 'omega=0.9/cum_pains/run_'+str(run_num)+'.npy')

    w000_cum_steps = np.load(env_folder + 'omega=0/cum_steps/run_'+str(run_num)+'.npy')
    w010_cum_steps = np.load(env_folder + 'omega=0.1/cum_steps/run_'+str(run_num)+'.npy')
    w020_cum_steps = np.load(env_folder + 'omega=0.2/cum_steps/run_'+str(run_num)+'.npy')
    w030_cum_steps = np.load(env_folder + 'omega=0.3/cum_steps/run_'+str(run_num)+'.npy')
    w040_cum_steps = np.load(env_folder + 'omega=0.4/cum_steps/run_'+str(run_num)+'.npy')
    w050_cum_steps = np.load(env_folder + 'omega=0.5/cum_steps/run_'+str(run_num)+'.npy')
    w060_cum_steps = np.load(env_folder + 'omega=0.6/cum_steps/run_'+str(run_num)+'.npy')
    w070_cum_steps = np.load(env_folder + 'omega=0.7/cum_steps/run_'+str(run_num)+'.npy')
    w080_cum_steps = np.load(env_folder + 'omega=0.8/cum_steps/run_'+str(run_num)+'.npy')
    w090_cum_steps = np.load(env_folder + 'omega=0.9/cum_steps/run_'+str(run_num)+'.npy')


    w000_cum_pains_all.append(w000_cum_pains[-1][0])
    w010_cum_pains_all.append(w010_cum_pains[-1][0])
    w020_cum_pains_all.append(w020_cum_pains[-1][0])
    w030_cum_pains_all.append(w030_cum_pains[-1][0])
    try:
        w040_cum_pains_all.append(w040_cum_pains[-1][0])
    except:
        w040_cum_pains_all.append(w040_cum_pains[-1])
    w050_cum_pains_all.append(w050_cum_pains[-1][0])
    w060_cum_pains_all.append(w060_cum_pains[-1][0])
    w070_cum_pains_all.append(w070_cum_pains[-1][0])
    w080_cum_pains_all.append(w080_cum_pains[-1][0])
    w090_cum_pains_all.append(w090_cum_pains[-1][0])

    w000_cum_steps_all.append(w000_cum_steps[-1])
    w010_cum_steps_all.append(w010_cum_steps[-1])
    w020_cum_steps_all.append(w020_cum_steps[-1])
    w030_cum_steps_all.append(w030_cum_steps[-1])
    w040_cum_steps_all.append(w040_cum_steps[-1])
    w050_cum_steps_all.append(w050_cum_steps[-1])
    w060_cum_steps_all.append(w060_cum_steps[-1])
    w070_cum_steps_all.append(w070_cum_steps[-1])
    w080_cum_steps_all.append(w080_cum_steps[-1])
    w090_cum_steps_all.append(w090_cum_steps[-1])
 
##########

w000_cum_pains_all = np.array(w000_cum_pains_all)
w010_cum_pains_all = np.array(w010_cum_pains_all)
w020_cum_pains_all = np.array(w020_cum_pains_all)
w030_cum_pains_all = np.array(w030_cum_pains_all)
w040_cum_pains_all = np.array(w040_cum_pains_all)
w050_cum_pains_all = np.array(w050_cum_pains_all)
w060_cum_pains_all = np.array(w060_cum_pains_all)
w070_cum_pains_all = np.array(w070_cum_pains_all)
w080_cum_pains_all = np.array(w080_cum_pains_all)
w090_cum_pains_all = np.array(w090_cum_pains_all)

w000_cum_steps_all = np.array(w000_cum_steps_all)
w010_cum_steps_all = np.array(w010_cum_steps_all)
w020_cum_steps_all = np.array(w020_cum_steps_all)
w030_cum_steps_all = np.array(w030_cum_steps_all)
w040_cum_steps_all = np.array(w040_cum_steps_all)
w050_cum_steps_all = np.array(w050_cum_steps_all)
w060_cum_steps_all = np.array(w060_cum_steps_all)
w070_cum_steps_all = np.array(w070_cum_steps_all)
w080_cum_steps_all = np.array(w080_cum_steps_all)
w090_cum_steps_all = np.array(w090_cum_steps_all)



mean_pains = np.array([np.mean(w000_cum_pains_all), np.mean(w010_cum_pains_all), np.mean(w020_cum_pains_all),\
     np.mean(w030_cum_pains_all), np.mean(w040_cum_pains_all), np.mean(w050_cum_pains_all),np.mean(w060_cum_pains_all),\
        np.mean(w070_cum_pains_all), np.mean(w080_cum_pains_all), np.mean(w090_cum_pains_all)])


std_pains = np.array([np.std(w000_cum_pains_all), np.std(w010_cum_pains_all), np.std(w020_cum_pains_all),\
     np.std(w030_cum_pains_all), np.std(w040_cum_pains_all), np.std(w050_cum_pains_all),np.std(w060_cum_pains_all),\
        np.std(w070_cum_pains_all), np.std(w080_cum_pains_all), np.std(w090_cum_pains_all)])


mean_steps = np.array([np.mean(w000_cum_steps_all), np.mean(w010_cum_steps_all), np.mean(w020_cum_steps_all),\
     np.mean(w030_cum_steps_all), np.mean(w040_cum_steps_all), np.mean(w050_cum_steps_all),np.mean(w060_cum_steps_all),\
        np.mean(w070_cum_steps_all), np.mean(w080_cum_steps_all), np.mean(w090_cum_steps_all)])


std_steps = np.array([np.std(w000_cum_steps_all), np.std(w010_cum_steps_all), np.std(w020_cum_steps_all),\
     np.std(w030_cum_steps_all), np.std(w040_cum_steps_all), np.std(w050_cum_steps_all),np.std(w060_cum_steps_all),\
        np.std(w070_cum_steps_all), np.std(w080_cum_steps_all), np.std(w090_cum_steps_all)])


colors = cm.rainbow(np.linspace(0, 1, 10))

plt.scatter(mean_pains, mean_steps, color=colors)
plt.errorbar(mean_pains, mean_steps, xerr=std_pains, yerr=std_steps, ecolor='grey', linestyle='' )
plt.xlabel('Cumulative pain (CP)')
plt.ylabel('Cumulative steps (CS)')
# plt.legend()
plt.show()

print(sp.stats.pearsonr(mean_steps, mean_pains))



CSs = [421, 324, 251, 377, 317, 444, 440, 453, 432, 487, 358, 470, 265, 356, 386, 438]
CPs = [24, 99, 174, 52, 114, 50, 16, 17, 23, 25, 69, 18, 169, 147, 105, 23]

print(len(CPs))

plt.scatter(CPs, CSs, cmap='hot')
plt.xlabel('Cumulative pain (CP)')
plt.ylabel('Cumulative steps (CS)')
plt.show()

print(sp.stats.pearsonr(CSs, CPs))


# # plot steps
# m0, s0 = np.mean(w000_cum_steps_all, axis=0), np.std(w000_cum_steps_all, axis=0)
# m1, s1 = np.mean(w010_cum_steps_all, axis=0), np.std(w010_cum_steps_all, axis=0)
# m2, s2 = np.mean(w050_cum_steps_all, axis=0), np.std(w050_cum_steps_all, axis=0)
# m3, s3 = np.mean(w090_cum_steps_all, axis=0), np.std(w090_cum_steps_all, axis=0)
# m4, s4 = np.mean(mw_1_cum_steps_all, axis=0), np.std(mw_1_cum_steps_all, axis=0)


# max_cum_step = np.max(w090_cum_steps_all[:, -1])
# norm_cum_step_w000 = w000_cum_steps_all[:, -1] / max_cum_step
# norm_cum_step_w010 = w010_cum_steps_all[:, -1] / max_cum_step
# norm_cum_step_w050 = w050_cum_steps_all[:, -1] / max_cum_step
# norm_cum_step_w090 = w090_cum_steps_all[:, -1] / max_cum_step
# norm_cum_step_mw_1 = mw_1_cum_steps_all[:, -1] / max_cum_step

# # max_cum_step = np.max(w090_cum_steps_all[:, -1])
# # min_cum_step = np.min(w000_cum_steps_all[:, -1])
# # norm_cum_step_w000 = (w000_cum_steps_all[:, -1] - min_cum_step)/ (max_cum_step - min_cum_step)
# # norm_cum_step_w010 = (w010_cum_steps_all[:, -1] - min_cum_step)/ (max_cum_step - min_cum_step)
# # norm_cum_step_w050 = (w050_cum_steps_all[:, -1] - min_cum_step)/ (max_cum_step - min_cum_step)
# # norm_cum_step_w090 = (w090_cum_steps_all[:, -1] - min_cum_step)/ (max_cum_step - min_cum_step)
# # norm_cum_step_mw_1 = (mw_1_cum_steps_all[:, -1] - min_cum_step)/ (max_cum_step - min_cum_step)


# if zoom200ep:
#     m0, s0 = m0[:200], s0[:200]
#     m1, s1 = m1[:200], s1[:200]
#     m2, s2 = m2[:200], s2[:200]
#     m3, s3 = m3[:200], s3[:200]
#     m4, s4 = m4[:200], s4[:200]

# fig = plt.figure()
# ax = fig.add_subplot(111)
# k=ax.plot(np.arange(len(m0)), m0, 'k', linewidth=2, label='instrumental')
# ax.fill_between(np.arange(len(m0)),m0-s0,m0+s0,alpha=.3, color='k')
# coral=ax.plot(np.arange(len(m1)), m1, 'coral', linewidth=2, label='$\omega=0.1$')
# ax.fill_between(np.arange(len(m1)),m1-s1,m1+s1,alpha=.3, color='coral')
# red=ax.plot(np.arange(len(m2)), m2, 'red', linewidth=2, label='$\omega=0.5$')
# ax.fill_between(np.arange(len(m2)),m2-s2,m2+s2,alpha=.3, color='red')
# darkred=ax.plot(np.arange(len(m3)), m3, 'darkred', linewidth=2, label='$\omega=0.9$')
# ax.fill_between(np.arange(len(m3)),m3-s3,m3+s3,alpha=.3, color='darkred')
# blue=ax.plot(np.arange(len(m4)), m4, 'blue', linewidth=2, label='flexible $\omega$')
# ax.fill_between(np.arange(len(m4)),m4-s4,m4+s4,alpha=.3, color='blue')
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# # plt.ylabel('Cumulative steps taken over episodes')
# plt.xlabel('Episode count')
# plt.legend()
# # plt.title("Cumulative steps")
# plt.show()

# # plot pain
# m0, s0 = np.mean(w000_cum_pains_all, axis=0), np.std(w000_cum_pains_all, axis=0)
# m1, s1 = np.mean(w010_cum_pains_all, axis=0), np.std(w010_cum_pains_all, axis=0)
# m2, s2 = np.mean(w050_cum_pains_all, axis=0), np.std(w050_cum_pains_all, axis=0)
# m3, s3 = np.mean(w090_cum_pains_all, axis=0), np.std(w090_cum_pains_all, axis=0)
# m4, s4 = np.mean(mw_1_cum_pains_all, axis=0), np.std(mw_1_cum_pains_all, axis=0)


# max_cum_pain = np.max(w090_cum_pains_all[:, -1])
# norm_cum_pain_w000 = w000_cum_pains_all[:, -1] / max_cum_pain
# norm_cum_pain_w010 = w010_cum_pains_all[:, -1] / max_cum_pain
# norm_cum_pain_w050 = w050_cum_pains_all[:, -1] / max_cum_pain
# norm_cum_pain_w090 = w090_cum_pains_all[:, -1] / max_cum_pain
# norm_cum_pain_mw_1 = mw_1_cum_pains_all[:, -1] / max_cum_pain


# # max_cum_pain = np.max(w090_cum_pains_all[:, -1])
# # min_cum_pain = np.min(w050_cum_pains_all[:, -1])
# # norm_cum_pain_w000 = (w000_cum_pains_all[:, -1] - min_cum_pain)/ (max_cum_pain - min_cum_pain)
# # norm_cum_pain_w010 = (w010_cum_pains_all[:, -1] - min_cum_pain)/ (max_cum_pain - min_cum_pain)
# # norm_cum_pain_w050 = (w050_cum_pains_all[:, -1] - min_cum_pain)/ (max_cum_pain - min_cum_pain)
# # norm_cum_pain_w090 = (w090_cum_pains_all[:, -1] - min_cum_pain)/ (max_cum_pain - min_cum_pain)
# # norm_cum_pain_mw_1 = (mw_1_cum_pains_all[:, -1] - min_cum_pain)/ (max_cum_pain - min_cum_pain)

# if zoom200ep:
#     m0, s0 = m0[:200], s0[:200]
#     m1, s1 = m1[:200], s1[:200]
#     m2, s2 = m2[:200], s2[:200]
#     m3, s3 = m3[:200], s3[:200]
#     m4, s4 = m4[:200], s4[:200]

# fig = plt.figure()
# ax = fig.add_subplot(111)
# k=ax.plot(np.arange(len(m0)), m0, 'k', linewidth=2, label='instrumental')
# ax.fill_between(np.arange(len(m0)),m0-s0,m0+s0,alpha=.3, color='k')
# coral=ax.plot(np.arange(len(m1)), m1, 'coral', linewidth=2, label='$\omega=0.1$')
# ax.fill_between(np.arange(len(m1)),m1-s1,m1+s1,alpha=.3, color='coral')
# red=ax.plot(np.arange(len(m2)), m2, 'red', linewidth=2, label='$\omega=0.5$')
# ax.fill_between(np.arange(len(m2)),m2-s2,m2+s2,alpha=.3, color='red')
# darkred=ax.plot(np.arange(len(m3)), m3, 'darkred', linewidth=2, label='$\omega=0.9$')
# ax.fill_between(np.arange(len(m3)),m3-s3,m3+s3,alpha=.3, color='darkred')
# blue=ax.plot(np.arange(len(m4)), m4, 'blue', linewidth=2, label='flexible $\omega$')
# ax.fill_between(np.arange(len(m4)),m4-s4,m4+s4,alpha=.3, color='blue')
# # plt.ylabel('Cumulative pain accrued over episodes')
# plt.xlabel('Episode count')
# ax.legend()
# # plt.title("Cumulative pain")
# plt.show()

# print(max_cum_pain, max_cum_step)

# metric_w000 = 1/(norm_cum_step_w000**2 + norm_cum_pain_w000**2)
# metric_w010 = 1/(norm_cum_step_w010**2 + norm_cum_pain_w010**2)
# metric_w050 = 1/(norm_cum_step_w050**2 + norm_cum_pain_w050**2)
# metric_w090 = 1/(norm_cum_step_w090**2 + norm_cum_pain_w090**2)
# metric_mw_1 = 1/(norm_cum_step_mw_1**2 + norm_cum_pain_mw_1**2)


# # metric_w000 = 1/(norm_cum_step_w000*norm_cum_pain_w000)
# # metric_w010 = 1/(norm_cum_step_w010*norm_cum_pain_w010)
# # metric_w050 = 1/(norm_cum_step_w050*norm_cum_pain_w050)
# # metric_w090 = 1/(norm_cum_step_w090*norm_cum_pain_w090)
# # metric_mw_1 = 1/(norm_cum_step_mw_1*norm_cum_pain_mw_1)


# fig = plt.figure()
# ax = fig.add_subplot(111)
# labels = ['instrumental', 'flexible $\omega$', '$\omega=0.1$','$\omega=0.5$', '$\omega=0.9$']
# mean_metric = [np.mean(metric_w000), np.mean(metric_mw_1), np.mean(metric_w010), np.mean(metric_w050), np.mean(metric_w090)]
# std_metric = [np.std(metric_w000), np.std(metric_mw_1), np.std(metric_w010), np.std(metric_w050), np.std(metric_w090)]
# ax.bar(labels,mean_metric, yerr=std_metric, color=['grey', 'blue', 'coral', 'red', 'darkred'], alpha=0.75)
# plt.ylabel('Trade-off metric')
# # plt.title('Performance on safety-efficiency trade-off metric')
# plt.show()


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


