import numpy as np
import matplotlib.pyplot as plt

number_of_runs = 10
zoom200ep = False

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title


env_folder = './multiple_run_outputs/fear_avoidance_gridworld/'

mw_config = 'modulate_omega_kappa=3_beta=0.01/'

#### Co-learn pav plotting ####

w000_cum_pains_all = []
w010_cum_pains_all = []
w050_cum_pains_all = []
w090_cum_pains_all = []
mw_1_cum_pains_all = []

w000_cum_CF_all = []
w010_cum_CF_all = []
w050_cum_CF_all = []
w090_cum_CF_all = []
mw_1_cum_CF_all = []

for run_num in range(number_of_runs):
    w000_cum_CF = np.load(env_folder + 'omega=0/cum_completion_failures/run_'+str(run_num)+'.npy')
    w010_cum_CF = np.load(env_folder + 'omega=0.1/cum_completion_failures/run_'+str(run_num)+'.npy')
    w050_cum_CF = np.load(env_folder + 'omega=0.5/cum_completion_failures/run_'+str(run_num)+'.npy')
    w090_cum_CF = np.load(env_folder + 'omega=0.9/cum_completion_failures/run_'+str(run_num)+'.npy')
    mw_1_cum_CF = np.load(env_folder + mw_config + 'cum_completion_failures/run_'+str(run_num)+'.npy')

    w000_cum_pains = np.load(env_folder + 'omega=0/cum_pains/run_'+str(run_num)+'.npy')
    w010_cum_pains = np.load(env_folder + 'omega=0.1/cum_pains/run_'+str(run_num)+'.npy')
    w050_cum_pains = np.load(env_folder + 'omega=0.5/cum_pains/run_'+str(run_num)+'.npy')
    w090_cum_pains = np.load(env_folder + 'omega=0.9/cum_pains/run_'+str(run_num)+'.npy')
    mw_1_cum_pains = np.load(env_folder + mw_config + 'cum_pains/run_'+str(run_num)+'.npy')

    w000_cum_CF_all.append(w000_cum_CF[-1])
    w010_cum_CF_all.append(w010_cum_CF[-1])
    w050_cum_CF_all.append(w050_cum_CF[-1])
    w090_cum_CF_all.append(w090_cum_CF[-1])
    mw_1_cum_CF_all.append(mw_1_cum_CF[-1])

    w000_cum_pains_all.append(w000_cum_pains[-1])
    w010_cum_pains_all.append(w010_cum_pains[-1])
    w050_cum_pains_all.append(w050_cum_pains[-1])
    w090_cum_pains_all.append(w090_cum_pains[-1])
    mw_1_cum_pains_all.append(mw_1_cum_pains[[-1]])

w000_cum_CF_all = np.array(w000_cum_CF_all)
w010_cum_CF_all = np.array(w010_cum_CF_all)
w050_cum_CF_all = np.array(w050_cum_CF_all)
w090_cum_CF_all = np.array(w090_cum_CF_all)
mw_1_cum_CF_all = np.array(mw_1_cum_CF_all)   

w000_cum_pains_all = np.array(w000_cum_pains_all)
w010_cum_pains_all = np.array(w010_cum_pains_all)
w050_cum_pains_all = np.array(w050_cum_pains_all)
w090_cum_pains_all = np.array(w090_cum_pains_all)
mw_1_cum_pains_all = np.array(mw_1_cum_pains_all)


fig = plt.figure()
ax = fig.add_subplot(111)
labels = ['instrumental', 'flexible $\omega$', '$\omega=0.1$','$\omega=0.5$', '$\omega=0.9$']
cum_CFs = [np.mean(w000_cum_CF_all), np.mean(mw_1_cum_CF_all), np.mean(w010_cum_CF_all), np.mean(w050_cum_CF_all), np.mean(w090_cum_CF_all)]
std_CFs = [np.std(w000_cum_CF_all), np.std(mw_1_cum_CF_all), np.std(w010_cum_CF_all), np.std(w050_cum_CF_all), np.std(w090_cum_CF_all)]
ax.bar(labels,cum_CFs, yerr=std_CFs, color=['grey', 'blue', 'coral', 'red', 'darkred'])
plt.ylabel('Cumulative failures to reach the reward')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
labels = ['instrumental', 'flexible $\omega$', '$\omega=0.1$','$\omega=0.5$', '$\omega=0.9$']
cum_pains = [np.mean(w000_cum_pains_all), np.mean(mw_1_cum_pains_all), np.mean(w010_cum_pains_all), np.mean(w050_cum_pains_all), np.mean(w090_cum_pains_all)]
std_pains = [np.std(w000_cum_pains_all), np.std(mw_1_cum_pains_all), np.std(w010_cum_pains_all), np.std(w050_cum_pains_all), np.std(w090_cum_pains_all)]
ax.bar(labels,cum_pains, yerr=std_pains, color=['grey', 'blue', 'coral', 'red', 'darkred'])
plt.ylabel('Cumulative pain accrued over episodes')
plt.show()

## Doesn't work

# max_cum_step = np.max(w090_cum_CF_all)
# norm_cum_step_w000 = w000_cum_CF_all / max_cum_step
# norm_cum_step_w010 = w010_cum_CF_all / max_cum_step
# norm_cum_step_w050 = w050_cum_CF_all / max_cum_step
# norm_cum_step_w090 = w090_cum_CF_all / max_cum_step
# norm_cum_step_mw_1 = mw_1_cum_CF_all / max_cum_step


# max_cum_pain = np.max(w090_cum_pains_all)
# norm_cum_pain_w000 = w000_cum_pains_all / max_cum_pain
# norm_cum_pain_w010 = w010_cum_pains_all / max_cum_pain
# norm_cum_pain_w050 = w050_cum_pains_all / max_cum_pain
# norm_cum_pain_w090 = w090_cum_pains_all / max_cum_pain
# norm_cum_pain_mw_1 = mw_1_cum_pains_all / max_cum_pain


# metric_w000 = 1/(norm_cum_step_w000**2 + norm_cum_pain_w000**2)
# metric_w010 = 1/(norm_cum_step_w010**2 + norm_cum_pain_w010**2)
# metric_w050 = 1/(norm_cum_step_w050**2 + norm_cum_pain_w050**2)
# metric_w090 = 1/(norm_cum_step_w090**2 + norm_cum_pain_w090**2)
# metric_mw_1 = 1/(norm_cum_step_mw_1**2 + norm_cum_pain_mw_1**2)


# metric_w000 = 1/(norm_cum_step_w000*norm_cum_pain_w000)
# metric_w010 = 1/(norm_cum_step_w010*norm_cum_pain_w010)
# metric_w050 = 1/(norm_cum_step_w050*norm_cum_pain_w050)
# metric_w090 = 1/(norm_cum_step_w090*norm_cum_pain_w090)
# metric_mw_1 = 1/(norm_cum_step_mw_1*norm_cum_pain_mw_1)


# fig = plt.figure()
# ax = fig.add_subplot(111)
# labels = ['instrumental', 'flexible $\omega$', '$\omega=0.1$','$\omega=0.5$', '$\omega=0.9$']
# mean_metric = [np.mean(metric_w000), np.mean(metric_mw_1), np.mean(metric_w010), np.mean(metric_w050), np.mean(metric_w090)]
# std_metric = [np.std(metric_w000), np.std(metric_mw_1), np.std(metric_w010), np.std(metric_w050), np.std(metric_w090)]
# ax.bar(labels,mean_metric, yerr=std_metric, color=['grey', 'blue', 'coral', 'red', 'darkred'])
# plt.ylabel('Trade-off metric')
# plt.show()