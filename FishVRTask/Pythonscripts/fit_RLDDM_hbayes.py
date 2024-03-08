# use conda activate stan

import os
import stan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

if __name__ == "__main__":
    
    model_name = 'rlddm_PAL1'
    # model_name = 'rlddm_PAL2b'
    # model_name = 'rlddm_PAL2w'
    # model_name = 'rlddm_PAL3wb'
    # model_name = 'rlddm_PAL4wb_flexible'
    model_code = open('./stan_files/'+model_name+'.stan', 'r').read()
    data = pd.read_csv('4_subj_hbayesdata.txt', sep='\t')

    # Use general_info(s) about raw_datas
    subj_ls = np.unique(data['subjID'])
    n_subj = len(subj_ls)
    t_subjs = np.array([data[data['subjID']==x].shape[0] for x in subj_ls])
    t_max = max(t_subjs)

    # Initialize (model-specific) data arrays
    cue = np.full((n_subj, t_max), -1, dtype=int)
    pressed = np.full((n_subj, t_max), -1, dtype=int)
    outcome = np.full((n_subj, t_max), 0, dtype=float)
    rt = np.full((n_subj, t_max), -1, dtype=float)
    subjID = np.full((n_subj),0,dtype=int)

    # Write from subj_data to the data arrays
    for s in range(n_subj):
        subj_data = data[data['subjID']==subj_ls[s]]
        t = t_subjs[s]
        cue[s][:t] = subj_data['cue']
        pressed[s][:t] = subj_data['keyPressed']
        outcome[s][:t] = subj_data['outcome']
        subjID[s] = pd.unique(subj_data['subjID'])
        rt[s][:t] = subj_data['rt']

    # Wrap into a dict for pystan
    data_dict = {
        'N': int(n_subj),
        'T': int(t_max),
        'Tsubj': t_subjs.astype(int),
        'cue': cue,
        'pressed': pressed,
        'outcome': outcome,
        'rt': rt,
        'subjID': subjID,
        'minRT': np.min(rt, axis=1),
        'RTbound': np.min(rt)
    }

    # keys_priors = ["mu_mu", "sd_mu", "mu_sd", "sd_sd"]
    # priors = dict(  alpha_priors={'mu_mu':0, 'sd_mu':1, 'mu_sd':0, 'sd_sd':.1},
    #                 inv_temp_priors={'mu_mu':1, 'sd_mu':30, 'mu_sd':0, 'sd_sd':30},
    #                 )
    # # Add priors:
    # print("Fitting the model using the priors:")
    # for par in priors.keys():
    #     data_dict.update({par: [priors[par][key] for key in keys_priors]})
    #     print(par, priors[par])
        
    # fit stan model
    posterior = stan.build(program_code=model_code, data=data_dict)
    fit = posterior.sample(num_samples=2000, num_chains=4, num_warmup=1000)
    df_fit = fit.to_frame()

    isExist = os.path.exists('output_fits/'+model_name)
    if not isExist:
        os.mkdir('output_fits/'+model_name)
    df_fit.to_csv('output_fits/'+model_name+'/df_fit.csv', index=False)

    # df_fit = pd.read_csv('df_fit.csv')

    # Get WAIC
    cols = []
    for n in range(int(n_subj)):
        for i in range(240):
            key = 'log_lik.'+str(n+1)+'.'+str(i+1)
            cols.append(key)

    log_likelihoods = np.array(df_fit[cols])
    likelihoods = np.exp(log_likelihoods)
    mean_l = np.mean(likelihoods, axis=0)
    pointwise_lppd = np.log(mean_l)
    lppd = np.sum(pointwise_lppd)
    pointwise_var_l = np.var(log_likelihoods, axis=0)
    var_l = np.sum(pointwise_var_l)

    N=np.sum(t_subjs)
    pointwise_waic = - 2*pointwise_lppd +  2*pointwise_var_l
    waic = -2*lppd + 2*var_l
    waic_se = np.sqrt(N * np.var(pointwise_waic))

    print("WAIC: ", str(waic), "; WAIC SE: ", str(waic_se))
    with open('output_fits/'+model_name+'/waic.txt', 'w') as f:
        f.write("WAIC: " +  str(waic) + "; WAIC SE: " + str(waic_se) + "\n Pointwise WAIC: " + str(pointwise_waic))

    # # Get hierarchical and individual parameter distributions plots
    # fig1, ax1 = plt.subplots()
    # ax1.hist(df_fit['mu_pr.1'], color='blue', label='mean')
    # ax1.hist(df_fit['sigma.1'], color='orange', label='std')
    # ax1.set_title("Exploration noise (hierarchical)")
    # ax1.legend()
    # plt.savefig('output_fits/'+model_name+'/hier_exp_noise.png')

    # isExist = os.path.exists('output_fits/'+model_name+'/ind_exp_noise')
    # if not isExist:
    #     os.mkdir('output_fits/'+model_name+'/ind_exp_noise')
    # for n in range(int(n_subj)):
    #     fig, ax = plt.subplots()
    #     ax.hist(df_fit['xi_pr.'+str(n+1)], color='blue')
    #     ax.set_title("Exploration noise (subject " + str(n+1)+")")
    #     plt.savefig('output_fits/'+model_name+'/ind_exp_noise/subject_' + str(n+1)+ '.png')


    # fig2, ax2 = plt.subplots()
    # ax2.hist(df_fit['mu_pr.2'], color='blue', label='mean')
    # ax2.hist(df_fit['sigma.2'], color='orange', label='std')
    # ax2.set_title("Learning rate (hierarchical)")
    # ax2.legend()
    # plt.savefig('output_fits/'+model_name+'/hier_learning_rate.png')

    # isExist = os.path.exists('output_fits/'+model_name+'/ind_learning_rate')
    # if not isExist:
    #     os.mkdir('output_fits/'+model_name+'/ind_learning_rate')
    # for n in range(int(n_subj)):
    #     fig, ax = plt.subplots()
    #     ax.hist(df_fit['ep_pr.'+str(n+1)], color='blue')
    #     ax.set_title("Learning rate (subject " + str(n+1)+")")
    #     plt.savefig('output_fits/'+model_name+'/ind_learning_rate/subject_' + str(n+1)+ '.png')
    
    # if False:
    #     fig3, ax3 = plt.subplots()
    #     ax3.hist(df_fit['mu_pr.3'], color='blue', label='mean')
    #     ax3.hist(df_fit['sigma.3'], color='orange', label='std')
    #     ax3.set_title("Baseline approach bias (hierarchical)")
    #     ax3.legend()
    #     plt.savefig('output_fits/'+model_name+'/hier_baseline_go_bias.png')

    #     isExist = os.path.exists('output_fits/'+model_name+'/ind_baseline_approach_bias')
    #     if not isExist:
    #         os.mkdir('output_fits/'+model_name+'/ind_baseline_approach_bias')
    #     for n in range(int(n_subj)):
    #         fig, ax = plt.subplots()
    #         ax.hist(df_fit['b_pr.'+str(n+1)], color='blue')
    #         ax.set_title("Baseline approach bias (subject " + str(n+1)+")")
    #         plt.savefig('output_fits/'+model_name+'/ind_baseLine_approach_bias/subject_' + str(n+1)+ '.png')

    #     fig4, ax4 = plt.subplots()
    #     ax4.hist(df_fit['mu_pr.4'], color='blue', label='mean')
    #     ax4.hist(df_fit['sigma.4'], color='orange', label='std')
    #     ax4.set_title("Pavlovian withdrawal bias parameter (hierarchical)")
    #     ax4.legend()
    #     plt.savefig('output_fits/'+model_name+'/hier_pavlovian_withdrawal_bias.png')

    #     isExist = os.path.exists('output_fits/'+model_name+'/ind_pavlovian_withdrawal_bias')
    #     if not isExist:
    #         os.mkdir('output_fits/'+model_name+'/ind_pavlovian_withdrawal_bias')
    #     for n in range(int(n_subj)):
    #         fig, ax = plt.subplots()
    #         ax.hist(df_fit['pi_pr.'+str(n+1)], color='blue')
    #         ax.set_title("Pavlovian withdrawal bias (subject " + str(n+1)+")")
    #         plt.savefig('output_fits/'+model_name+'/ind_pavlovian_withdrawal_bias/subject_' + str(n+1)+ '.png')

    # fig5, ax5 = plt.subplots()
    # ax5.hist(df_fit['mu_pr.3'], color='blue', label='mean')
    # ax5.hist(df_fit['sigma.3'], color='orange', label='std')
    # ax5.set_title("Punishment sensitivity (hierarchical)")
    # ax5.legend()
    # plt.savefig('output_fits/'+model_name+'/hier_punishment_sensitivity.png')

    # isExist = os.path.exists('output_fits/'+model_name+'/ind_punishment_sensitivity')
    # if not isExist:
    #     os.mkdir('output_fits/'+model_name+'/ind_punishment_sensitivity')
    # for n in range(int(n_subj)):
    #     fig, ax = plt.subplots()
    #     ax.hist(df_fit['rho_pr.'+str(n+1)], color='blue')
    #     ax.set_title("Punishment sensitivity (subject " + str(n+1)+")")
    #     plt.savefig('output_fits/'+model_name+'/ind_punishment_sensitivity/subject_' + str(n+1)+ '.png')


    # # Get posterior predictive checks
    # isExist = os.path.exists('output_fits/'+model_name+'/posterior_predictive_checks')
    # if not isExist:
    #     os.mkdir('output_fits/'+model_name+'/posterior_predictive_checks')
    # for n in range(int(n_subj)):
    #     cols = []
    #     for i in range(240):
    #         key = 'y_pred.'+str(n+1)+'.'+str(i+1)
    #         cols.append(key)
    #     pp_y_pred = np.array(np.mean(df_fit[cols], axis=0))
    #     fig1, ax1 = plt.subplots()
    #     fig1.set_figheight(5)
    #     fig1.set_figwidth(10)
    #     ax1.plot(pressed[n,:], color='blue', label='true_choices')
    #     ax1.plot(pp_y_pred, color='orange', label='model_predictions')
    #     ax1.set_title("Pavlovian withdrawal bias (subject " + str(n+1)+")")
    #     ax1.legend()
    #     plt.savefig('output_fits/'+model_name+'/posterior_predictive_checks/subject_' + str(n+1)+ '.png')


    # Get trace plots



    


