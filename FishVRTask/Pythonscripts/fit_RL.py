# use conda activate stan

import stan
import pandas as pd
import numpy as np

if __name__ == "__main__":
    data = pd.read_csv('data_df.csv')
    model_code = open('./stan_files/hierRL_instrumental.stan', 'r').read()


    data.reset_index(inplace=True) # reset index
    N = data.shape[0] # n observations
    L = len(pd.unique(data.participant)) # n subjects (levels)

    keys_priors = ["mu_mu", "sd_mu", "mu_sd", "sd_sd"]
    priors = dict(  alpha_priors={'mu_mu':0, 'sd_mu':1, 'mu_sd':0, 'sd_sd':.1},
                    inv_temp_priors={'mu_mu':1, 'sd_mu':30, 'mu_sd':0, 'sd_sd':30},
                    )

    data_dict = {'N': N,
                'trial_block': data['trial_block'].values.astype(int),
                'block_label': data['block_label'].values.astype(int),
                'cue': data['cue'].values.astype(int),
                'action': data['action'].values.astype(int),
                'outcome': data['outcome'].values,
                
                'L': L, 
                'participant': data['participant'].values.astype(int),

                'initial_value': 0,
                }

    # Add priors:
    print("Fitting the model using the priors:")
    for par in priors.keys():
        data_dict.update({par: [priors[par][key] for key in keys_priors]})
        print(par, priors[par])
        
    # fit stan model
    posterior = stan.build(program_code=model_code, data=data_dict)
    fit = posterior.sample(num_samples=2000, num_chains=4, num_warmup=1000)
    df_fit = fit.to_frame()

    # print(df_fit)
    df_fit.to_csv('df_fit.csv', index=False)

    # df_fit = pd.read_csv('df_fit.csv')
    # print(df_fit['log_lik.1'])
    
    log_likelihood = fit['log_lik'] # n_samples X N observations
    likelihood = np.exp(log_likelihood)

    mean_l = np.mean(likelihood, axis=0) # N observations

    pointwise_lppd = np.log(mean_l)
    lppd = np.sum(pointwise_lppd)

    pointwise_var_l = np.var(log_likelihood, axis=0) # N observations
    var_l = np.sum(pointwise_var_l)

    pointwise_waic = - 2*pointwise_lppd +  2*pointwise_var_l
    waic = -2*lppd + 2*var_l
    waic_se = np.sqrt(N * np.var(pointwise_waic))

    print(lppd, pointwise_waic, waic, waic_se)




