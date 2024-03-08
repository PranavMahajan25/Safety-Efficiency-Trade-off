// Model 6 from Pedersen, Frank & Biele (2017) https://doi.org/10.3758/s13423-016-1199-y

functions{
  // Random number generator from Shahar et al. (2019) https://doi.org/10.1371/journal.pcbi.1006803
  vector wiener_rng(real a, real tau, real z, real d) {
    real dt;
    real sigma;
    real p;
    real y;
    real i;
    real aa;
    real ch;
    real rt;
    vector[2] ret;

    dt = .0001;
    sigma = 1;

    y = z * a;  // starting point
    p = .5 * (1 + ((d * sqrt(dt)) / sigma));
    i = 0;
    while (y < a && y > 0){
      aa = uniform_rng(0,1);
      if (aa <= p){
        y = y + sigma * sqrt(dt);
        i = i + 1;
      } else {
        y = y - sigma * sqrt(dt);
        i = i + 1;
      }
    }
    ch = (y >= a) * 1 + 0;  // Upper boundary choice -> 1, lower boundary choice -> 0
    rt = i * dt + tau;

    ret[1] = ch;
    ret[2] = rt;
    return ret;
  }
}

data {
  int<lower=1> N; 
  int<lower=1> T;
  int<lower=1, upper=T> Tsubj[N];
  int<lower=1, upper=4> cue[N, T];
  int<lower=-1, upper=1> pressed[N, T];
  real outcome[N, T];
  real <lower=0> rt[N, T];
  real <lower=0> minRT[N];                          // Minimum RT for each subject of the observed data
  real <lower=0> RTbound;                           // Lower bound or RT across all subjects (e.g., 0.1 second)
}

transformed data {
  vector[4] initV;
  initV = rep_vector(0.0, 4);
  real<lower=0, upper=1> starting_point;			// starting point diffusion model not to estimate
  starting_point = 0.5; 

}

parameters {
  // declare as vectors for vectorizing
  vector[4] mu_pr;
  vector<lower=0>[4] sigma;
  vector[N] beta_pr;        // drift rate scaling, similar to inv temp
  vector[N] alpha_pr;       // learning rate
  vector[N] threshold_pr;   // threshold for DDM
  vector[N] n_dt_pr;        // non decision time
}

transformed parameters {
  vector<lower=0>[N] beta;
  vector<lower=0, upper=1>[N] alpha;
  vector<lower=0>[N] threshold_t;
  // vector<lower=RTbound, upper=max(minRT)>[N] ndt_t;
  vector<lower=0> [N] ndt_t;


  beta = exp(mu_pr[1] + sigma[1] * beta_pr); // can also do log(1+exp())
  for (i in 1:N) {
    alpha[i] = Phi_approx(mu_pr[2] + sigma[2] * alpha_pr[i]);
  }
  threshold_t = exp(mu_pr[3] + sigma[3] * threshold_pr);
  ndt_t = exp(mu_pr[4] + sigma[4] * n_dt_pr);
  
}

model {
// gng_m4: RW(rew/pun) + noise + bias + pi model (M5 in Cavanagh et al 2013 J Neuro)
  // hyper parameters
  mu_pr[1:4]  ~ normal(0, 1.0);
  sigma[1:4] ~ normal(0, 0.2);

  // individual parameters w/ Matt trick
  beta_pr  ~ normal(0, 1.0);
  alpha_pr  ~ normal(0, 1.0);
  threshold_pr ~ normal(0, 1.0);
  n_dt_pr ~ normal(0, 1.0);

  for (i in 1:N) {
    vector[4] wv_g;  // action weight for go
    vector[4] wv_ng; // action weight for nogo
    vector[4] qv_g;  // Q value for go
    vector[4] qv_ng; // Q value for nogo
    vector[4] sv;    // stimulus value
    vector[4] drift_t;   // drift rate (scaled Go minus No Go) for each stimulus

    wv_g  = initV;
    wv_ng = initV;
    qv_g  = initV;
    qv_ng = initV;
    sv    = initV;

    for (t in 1:Tsubj[i]) {
      wv_g[cue[i, t]]  = qv_g[cue[i, t]];
      wv_ng[cue[i, t]] = qv_ng[cue[i, t]];  
      drift_t[cue[i, t]]   = beta[i] * (wv_g[cue[i, t]] - wv_ng[cue[i, t]]); // might need it to be w_chosen_action - w_non_chosen_action 
      if (pressed[i, t]) { // update go value
        rt[i,t] ~ wiener(threshold_t[i], ndt_t[i], starting_point, drift_t[cue[i, t]]); // check weiner; it only gives rt to upper boundary
      } else { // update no-go value
        rt[i,t] ~ wiener(threshold_t[i], ndt_t[i], starting_point, -drift_t[cue[i, t]]); // check weiner; it only gives rt to upper boundary
      }
      
      // after receiving feedback, update sv[t + 1]
      sv[cue[i, t]] += alpha[i] * (outcome[i, t] - sv[cue[i, t]]);

      // update action values
      if (pressed[i, t]) { // update go value
        qv_g[cue[i, t]] += alpha[i] * (outcome[i, t] - qv_g[cue[i, t]]);
      } else { // update no-go value
        qv_ng[cue[i, t]] += alpha[i] * (outcome[i, t] - qv_ng[cue[i, t]]);
      }
    } // end of t loop
  } // end of i loop
}

generated quantities {
  real<lower=0> mu_beta;
  real<lower=0, upper=1> mu_alpha;
  real<lower=0> mu_threshold;
  real<lower=0> mu_ndt;

  real log_lik[N, T]; //real log_lik[N];
  real Qgo[N, T];
  real Qnogo[N, T];
  real Wgo[N, T];
  real Wnogo[N, T];
  real SV[N, T];
  
  // For posterior predictive check (one-step method)
  matrix[N, T] choice_os;
  matrix[N, T] RT_os;
  vector[2]    tmp_os;

  // // For posterior predictive check (simulation method)
  // matrix[N, T] choice_sm;
  // matrix[N, T] RT_sm;
  // matrix[N, T] fd_sm;
  // vector[2]    tmp_sm;
  // real         rand;

  mu_beta = exp(mu_pr[1]);
  mu_alpha  = Phi_approx(mu_pr[2]);
  mu_threshold = exp(mu_pr[3]);
  mu_ndt = exp(mu_pr[4]);

  // Set all posterior predictions to -1 (avoids NULL values)
  for (i in 1:N) {
    for (t in 1:T) {
      choice_os[i, t] = -1;
      RT_os[i, t]     = -1;
      // choice_sm[i, t] = -1;
      // RT_sm[i, t]     = -1;
      // fd_sm[i, t]     = -1;
    }
  }

  { // local section, this saves time and space
    for (i in 1:N) {
      vector[4] wv_g;  // action weight for go
      vector[4] wv_ng; // action weight for nogo
      vector[4] qv_g;  // Q value for go
      vector[4] qv_ng; // Q value for nogo
      vector[4] sv;    // stimulus value
      vector[4] drift_t;   // drift rate (scaled Go minus No Go) for each stimulus

      wv_g  = initV;
      wv_ng = initV;
      qv_g  = initV;
      qv_ng = initV;
      sv    = initV;

      // log_lik[i] = 0;

      for (t in 1:Tsubj[i]) {
        log_lik[i,t] = 0;
        wv_g[cue[i, t]]  = qv_g[cue[i, t]];
        wv_ng[cue[i, t]] = qv_ng[cue[i, t]];  

        //////////// Posterior predictive check (one-step method) ////////////
        drift_t[cue[i, t]]   = beta[i] * (wv_g[cue[i, t]] - wv_ng[cue[i, t]]);
        // Drift diffusion process
        if (pressed[i, t]) {
          log_lik[i, t] = wiener_lpdf(rt[i, t] | threshold_t[i], ndt_t[i], starting_point, drift_t[cue[i, t]]);
        } else {
          log_lik[i, t] = wiener_lpdf(rt[i, t] | threshold_t[i], ndt_t[i], starting_point, -drift_t[cue[i, t]]);
        }
        tmp_os          = wiener_rng(threshold_t[i], ndt_t[i], starting_point, drift_t[cue[i, t]]);
        choice_os[i, t] = tmp_os[1];
        RT_os[i, t]     = tmp_os[2];


        // Model regressors --> store values before being updated
        Qgo[i, t]   = qv_g[cue[i, t]];
        Qnogo[i, t] = qv_ng[cue[i, t]];
        Wgo[i, t]   = wv_g[cue[i, t]];
        Wnogo[i, t] = wv_ng[cue[i, t]];
        SV[i, t]    = sv[cue[i, t]];

        // after receiving feedback, update sv[t + 1]
        sv[cue[i, t]] += alpha[i] * (outcome[i, t] - sv[cue[i, t]]);

        // update action values
        if (pressed[i, t]) { // update go value
          qv_g[cue[i, t]] += alpha[i] * (outcome[i, t] - qv_g[cue[i, t]]);
        } else { // update no-go value
          qv_ng[cue[i, t]] += alpha[i] * (outcome[i, t] - qv_ng[cue[i, t]]);
        }

        //////////// Posterior predictive check (simulation method) ////////////
        // Not yet implemented: https://github.com/CCS-Lab/hBayesDM/blob/develop/commons/stan_files/pstRT_rlddm6.stan


      } // end of t loop
    } // end of i loop
  } // end of local section
}

