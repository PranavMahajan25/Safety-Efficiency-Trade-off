data {
  int<lower=1> N; 
  int<lower=1> T;
  int<lower=1, upper=T> Tsubj[N];
  int<lower=1, upper=4> cue[N, T];
  int<lower=-1, upper=1> pressed[N, T];
  real outcome[N, T];
}

transformed data {
  vector[4] initV;
  initV = rep_vector(0.0, 4);
}

parameters {
  // declare as vectors for vectorizing
  vector[6] mu_pr;
  vector<lower=0>[6] sigma;
  vector[N] beta_pr;        // inv temp
  vector[N] alpha_pr;       // learning rate
  vector[N] b_pr;           // baseline bias
  vector[N] alpha_assoc_pr;     //assoc learning rate
  vector[N] kappa_pr;       // scalar multiplier for omega
  vector[N] assoc0_pr;         // initial value of assoc
}

transformed parameters {
  vector<lower=0>[N] beta;
  vector<lower=0, upper=1>[N] alpha;
  vector[N] b;
  vector<lower=0, upper=1>[N] alpha_assoc;
  vector<lower=0>[N] kappa;
  vector<lower=0>[N] assoc0;

  beta = exp(mu_pr[1] + sigma[1] * beta_pr);
  for (i in 1:N) {
    alpha[i] = Phi_approx(mu_pr[2] + sigma[2] * alpha_pr[i]);
    alpha_assoc[i] = Phi_approx(mu_pr[4] + sigma[4] * alpha_assoc_pr[i]);
  }
  b = mu_pr[3] + sigma[3] * b_pr;
  kappa = exp(mu_pr[5] + sigma[5] * kappa_pr);
  assoc0 = exp(mu_pr[6] + sigma[6] * assoc0_pr);

}

model {
// gng_m4: RW(rew/pun) + noise + bias + pi model (M5 in Cavanagh et al 2013 J Neuro)
  // hyper parameters
  mu_pr[1:6]  ~ normal(0, 1.0);
  sigma[1:6] ~ normal(0, 0.2);

  // individual parameters w/ Matt trick
  beta_pr  ~ normal(0, 1.0);
  alpha_pr  ~ normal(0, 1.0);
  b_pr ~ normal(0, 1.0);
  alpha_assoc_pr  ~ normal(0, 1.0);
  kappa_pr  ~ normal(0, 1.0);
  assoc0_pr  ~ normal(0, 1.0);

  for (i in 1:N) {
    vector[4] wv_g;  // action weight for go
    vector[4] wv_ng; // action weight for nogo
    vector[4] qv_g;  // Q value for go
    vector[4] qv_ng; // Q value for nogo
    vector[4] sv;    // stimulus value
    vector[4] pGo;   // prob of go (press)

    real omega; // PIT parameter flexible
    real assoc; // assoc value which is updated each trial
    real absRPE; // absRPE each trial

    wv_g  = initV;
    wv_ng = initV;
    qv_g  = initV;
    qv_ng = initV;
    sv    = initV;
    assoc = assoc0[i];

    for (t in 1:Tsubj[i]) {
      // omega = 1/(1+exp(-kappa[i]*(assoc - assoc0[i])));
      if (kappa[i]*assoc < 1.0){
        omega = kappa[i]*assoc;
      }
      else {
        omega = 1.0;
      }

      wv_g[cue[i, t]]  = (1 - omega) * qv_g[cue[i, t]] + b[i];
      wv_ng[cue[i, t]] = (1 - omega) * qv_ng[cue[i, t]] + omega * (-sv[cue[i, t]]);  
      pGo[cue[i, t]]   = inv_logit(beta[i] * (wv_g[cue[i, t]] - wv_ng[cue[i, t]]));
      pressed[i, t] ~ bernoulli(pGo[cue[i, t]]);

      // after receiving feedback, update sv[t + 1]
      sv[cue[i, t]] += alpha[i] * (outcome[i, t] - sv[cue[i, t]]);

      // update action values
      if (pressed[i, t]) { // update go value
        qv_g[cue[i, t]] += alpha[i] * (outcome[i, t] - qv_g[cue[i, t]]);
        absRPE = abs(outcome[i, t] - qv_g[cue[i, t]]);
      } else { // update no-go value
        qv_ng[cue[i, t]] += alpha[i] * (outcome[i, t] - qv_ng[cue[i, t]]);
        absRPE = abs(outcome[i, t] - qv_ng[cue[i, t]]);
      }
      assoc += alpha_assoc[i]*alpha[i]*(absRPE - assoc); 
    } // end of t loop
  } // end of i loop
}

generated quantities {
  real<lower=0> mu_beta;
  real<lower=0, upper=1> mu_alpha;
  real mu_b;
  real<lower=0, upper=1> mu_alpha_assoc;
  real<lower=0> mu_kappa;
  real<lower=0> mu_assoc0;
  real log_lik[N, T]; //real log_lik[N];
  real Qgo[N, T];
  real Qnogo[N, T];
  real Wgo[N, T];
  real Wnogo[N, T];
  real SV[N, T];
  real Assoc_Arr[N, T];
  real Omega_Arr[N, T];

  // For posterior predictive check
  real y_pred[N, T];

  // Set all posterior predictions to 0 (avoids NULL values)
  for (i in 1:N) {
    for (t in 1:T) {
      y_pred[i, t] = -1;
    }
  }

  mu_beta = exp(mu_pr[1]);
  mu_alpha  = Phi_approx(mu_pr[2]);
  mu_b = mu_pr[3];
  mu_alpha_assoc  = Phi_approx(mu_pr[4]);
  mu_kappa  = exp(mu_pr[5]);
  mu_assoc0  = exp(mu_pr[6]);

  { // local section, this saves time and space
    for (i in 1:N) {
      vector[4] wv_g;  // action weight for go
      vector[4] wv_ng; // action weight for nogo
      vector[4] qv_g;  // Q value for go
      vector[4] qv_ng; // Q value for nogo
      vector[4] sv;    // stimulus value
      vector[4] pGo;   // prob of go (press)

      real omega; // PIT parameter flexible
      real assoc; // assoc value which is updated each trial
      real absRPE; // absRPE each trial

      wv_g  = initV;
      wv_ng = initV;
      qv_g  = initV;
      qv_ng = initV;
      sv    = initV;
      assoc = assoc0[i];

      // log_lik[i] = 0;

      for (t in 1:Tsubj[i]) {
        log_lik[i,t] = 0;
        // omega = 1/(1+exp(-kappa[i]*(assoc - assoc0[i])));
        if (kappa[i]*assoc < 1.0){
          omega = kappa[i]*assoc;
        }
        else {
          omega = 1.0;
        }
        wv_g[cue[i, t]]  = (1 - omega) * qv_g[cue[i, t]] + b[i];
        wv_ng[cue[i, t]] = (1 - omega) * qv_ng[cue[i, t]] + omega * (-sv[cue[i, t]]);   
        pGo[cue[i, t]]   = inv_logit(beta[i] * (wv_g[cue[i, t]] - wv_ng[cue[i, t]]));
        log_lik[i, t] += bernoulli_lpmf(pressed[i, t] | pGo[cue[i, t]]);

        // generate posterior prediction for current trial
        y_pred[i, t] = bernoulli_rng(pGo[cue[i, t]]);

        // Model regressors --> store values before being updated
        Qgo[i, t]   = qv_g[cue[i, t]];
        Qnogo[i, t] = qv_ng[cue[i, t]];
        Wgo[i, t]   = wv_g[cue[i, t]];
        Wnogo[i, t] = wv_ng[cue[i, t]];
        SV[i, t]    = sv[cue[i, t]];
        Assoc_Arr[i, t] = assoc;
        Omega_Arr[i, t] = omega;

        // after receiving feedback, update sv[t + 1]
        sv[cue[i, t]] += alpha[i] * (outcome[i, t] - sv[cue[i, t]]);

        // update action values
        if (pressed[i, t]) { // update go value
          qv_g[cue[i, t]] += alpha[i] * (outcome[i, t] - qv_g[cue[i, t]]);
          absRPE = abs(outcome[i, t] - qv_g[cue[i, t]]);
        } else { // update no-go value
          qv_ng[cue[i, t]] += alpha[i] * (outcome[i, t] - qv_ng[cue[i, t]]);
          absRPE = abs(outcome[i, t] - qv_ng[cue[i, t]]);
        }
        assoc += alpha_assoc[i]*alpha[i]*(absRPE - assoc); 
      } // end of t loop
    } // end of i loop
  } // end of local section
}

