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
  vector[3] mu_pr;
  vector<lower=0>[3] sigma;
  vector[N] invtemp_pr;        // inv temp
  vector[N] alpha_pr;       // learning rate
  vector[N] b_pr;           // baseline bias
}

transformed parameters {
  vector<lower=0>[N] invtemp;
  vector<lower=0, upper=1>[N] alpha;
  vector[N] b;


  invtemp = exp(mu_pr[1] + sigma[1] * invtemp_pr);
  for (i in 1:N) {
    alpha[i] = Phi_approx(mu_pr[2] + sigma[2] * alpha_pr[i]);
  }
  b = mu_pr[3] + sigma[3] * b_pr;
}

model {
// gng_m4: RW(rew/pun) + noise + bias + pi model (M5 in Cavanagh et al 2013 J Neuro)
  // hyper parameters
  mu_pr[1:3]  ~ normal(0, 1.0);
  sigma[1:3] ~ normal(0, 0.2);

  // individual parameters w/ Matt trick
  invtemp_pr  ~ normal(0, 1.0);
  alpha_pr  ~ normal(0, 1.0);
  b_pr  ~ normal(0, 1.0);

  for (i in 1:N) {
    vector[4] wv_g;  // action weight for go
    vector[4] wv_ng; // action weight for nogo
    vector[4] qv_g;  // Q value for go
    vector[4] qv_ng; // Q value for nogo
    vector[4] sv;    // stimulus value
    vector[4] pGo;   // prob of go (press)

    wv_g  = initV;
    wv_ng = initV;
    qv_g  = initV;
    qv_ng = initV;
    sv    = initV;

    for (t in 1:Tsubj[i]) {
      wv_g[cue[i, t]]  = qv_g[cue[i, t]] + b[i];
      wv_ng[cue[i, t]] = qv_ng[cue[i, t]];  
      pGo[cue[i, t]]   = inv_logit(invtemp[i] * (wv_g[cue[i, t]] - wv_ng[cue[i, t]]));
      pressed[i, t] ~ bernoulli(pGo[cue[i, t]]);

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
  real<lower=0> mu_invtemp;
  real<lower=0, upper=1> mu_alpha;
  real mu_b;
  real log_lik[N];
  real Qgo[N, T];
  real Qnogo[N, T];
  real Wgo[N, T];
  real Wnogo[N, T];
  real SV[N, T];

  // For posterior predictive check
  real y_pred[N, T];

  // Set all posterior predictions to 0 (avoids NULL values)
  for (i in 1:N) {
    for (t in 1:T) {
      y_pred[i, t] = -1;
    }
  }

  mu_invtemp = exp(mu_pr[1]);
  mu_alpha  = Phi_approx(mu_pr[2]);
  mu_b = mu_pr[3];

  { // local section, this saves time and space
    for (i in 1:N) {
      vector[4] wv_g;  // action weight for go
      vector[4] wv_ng; // action weight for nogo
      vector[4] qv_g;  // Q value for go
      vector[4] qv_ng; // Q value for nogo
      vector[4] sv;    // stimulus value
      vector[4] pGo;   // prob of go (press)

      wv_g  = initV;
      wv_ng = initV;
      qv_g  = initV;
      qv_ng = initV;
      sv    = initV;

      log_lik[i] = 0;

      for (t in 1:Tsubj[i]) {
        wv_g[cue[i, t]]  = qv_g[cue[i, t]] + b[i];
        wv_ng[cue[i, t]] = qv_ng[cue[i, t]];  
        pGo[cue[i, t]]   = inv_logit(invtemp[i] * (wv_g[cue[i, t]] - wv_ng[cue[i, t]]));
        log_lik[i] += bernoulli_lpmf(pressed[i, t] | pGo[cue[i, t]]);

        // generate posterior prediction for current trial
        y_pred[i, t] = bernoulli_rng(pGo[cue[i, t]]);

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
      } // end of t loop
    } // end of i loop
  } // end of local section
}

