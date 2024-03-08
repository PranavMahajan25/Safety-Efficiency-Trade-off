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
    // ch = (y <= 0) * 1 + 1;  // Upper boundary choice -> 1, lower boundary choice -> 2
    ch = (y >= a) * 1 + 0;  // Upper boundary choice -> 1, lower boundary choice -> 0
    rt = i * dt + tau;

    ret[1] = ch;
    ret[2] = rt;
    return ret;
  }
}

data {
  int<lower=1> N;                         // Number of subjects
  int<lower=1> T;                         // Maximum number of trials
  int<lower=1, upper=T>  Tsubj[N];        // Number of trials for each subject
  int<lower=1, upper=4> cue[N, T];        // Cues condition  (NA: -1)
  int<lower=-1, upper=1> pressed[N, T];   // Response (NA: -1)
  real outcome[N, T];                     // Feedback
  real RT[N, T];                          // Response time
  real minRT[N];                          // Minimum RT for each subject of the observed data
  real RTbound;                           // Lower bound or RT across all subjects (e.g., 0.1 second)
}

transformed data {
  vector[4] initV;
  initV = rep_vector(0.0, 4);
}

parameters {
  // Group-level raw parameters
  vector[8] mu_pr;
  vector<lower=0>[8] sigma;

  // Subject-level raw parameters (for Matt trick)
  // vector[N] a_pr;         // Boundary separation
  // vector[N] tau_pr;       // Non-decision time
  // vector[N] v_pr;         // Drift rate scaling
  // vector[N] alpha_pr;     // Learning rate

  vector[N] invtemp_pr;     // drift rate scaling, similar to inv temp
  vector[N] alpha_pr;       // learning rate
  vector[N] threshold_pr;   // threshold for DDM
  vector[N] ndt_pr;         // non decision time
  vector[N] b_pr;           // starting point bias
  vector[N] alpha_assoc_pr; // assoc learning rate
  vector[N] kappa_pr;       // scalar multiplier for omega
  vector[N] assoc0_pr;      // initial value of assoc
}

transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=0>[N] threshold;
  vector<lower=RTbound, upper=max(minRT)>[N] ndt;
  vector<lower=0>[N] invtemp;
  vector<lower=0, upper=1>[N] alpha;
  vector[N] b;
  vector<lower=0, upper=1>[N] alpha_assoc;
  vector<lower=0>[N] kappa;
  vector<lower=0>[N] assoc0;

  invtemp = exp(mu_pr[1] + sigma[1] * invtemp_pr); 
  for (i in 1:N) {
    alpha[i] = Phi_approx(mu_pr[2] + sigma[2] * alpha_pr[i]);
    ndt[i]   = Phi_approx(mu_pr[4] + sigma[4] * ndt_pr[i]) * (minRT[i] - RTbound) + RTbound;
    alpha_assoc[i] = Phi_approx(mu_pr[6] + sigma[6] * alpha_assoc_pr[i]);
  }
  threshold = exp(mu_pr[3] + sigma[3] * threshold_pr);
  b = mu_pr[5] + sigma[5] * b_pr;
  kappa = exp(mu_pr[7] + sigma[7] * kappa_pr);
  assoc0 = exp(mu_pr[8] + sigma[8] * assoc0_pr);
  
  
}

model {
  // Group-level raw parameters
  mu_pr ~ normal(0, 1);
  sigma ~ normal(0, 0.2);

  // Individual parameters
  threshold_pr ~ normal(0, 1);
  ndt_pr ~ normal(0, 1);
  invtemp_pr ~ normal(0, 1);
  alpha_pr ~ normal(0, 1);
  b_pr ~ normal(0, 1);
  alpha_assoc_pr  ~ normal(0, 1.0);
  kappa_pr  ~ normal(0, 1.0);
  assoc0_pr  ~ normal(0, 1.0);

  // Subject loop
  for (i in 1:N) {
    vector[4] wv_g;  // action weight for go
    vector[4] wv_ng; // action weight for nogo
    vector[4] qv_g;  // Q value for go
    vector[4] qv_ng; // Q value for nogo
    vector[4] sv;    // stimulus value
    vector[4] drift;   // drift rate (scaled Go minus No Go) for each stimulus

    real omega; // PIT parameter flexible
    real assoc; // assoc value which is updated each trial
    real absRPE; // absRPE each trial

    wv_g  = initV;
    wv_ng = initV;
    qv_g  = initV;
    qv_ng = initV;
    sv    = initV;
    assoc = assoc0[i];


    // Trial loop
    for (t in 1:Tsubj[i]) {

      if (kappa[i]*assoc < 1.0){
        omega = kappa[i]*assoc;
      }
      else {
        omega = 1.0;
      }

      // Save values to variables
      wv_g[cue[i, t]]  = (1-omega) * qv_g[cue[i, t]] + b[i];
      wv_ng[cue[i, t]] = (1-omega) * qv_ng[cue[i, t]] + omega * (-sv[cue[i, t]]); 
      // Drift diffusion process
      drift[cue[i, t]] = invtemp[i] * (wv_g[cue[i, t]] - wv_ng[cue[i, t]]); 
    
      if (pressed[i, t]) { // drift to approach
        RT[i,t] ~ wiener(threshold[i], ndt[i], 0.5, drift[cue[i, t]]); // check weiner; it only gives rt to upper boundary
      } else { // drift to withdrawal
        RT[i,t] ~ wiener(threshold[i], ndt[i], 0.5, -drift[cue[i, t]]); // check weiner; it only gives rt to upper boundary
      }

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
    }
  }
}

generated quantities {
  // For group level parameters
  real<lower=0> mu_invtemp;
  real<lower=0, upper=1> mu_alpha;
  real<lower=0> mu_threshold;
  real<lower=RTbound, upper=max(minRT)> mu_ndt;
  real mu_b;
  real<lower=0, upper=1> mu_alpha_assoc;
  real<lower=0> mu_kappa;
  real<lower=0> mu_assoc0;

  // For log likelihood
  real log_lik[N];

  // For model regressors
  real Qgo[N, T];
  real Qnogo[N, T];
  real Wgo[N, T];
  real Wnogo[N, T];
  real SV[N, T];
  real Assoc_Arr[N, T];
  real Omega_Arr[N, T];

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

  // Assign group-level parameter values
  mu_invtemp    = exp(mu_pr[1]);
  mu_alpha      = Phi_approx(mu_pr[2]);
  mu_threshold  = exp(mu_pr[3]);
  mu_ndt        = Phi_approx(mu_pr[4]) * (mean(minRT) - RTbound) + RTbound;
  mu_b          = mu_pr[5];
  mu_alpha_assoc  = Phi_approx(mu_pr[6]);
  mu_kappa      = exp(mu_pr[7]);
  mu_assoc0     = exp(mu_pr[8]);
  

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
    // Subject loop
    for (i in 1:N) {
      vector[4] wv_g;  // action weight for go
      vector[4] wv_ng; // action weight for nogo
      vector[4] qv_g;  // Q value for go
      vector[4] qv_ng; // Q value for nogo
      vector[4] sv;    // stimulus value
      vector[4] drift;   // drift rate (scaled Go minus No Go) for each stimulus

      real omega; // PIT parameter flexible
      real assoc; // assoc value which is updated each trial
      real absRPE; // absRPE each trial
      
      wv_g  = initV;
      wv_ng = initV;
      qv_g  = initV;
      qv_ng = initV;
      sv    = initV;
      assoc = assoc0[i];

      // Initialized log likelihood
      log_lik[i] = 0;

      // Trial loop
      for (t in 1:Tsubj[i]) {
        if (kappa[i]*assoc < 1.0){
          omega = kappa[i]*assoc;
        }
        else {
          omega = 1.0;
        }

        // Save values to variables
        wv_g[cue[i, t]]  = (1-omega) * qv_g[cue[i, t]] + b[i];
        wv_ng[cue[i, t]] = (1-omega) * qv_ng[cue[i, t]] + omega * (-sv[cue[i, t]]);  

        //////////// Posterior predictive check (one-step method) ////////////
        drift[cue[i, t]] = invtemp[i] * (wv_g[cue[i, t]] - wv_ng[cue[i, t]]);
        // Drift diffusion process
        if (pressed[i, t]) {
          log_lik[i] += wiener_lpdf(RT[i, t] | threshold[i], ndt[i], 0.5, drift[cue[i, t]]);
        } else {
          log_lik[i] += wiener_lpdf(RT[i, t] | threshold[i], ndt[i], 0.5, -drift[cue[i, t]]);
        }
        tmp_os          = wiener_rng(threshold[i], ndt[i], 0.5, drift[cue[i, t]]);
        choice_os[i, t] = tmp_os[1];
        RT_os[i, t]     = tmp_os[2];

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

        // //////////// Posterior predictive check (simulation method) ////////////

        // // Calculate Drift rate
        // d_sm = (Q_sm[s, 1] - Q_sm[s, 2]) * v[i];  // Q[s, 1]: upper boundary option, Q[s, 2]: lower boundary option

        // // Drift diffusion process
        // tmp_sm          = wiener_rng(a[i], tau[i], 0.5, d_sm);
        // choice_sm[i, t] = tmp_sm[1];
        // RT_sm[i, t]     = tmp_sm[2];

        // // Determine feedback
        // rand = uniform_rng(0, 1);
        // if (choice_sm[i, t] == 1) {
        //   fd_sm[i, t] = rand <= prob[s];  // Upper boundary choice (correct)
        // } else {
        //   fd_sm[i, t] = rand > prob[s];   // Lower boundary choice (incorrect)
        // }

        // // Update Q-value
        // r_sm = (choice_sm[i, t] == 2) + 1;  // 'real' to 'int' conversion. 1 -> 1, 2 -> 2
        // Q_sm[s, r_sm] += alpha[i] * (fd_sm[i, t] - Q_sm[s, r_sm]);
      }
    }
  }
}
