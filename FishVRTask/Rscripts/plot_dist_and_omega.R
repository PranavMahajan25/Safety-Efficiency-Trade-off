library(hBayesDM)
library(ggplot2)
library(ggthemes)
library(rstan)
x=output_PAL4wb_all
# x=output4wb_rlddm_all

hist(x$parVals$mu_alpha)
x$fit

plotInd(x)

# stan_plot(x$fit, list("mu_alpha", "mu_invtemp", "mu_b", "mu_alpha_assoc", "mu_kappa", "mu_assoc0"), show_density=T, )
  
plot_obj <- stan_plot(x$fit, "mu_alpha", show_density=T, ci_level=0.8) 
plot_obj +
  theme(axis.text.x=element_text(size=15)) +
  theme(axis.text.y=element_blank(),
        axis.ticks.y=element_blank()) +
  ggtitle(expression(paste("Learning rate (" , alpha ,")")))+
  theme(plot.title=element_text(hjust=0.5))

plot_obj <- stan_plot(x$fit, "mu_invtemp", show_density=T, ci_level=0.8) 
plot_obj +
  theme(axis.text.x=element_text(size=15)) +
  theme(axis.text.y=element_blank(),
        axis.ticks.y=element_blank()) +
  ggtitle(expression(paste("Inverse temperature (" , beta ,")")))+
  theme(plot.title=element_text(hjust=0.5))

plot_obj <- stan_plot(x$fit, "mu_b", show_density=T, ci_level=0.8) 
plot_obj +
  theme(axis.text.x=element_text(size=15)) +
  theme(axis.text.y=element_blank(),
        axis.ticks.y=element_blank()) +
  ggtitle(expression(paste("Baseline bias (" , b ,")")))+
  theme(plot.title=element_text(hjust=0.5))

plot_obj <- stan_plot(x$fit, "mu_alpha_assoc", show_density=T, ci_level=0.8) 
plot_obj +
  theme(axis.text.x=element_text(size=15)) +
  theme(axis.text.y=element_blank(),
        axis.ticks.y=element_blank()) +
  ggtitle(expression(paste("Learning rate multiplier for " , Omega ," (", alpha[Omega],")")))+
  theme(plot.title=element_text(hjust=0.5))

plot_obj <- stan_plot(x$fit, "mu_kappa", show_density=T, ci_level=0.8) 
plot_obj +
  theme(axis.text.x=element_text(size=15)) +
  theme(axis.text.y=element_blank(),
        axis.ticks.y=element_blank()) +
  ggtitle(expression(paste("Scalar multiplier for " , omega , " (",  kappa , ")")))+
  theme(plot.title=element_text(hjust=0.5))

plot_obj <- stan_plot(x$fit, "mu_assoc0", show_density=T, ci_level=0.8) 
plot_obj +
  theme(axis.text.x=element_text(size=15)) +
  theme(axis.text.y=element_blank(),
        axis.ticks.y=element_blank()) +
  ggtitle(expression(paste("Initial associability (" , Omega[0] ,")")))+
  theme(plot.title=element_text(hjust=0.5))

######################################

# x=output_PAL3wb_all
x=output3wb_rlddm_all

# stan_plot(x$fit, list("mu_alpha", "mu_invtemp", "mu_b", "mu_omega"), show_density=T, )

plot_obj <- stan_plot(x$fit, "mu_alpha", show_density=T, ci_level=0.8) 
plot_obj +
  theme(axis.text.x=element_text(size=15)) +
  theme(axis.text.y=element_blank(),
        axis.ticks.y=element_blank()) +
  ggtitle(expression(paste("Learning rate (" , alpha ,")")))+
  theme(plot.title=element_text(hjust=0.5))

plot_obj <- stan_plot(x$fit, "mu_invtemp", show_density=T, ci_level=0.8) 
plot_obj +
  theme(axis.text.x=element_text(size=15)) +
  theme(axis.text.y=element_blank(),
        axis.ticks.y=element_blank()) +
  ggtitle(expression(paste("Inverse temperature (" , beta ,")")))+
  theme(plot.title=element_text(hjust=0.5))

plot_obj <- stan_plot(x$fit, "mu_b", show_density=T, ci_level=0.8) 
plot_obj +
  theme(axis.text.x=element_text(size=15)) +
  theme(axis.text.y=element_blank(),
        axis.ticks.y=element_blank()) +
  ggtitle(expression(paste("Baseline bias (" , b ,")")))+
  theme(plot.title=element_text(hjust=0.5))

plot_obj <- stan_plot(x$fit, "mu_omega", show_density=T, ci_level=0.8) 
plot_obj +
  theme(axis.text.x=element_text(size=15)) +
  theme(axis.text.y=element_blank(),
        axis.ticks.y=element_blank()) +
  ggtitle(expression(paste("Pavlovian bias (" , omega ,")")))+
  theme(plot.title=element_text(hjust=0.5))


##################################################

x=output_PAL4wb_all
# x=output4wb_rlddm_all

plot(x$modelRegressor$Omega_Arr[28,], type="l")

plot(-x$modelRegressor$SV[28,], type="l")

# Generate sample data
data <- x$modelRegressor$Omega_Arr

# Define colors and opacity
colors <- rainbow(28)
opacity <- 0.1

# Plot lines
matplot(t(data), type="l", col=colors, lty=1, lwd=1, , xlab="Trials", ylab=expression(omega))
title(expression(paste("Flexible ", omega)))


########################################################
data_df <- read.csv('data_df.csv', nrows = 240)
cue_list = data_df$cue
cue_list

Vp = -x$modelRegressor$SV[28,][which(cue_list==2)]
w = x$modelRegressor$Omega_Arr[28,][which(cue_list==2)]

plot(-x$modelRegressor$SV[28,][which(cue_list==2)],  type="l")
plot(x$modelRegressor$Omega_Arr[28,][which(cue_list==2)],  type="l")

plot(Vp*w, type="l")

maxT = max(table(x$rawdata$subjID))  # maximum number of trials
bias_cue2 = array(NA, c(numSubjs, maxT/4)) 
for (i in 1:numSubjs) {
  Vp = -x$modelRegressor$SV[i,][which(cue_list==2)]
  w = x$modelRegressor$Omega_Arr[i,][which(cue_list==2)]
  bias_cue2[i,] = Vp*w
}

plot(bias_cue2[28,], type="l")

# Plot lines
matplot(t(bias_cue2), type="l", col=colors, lty=1, lwd=1, , xlab="Trials", ylab=expression(paste( "Vp * ", omega)))
title(expression(paste("Pavlovian bias for Cue 2:  Vp * ", omega )))