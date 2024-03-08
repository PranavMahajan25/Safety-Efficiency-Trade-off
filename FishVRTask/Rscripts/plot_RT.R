output1_rlddm_all <- readRDS('output1_rlddm_all.rds')
output2b_rlddm_all <- readRDS('output2b_rlddm_all.rds')
output3wb_rlddm_all <- readRDS('output3wb_rlddm_all.rds')
output4wb_rlddm_all <- readRDS('output4wb_rlddm_all.rds')

x=output4wb_rlddm_all

dim(x$parVals$choice_os)
dim(x$parVals$RT_os)


plot(density(x$rawdata$rt))
lines(density(x$parVals$RT_os), col='red')

withdrawal_rt_data = x$rawdata$rt[x$rawdata$keyPressed == 0]
approach_rt_data = x$rawdata$rt[x$rawdata$keyPressed == 1]

withdrawal_rt_ppc = x$parVals$RT_os[x$parVals$choice_os == 0]
approach_rt_ppc = x$parVals$RT_os[x$parVals$choice_os == 1]


plot(density(withdrawal_rt_data), xlab = 'Time in seconds', main = "Withdrawal RT distribution (lower bound of the DDM)")
lines(density(withdrawal_rt_ppc), col='red', lty=2)
axis(side=2, at = c(0,1) )
legend("bottomleft", legend=c("True", "PPC"), col=c("black", "red"), lty=1:2)

plot(density(approach_rt_data), xlab = 'Time in seconds', main = "Approach RT distribution (upper bound of the DDM)")
lines(density(approach_rt_ppc), col='red', lty=2)
axis(side=2, at = c(0,1) )
legend("bottomleft", legend=c("True", "PPC"), col=c("black", "red"), lty=1:2)
