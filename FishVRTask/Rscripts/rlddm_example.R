library(hBayesDM)
library(bayestestR)

output1_rlddm = rlddm_PAL1('example', niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE)
output2_rlddm = rlddm_PAL2('example', niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE)
output3a_rlddm = rlddm_PAL3a('example', niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE)
output3b_rlddm = rlddm_PAL3b('example', niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE)
output4a_rlddm = rlddm_PAL4a('example', niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE, modelRegressor = TRUE)
printFit(output1_rlddm, output2_rlddm, output3a_rlddm, output3b_rlddm, output4a_rlddm)

dataPath = "28_subj_hbayesdata.txt"
output1_rlddm_all = rlddm_PAL1(data=dataPath, niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE)
output2b_rlddm_all = rlddm_PAL2b(data=dataPath, niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE)
output3wb_rlddm_all = rlddm_PAL3wb(data=dataPath, niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE)
output4wb_rlddm_all = rlddm_PAL4wb(data=dataPath, niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE, modelRegressor = TRUE)
printFit(output1_rlddm_all, output2b_rlddm_all, output3wb_rlddm_all, output4wb_rlddm_all)
saveRDS(output1_rlddm_all, 'output1_rlddm_all.rds')
saveRDS(output2b_rlddm_all, 'output2b_rlddm_all.rds')
saveRDS(output3wb_rlddm_all, 'output3wb_rlddm_all.rds')
saveRDS(output4wb_rlddm_all, 'output4wb_rlddm_all.rds')

# x=output2b_rlddm_all
# x=output
dim(x$parVals$choice_os)
dim(x$parVals$RT_os)

choice_pred_mean = apply(x$parVals$choice_os, c(2,3), mean)  # average of 4000 MCMC samples
RT_pred_mean = apply(x$parVals$RT_os, c(2,3), mean)  # average of 4000 MCMC samples
# RT_pred_mean = apply(x$parVals$RT_os, c(2,3), map_estimate)  # average of 4000 MCMC samples

numSubjs = dim(x$allIndPars)[1]  # number of subjects

subjList = unique(x$rawdata$subjID)  # list of subject IDs
maxT = max(table(x$rawdata$subjID))  # maximum number of trials
true_choice = array(NA, c(numSubjs, maxT)) # true data (`true_y`)
true_RT = array(NA, c(numSubjs, maxT)) # true data (`true_y`)

## true data for each subject
for (i in 1:numSubjs) {
  tmpID = subjList[i]
  tmpData = subset(x$rawdata, subjID == tmpID)
  true_choice[i, ] = tmpData$keyPressed  # only for data with a 'choice' column
  true_RT[i, ] = tmpData$rt  # only for data with a 'rt' column
  # true_choice[i, ] = tmpData$choice  # only for data with a 'choice' column
  # true_RT[i, ] = tmpData$RT  # only for data with a 'rt' column
}

## Subject #1
plot(true_choice[6, ], type="l", xlab="Trial", ylab="Choice (0 or 1)", yaxt="n")
lines(choice_pred_mean[6,], col="red", lty=2)
axis(side=2, at = c(0,1) )
legend("bottomleft", legend=c("True", "PPC"), col=c("black", "red"), lty=1:2)

## Subject #1
plot(true_RT[6, ], type="l", xlab="Trial", ylab="RT (in seconds)", yaxt="n")
lines(RT_pred_mean[6,], col="red", lty=2)
axis(side=2, at = c(0,1) )
legend("bottomleft", legend=c("True", "PPC"), col=c("black", "red"), lty=1:2)