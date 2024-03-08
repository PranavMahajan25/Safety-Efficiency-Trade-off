library(hBayesDM)

# dataPath = "./gng_exampleData.txt"
# dataPath = "./guitartmasip_hbayesdata.txt"
# output1_guitartmasip = gng_m1(data=dataPath, niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE)
# output2_guitartmasip = gng_m2(data=dataPath, niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE)
# output3_guitartmasip = gng_m3(data=dataPath, niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE)
# output4_guitartmasip = gng_m4(data=dataPath, niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE)
# printFit(output1_guitartmasip, output2_guitartmasip, output3_guitartmasip, output4_guitartmasip, ic="waic")

dataPath = "subj04-07_hbayesdata.txt"
# dataPath = "21_subj_hbayesdata.txt"
# output1 = gng_m1(data=dataPath, niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE)
# output2 = gng_m2(data=dataPath, niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE)
# output3 = gng_m3(data=dataPath, niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE)
# output4 = gng_m4(data=dataPath, niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE)
# printFit(output1, output2, output3, output4, ic="waic")

dataPath = "28_subj_hbayesdata.txt"
output_PAL1_all = gng_PAL1(data=dataPath, niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE)
output_PAL2b_all = gng_PAL2b(data=dataPath, niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE)
output_PAL2w_all = gng_PAL2w(data=dataPath, niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE)
output_PAL3wb_all = gng_PAL3wb(data=dataPath, niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE)
output_PAL4wb_all = gng_PAL4wb(data=dataPath, niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE, modelRegressor=TRUE)
printFit(output_PAL1_all, output_PAL2b_all, output_PAL2w_all, output_PAL3wb_all, output_PAL4wb_all, ic="looic")
saveRDS(output_PAL1_all, 'output_PAL1_all.rds')
saveRDS(output_PAL2b_all, 'output_PAL2b_all.rds')
saveRDS(output_PAL2w_all, 'output_PAL2w_all.rds')
saveRDS(output_PAL3wb_all, 'output_PAL3wb_all.rds')
saveRDS(output_PAL4wb_all, 'output_PAL4wb_all.rds')

output_PAL5_all = gng_PAL5(data=dataPath, niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE, modelRegressor=TRUE)
printFit(output_PAL5_all, ic="looic")
saveRDS(output_PAL5_all, 'output_PAL5_all.rds')

# output_PAL1 = gng_PAL1(data=dataPath, niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE)
# output_PAL2b = gng_PAL2b(data=dataPath, niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE)
# output_PAL2w = gng_PAL2w(data=dataPath, niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE)
# output_PAL3wb = gng_PAL3wb(data=dataPath, niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE)
# output_PAL4wb = gng_PAL4wb_flexible(data=dataPath, niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE, modelRegressor=TRUE)
# printFit(output_PAL1, output_PAL2b, output_PAL2w, output_PAL3wb, output_PAL4wb, ic="looic")
  
# dataPath = "./pilot_data/gonogo_grpB.txt"
# output1 = gng_3stimuli_fixed(data=dataPath, niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE)
# output2 = gng_3stimuli_flexible(data=dataPath, niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE)
# output3 = gng_m3(data=dataPath, niter=2000, nwarmup=1000, nchain=4, ncore=4, inc_postpred = TRUE)



x=output_PAL4wb_all
dim(x$parVals$y_pred) 

y_pred_mean = apply(x$parVals$y_pred, c(2,3), mean)  # average of 4000 MCMC samples

dim(y_pred_mean)  # y_pred_mean --> 30 (subjects) x 240 (trials)

numSubjs = dim(x$allIndPars)[1]  # number of subjects

subjList = unique(x$rawdata$subjID)  # list of subject IDs
maxT = max(table(x$rawdata$subjID))  # maximum number of trials
true_y = array(NA, c(numSubjs, maxT)) # true data (`true_y`)

## true data for each subject
for (i in 1:numSubjs) {
  tmpID = subjList[i]
  tmpData = subset(x$rawdata, subjID == tmpID)
  true_y[i, ] = tmpData$keyPressed  # only for data with a 'choice' column
}

## Subject #1
plot(true_y[20, ], type="l", xlab="Trial", ylab="Choice (0 or 1)", yaxt="n")
lines(y_pred_mean[20,], col="red", lty=2)
axis(side=2, at = c(0,1) )
legend("bottomleft", legend=c("True", "PPC"), col=c("black", "red"), lty=1:2)