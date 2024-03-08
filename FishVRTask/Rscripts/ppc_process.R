library(ggplot2)
output_PAL1_all <- readRDS('output_PAL1_all.rds')
output_PAL2b_all <- readRDS('output_PAL2b_all.rds')
output_PAL3wb_all <- readRDS('output_PAL3wb_all.rds')
output_PAL4wb_all <- readRDS('output_PAL4wb_all.rds')

confidence_interval <- function(vector, interval) {
  # Standard deviation of sample
  vec_sd <- sd(vector)
  # Sample size
  n <- length(vector)
  # Mean of sample
  vec_mean <- mean(vector)
  # Error according to t distribution
  error <- qt((interval + 1)/2, df = n - 1) * vec_sd / sqrt(n)
  # Confidence interval as a vector
  result <- c("lower" = vec_mean - error, "upper" = vec_mean + error)
  return(result)
}



data_df <- read.csv('data_df.csv', nrows = 240)
cue_list = data_df$cue
cue_list


x=output_PAL4wb_all

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

cue1_true_choices = array(NA, c(numSubjs, maxT/4))
cue2_true_choices = array(NA, c(numSubjs, maxT/4))
cue3_true_choices = array(NA, c(numSubjs, maxT/4))
cue4_true_choices = array(NA, c(numSubjs, maxT/4))
for (i in 1:numSubjs) {
  tmpID = subjList[i]
  cue1_true_choices[i, ] = true_y[tmpID, which(cue_list==1)]
  cue2_true_choices[i, ] = true_y[tmpID, which(cue_list==2)]
  cue3_true_choices[i, ] = true_y[tmpID, which(cue_list==3)]
  cue4_true_choices[i, ] = true_y[tmpID, which(cue_list==4)]
}

average_cue1_true_choices = colMeans(cue1_true_choices)
average_cue2_true_choices = colMeans(cue2_true_choices)
average_cue3_true_choices = colMeans(cue3_true_choices)
average_cue4_true_choices = colMeans(cue4_true_choices)


x=output_PAL1_all
y_pred_mean = apply(x$parVals$y_pred, c(2,3), mean)  # average of 4000 MCMC samples
cue1_pred1_choices = array(NA, c(numSubjs, maxT/4))
cue2_pred1_choices = array(NA, c(numSubjs, maxT/4))
cue3_pred1_choices = array(NA, c(numSubjs, maxT/4))
cue4_pred1_choices = array(NA, c(numSubjs, maxT/4))
for (i in 1:numSubjs) {
  tmpID = subjList[i]
  cue1_pred1_choices[i, ] = y_pred_mean[tmpID, which(cue_list==1)]
  cue2_pred1_choices[i, ] = y_pred_mean[tmpID, which(cue_list==2)]
  cue3_pred1_choices[i, ] = y_pred_mean[tmpID, which(cue_list==3)]
  cue4_pred1_choices[i, ] = y_pred_mean[tmpID, which(cue_list==4)]
}

average_cue1_pred1_choices = colMeans(cue1_pred1_choices)
average_cue2_pred1_choices = colMeans(cue2_pred1_choices)
average_cue3_pred1_choices = colMeans(cue3_pred1_choices)
average_cue4_pred1_choices = colMeans(cue4_pred1_choices)

lower_cue1_pred1_choices = array(NA, c(maxT/4))
lower_cue2_pred1_choices = array(NA, c(maxT/4))
lower_cue3_pred1_choices = array(NA, c(maxT/4))
lower_cue4_pred1_choices = array(NA, c(maxT/4))

upper_cue1_pred1_choices = array(NA, c(maxT/4))
upper_cue2_pred1_choices = array(NA, c(maxT/4))
upper_cue3_pred1_choices = array(NA, c(maxT/4))
upper_cue4_pred1_choices = array(NA, c(maxT/4))

for (i in 1:maxT/4) {
  lower_cue1_pred1_choices[i] = confidence_interval(cue1_pred1_choices[,i], 0.95)[1]
  upper_cue1_pred1_choices[i] = confidence_interval(cue1_pred1_choices[,i], 0.95)[2]
  
  lower_cue2_pred1_choices[i] = confidence_interval(cue2_pred1_choices[,i], 0.95)[1]
  upper_cue2_pred1_choices[i] = confidence_interval(cue2_pred1_choices[,i], 0.95)[2]
  
  lower_cue3_pred1_choices[i] = confidence_interval(cue3_pred1_choices[,i], 0.95)[1]
  upper_cue3_pred1_choices[i] = confidence_interval(cue3_pred1_choices[,i], 0.95)[2]
  
  lower_cue4_pred1_choices[i] = confidence_interval(cue4_pred1_choices[,i], 0.95)[1]
  upper_cue4_pred1_choices[i] = confidence_interval(cue4_pred1_choices[,i], 0.95)[2]
}

x=output_PAL2b_all
y_pred_mean = apply(x$parVals$y_pred, c(2,3), mean)  # average of 4000 MCMC samples
cue1_pred2b_choices = array(NA, c(numSubjs, maxT/4))
cue2_pred2b_choices = array(NA, c(numSubjs, maxT/4))
cue3_pred2b_choices = array(NA, c(numSubjs, maxT/4))
cue4_pred2b_choices = array(NA, c(numSubjs, maxT/4))
for (i in 1:numSubjs) {
  tmpID = subjList[i]
  cue1_pred2b_choices[i, ] = y_pred_mean[tmpID, which(cue_list==1)]
  cue2_pred2b_choices[i, ] = y_pred_mean[tmpID, which(cue_list==2)]
  cue3_pred2b_choices[i, ] = y_pred_mean[tmpID, which(cue_list==3)]
  cue4_pred2b_choices[i, ] = y_pred_mean[tmpID, which(cue_list==4)]
}

average_cue1_pred2b_choices = colMeans(cue1_pred2b_choices)
average_cue2_pred2b_choices = colMeans(cue2_pred2b_choices)
average_cue3_pred2b_choices = colMeans(cue3_pred2b_choices)
average_cue4_pred2b_choices = colMeans(cue4_pred2b_choices)

lower_cue1_pred2b_choices = array(NA, c(maxT/4))
lower_cue2_pred2b_choices = array(NA, c(maxT/4))
lower_cue3_pred2b_choices = array(NA, c(maxT/4))
lower_cue4_pred2b_choices = array(NA, c(maxT/4))

upper_cue1_pred2b_choices = array(NA, c(maxT/4))
upper_cue2_pred2b_choices = array(NA, c(maxT/4))
upper_cue3_pred2b_choices = array(NA, c(maxT/4))
upper_cue4_pred2b_choices = array(NA, c(maxT/4))

for (i in 1:maxT/4) {
  lower_cue1_pred2b_choices[i] = confidence_interval(cue1_pred2b_choices[,i], 0.95)[1]
  upper_cue1_pred2b_choices[i] = confidence_interval(cue1_pred2b_choices[,i], 0.95)[2]
  
  lower_cue2_pred2b_choices[i] = confidence_interval(cue2_pred2b_choices[,i], 0.95)[1]
  upper_cue2_pred2b_choices[i] = confidence_interval(cue2_pred2b_choices[,i], 0.95)[2]
  
  lower_cue3_pred2b_choices[i] = confidence_interval(cue3_pred2b_choices[,i], 0.95)[1]
  upper_cue3_pred2b_choices[i] = confidence_interval(cue3_pred2b_choices[,i], 0.95)[2]
  
  lower_cue4_pred2b_choices[i] = confidence_interval(cue4_pred2b_choices[,i], 0.95)[1]
  upper_cue4_pred2b_choices[i] = confidence_interval(cue4_pred2b_choices[,i], 0.95)[2]
}


x=output_PAL3wb_all
y_pred_mean = apply(x$parVals$y_pred, c(2,3), mean)  # average of 4000 MCMC samples
cue1_pred3wb_choices = array(NA, c(numSubjs, maxT/4))
cue2_pred3wb_choices = array(NA, c(numSubjs, maxT/4))
cue3_pred3wb_choices = array(NA, c(numSubjs, maxT/4))
cue4_pred3wb_choices = array(NA, c(numSubjs, maxT/4))
for (i in 1:numSubjs) {
  tmpID = subjList[i]
  cue1_pred3wb_choices[i, ] = y_pred_mean[tmpID, which(cue_list==1)]
  cue2_pred3wb_choices[i, ] = y_pred_mean[tmpID, which(cue_list==2)]
  cue3_pred3wb_choices[i, ] = y_pred_mean[tmpID, which(cue_list==3)]
  cue4_pred3wb_choices[i, ] = y_pred_mean[tmpID, which(cue_list==4)]
}

average_cue1_pred3wb_choices = colMeans(cue1_pred3wb_choices)
average_cue2_pred3wb_choices = colMeans(cue2_pred3wb_choices)
average_cue3_pred3wb_choices = colMeans(cue3_pred3wb_choices)
average_cue4_pred3wb_choices = colMeans(cue4_pred3wb_choices)

lower_cue1_pred3wb_choices = array(NA, c(maxT/4))
lower_cue2_pred3wb_choices = array(NA, c(maxT/4))
lower_cue3_pred3wb_choices = array(NA, c(maxT/4))
lower_cue4_pred3wb_choices = array(NA, c(maxT/4))

upper_cue1_pred3wb_choices = array(NA, c(maxT/4))
upper_cue2_pred3wb_choices = array(NA, c(maxT/4))
upper_cue3_pred3wb_choices = array(NA, c(maxT/4))
upper_cue4_pred3wb_choices = array(NA, c(maxT/4))

for (i in 1:maxT/4) {
  lower_cue1_pred3wb_choices[i] = confidence_interval(cue1_pred3wb_choices[,i], 0.95)[1]
  upper_cue1_pred3wb_choices[i] = confidence_interval(cue1_pred3wb_choices[,i], 0.95)[2]
  
  lower_cue2_pred3wb_choices[i] = confidence_interval(cue2_pred3wb_choices[,i], 0.95)[1]
  upper_cue2_pred3wb_choices[i] = confidence_interval(cue2_pred3wb_choices[,i], 0.95)[2]
  
  lower_cue3_pred3wb_choices[i] = confidence_interval(cue3_pred3wb_choices[,i], 0.95)[1]
  upper_cue3_pred3wb_choices[i] = confidence_interval(cue3_pred3wb_choices[,i], 0.95)[2]

  lower_cue4_pred3wb_choices[i] = confidence_interval(cue4_pred3wb_choices[,i], 0.95)[1]
  upper_cue4_pred3wb_choices[i] = confidence_interval(cue4_pred3wb_choices[,i], 0.95)[2]
}



x=output_PAL4wb_all
y_pred_mean = apply(x$parVals$y_pred, c(2,3), mean)  # average of 4000 MCMC samples
cue1_pred4wb_choices = array(NA, c(numSubjs, maxT/4))
cue2_pred4wb_choices = array(NA, c(numSubjs, maxT/4))
cue3_pred4wb_choices = array(NA, c(numSubjs, maxT/4))
cue4_pred4wb_choices = array(NA, c(numSubjs, maxT/4))
for (i in 1:numSubjs) {
  tmpID = subjList[i]
  cue1_pred4wb_choices[i, ] = y_pred_mean[tmpID, which(cue_list==1)]
  cue2_pred4wb_choices[i, ] = y_pred_mean[tmpID, which(cue_list==2)]
  cue3_pred4wb_choices[i, ] = y_pred_mean[tmpID, which(cue_list==3)]
  cue4_pred4wb_choices[i, ] = y_pred_mean[tmpID, which(cue_list==4)]
}

average_cue1_pred4wb_choices = colMeans(cue1_pred4wb_choices)
average_cue2_pred4wb_choices = colMeans(cue2_pred4wb_choices)
average_cue3_pred4wb_choices = colMeans(cue3_pred4wb_choices)
average_cue4_pred4wb_choices = colMeans(cue4_pred4wb_choices)

lower_cue1_pred4wb_choices = array(NA, c(maxT/4))
lower_cue2_pred4wb_choices = array(NA, c(maxT/4))
lower_cue3_pred4wb_choices = array(NA, c(maxT/4))
lower_cue4_pred4wb_choices = array(NA, c(maxT/4))

upper_cue1_pred4wb_choices = array(NA, c(maxT/4))
upper_cue2_pred4wb_choices = array(NA, c(maxT/4))
upper_cue3_pred4wb_choices = array(NA, c(maxT/4))
upper_cue4_pred4wb_choices = array(NA, c(maxT/4))

for (i in 1:maxT/4) {
  lower_cue1_pred4wb_choices[i] = confidence_interval(cue1_pred4wb_choices[,i], 0.95)[1]
  upper_cue1_pred4wb_choices[i] = confidence_interval(cue1_pred4wb_choices[,i], 0.95)[2]
  
  lower_cue2_pred4wb_choices[i] = confidence_interval(cue2_pred4wb_choices[,i], 0.95)[1]
  upper_cue2_pred4wb_choices[i] = confidence_interval(cue2_pred4wb_choices[,i], 0.95)[2]
  
  lower_cue3_pred4wb_choices[i] = confidence_interval(cue3_pred4wb_choices[,i], 0.95)[1]
  upper_cue3_pred4wb_choices[i] = confidence_interval(cue3_pred4wb_choices[,i], 0.95)[2]
  
  lower_cue4_pred4wb_choices[i] = confidence_interval(cue4_pred4wb_choices[,i], 0.95)[1]
  upper_cue4_pred4wb_choices[i] = confidence_interval(cue4_pred4wb_choices[,i], 0.95)[2]
}





true_choice_df <- data.frame(X = seq(1,60),
                 Y = average_cue1_true_choices)
model1_df <- data.frame(X = seq(1,60),
                        Y = average_cue1_pred1_choices)
model2_df <- data.frame(X = seq(1,60),
                        Y = average_cue1_pred2b_choices)
model3_df <- data.frame(X = seq(1,60),
                     Y = average_cue1_pred3wb_choices)
model4_df <- data.frame(X = seq(1,60),
                        Y = average_cue1_pred4wb_choices)

ggplot(true_choice_df, aes(X, Y))+
  geom_line(color = "black", size = 2) +
  ylim(0,1) +
  # geom_line(data = model1_df, color = "blue", size = 1) +
  geom_ribbon(aes(ymin=lower_cue1_pred1_choices, ymax=upper_cue1_pred1_choices), alpha=0.5, fill = "blue", 
              color = "blue", linetype = "dotted") +
  # geom_line(data = model2_df, color = "violet", size = 1) +
  geom_ribbon(aes(ymin=lower_cue1_pred2b_choices, ymax=upper_cue1_pred2b_choices), alpha=0.5, fill = "violet", 
              color = "violet", linetype = "dotted") +
  # geom_line(data = model3_df, color = "red", size = 1) +
  geom_ribbon(aes(ymin=lower_cue1_pred3wb_choices, ymax=upper_cue1_pred3wb_choices), alpha=0.5, fill = "red", 
              color = "red", linetype = "dotted") +
  # geom_line(data = model4_df, color = "green", size = 1) +
  geom_ribbon(aes(ymin=lower_cue1_pred4wb_choices, ymax=upper_cue1_pred4wb_choices), alpha=0.5, fill = "green", 
              color = "green", linetype = "dotted") 



plot(average_cue1_true_choices, type="l", col="black", ylim=c(0,1), lwd=2, xlab="Cue trial", ylab= "Average choice (Cue 1)")
# lines(average_cue1_pred1_choices, col="blue", ylim=c(0,1), lwd=2)
# lines(average_cue1_pred2b_choices, col="violet", ylim=c(0,1), lwd=2)
# lines(average_cue1_pred3wb_choices, col="red", ylim=c(0,1), lwd=2)
lines(average_cue1_pred4wb_choices, col="red", ylim=c(0,1), lwd=2, lty=2)
legend("bottomleft", legend=c("True", "PPC"), col=c("black", "red"), lty=1:2)


plot(average_cue2_true_choices, type="l", col="black", ylim=c(0,1), lwd=2, xlab="Cue trial", ylab= "Average choice (Cue 2)")
lines(average_cue2_pred4wb_choices, col="red", ylim=c(0,1), lwd=2, lty=2)
legend("bottomleft", legend=c("True", "PPC"), col=c("black", "red"), lty=1:2)


plot(average_cue3_true_choices, type="l", col="black", ylim=c(0,1), lwd=2,  xlab="Cue trial", ylab= "Average choice (Cue 3)")
lines(average_cue3_pred4wb_choices, col="red", ylim=c(0,1), lwd=2, lty=2)
legend("bottomleft", legend=c("True", "PPC"), col=c("black", "red"), lty=1:2)


plot(average_cue4_true_choices, type="l", col="black", ylim=c(0,1), lwd=2, xlab="Cue trial", ylab= "Average choice (Cue 4)")
lines(average_cue4_pred4wb_choices, col="red", ylim=c(0,1), lwd=2, lty=2)
legend("bottomleft", legend=c("True", "PPC"), col=c("black", "red"), lty=1:2)




