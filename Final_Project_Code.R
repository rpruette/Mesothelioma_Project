library(caret)
library(ggplot2)
library(corrplot)
library(pROC)
library(Hmisc)
library(tidyr)
library(OptimalCutpoints)
library(glmnet)
library(factoextra)
library(cluster)
library(dplyr)


# Read in Data
meso <- read.csv("/Users/rob.pruette/Documents/SMU Spring 2020/STAT 6302/Final Project/MesotheliomaData.csv")
# 34 features, 1 outcome, and 324 observations
dim(meso)
summary(meso)

# There is a correlation of -1 between diagnosis method and the diagnosis outcome
cor(meso$diagnosis.method, meso$class.of.diagnosis)


# Histograms of Skewed Variables
########################################
ggplot(data = meso, aes(x = city)) +
  geom_histogram(bins = 10) + 
  ggtitle("City")

ggplot(data = meso, aes(x = asbestos.exposure)) +
  geom_histogram(bins = 3) +
  ggtitle("Asbestos Exposure")

ggplot(data = meso, aes(x = type.of.MM)) +
  geom_histogram(bins = 4) + ggtitle("Type of MM")

ggplot(data = meso, aes(x = duration.of.asbestos.exposure)) +
  geom_histogram(bins = 20) + ggtitle("Duration of Asbestos Exposure")

ggplot(data = meso, aes(x = cytology)) +
  geom_histogram(bins = 3) + ggtitle("Cytology")

ggplot(data = meso, aes(x = duration.of.symptoms)) +
  geom_histogram(bins = 30) + ggtitle("Duration of Symptoms")

ggplot(data = meso, aes(x = dyspnoea)) +
  geom_histogram(bins = 3) + ggtitle("Dyspnoea")

ggplot(data = meso, aes(x = ache.on.chest)) +
  geom_histogram(bins = 3) + ggtitle("Ache on Chest")

ggplot(data = meso, aes(x = platelet.count..PLT.)) +
  geom_histogram(bins = 30) + ggtitle("Platelet Count PLT")

ggplot(data = meso, aes(x = blood.lactic.dehydrogenise..LDH.)) +
  geom_histogram(bins = 30) + ggtitle("Blood Lactic Dehydrogenise LDH")

ggplot(data = meso, aes(x = alkaline.phosphatise..ALP.)) +
  geom_histogram(bins = 30) + ggtitle("Alkaline Phosphatise ALP")

ggplot(data = meso, aes(x = pleural.lactic.dehydrogenise)) +
  geom_histogram(bins = 30) + ggtitle("Pleural Lactic Dehydrogenise")

ggplot(data = meso, aes(x = dead.or.not)) +
  geom_histogram(bins = 3) + ggtitle("Dead or Alive")

ggplot(data = meso, aes(x = pleural.effusion)) +
  geom_histogram(bins = 3) + ggtitle("Pleural Effusion")

########################################

# Function that makes a nice matrix with all the correlations
flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  =(cormat)[ut],
    p = pmat[ut]
  )
}
# correlations between all variables
res2 <- rcorr(as.matrix(meso))
correlation.matrix <- flattenCorrMatrix(res2$r, res2$P)

# correlations greater than 0.5
correlation.matrix[which(abs(correlation.matrix$cor) > 0.6),]

# correlations with outcome variable
correlation.matrix[which(correlation.matrix$column == "class.of.diagnosis"),]


# Create new outcome variable that is binary 0, 1
meso$diagnosis_label <- ifelse(meso$class.of.diagnosis == 2, "Mesothelioma", "Healthy")

# The following variables are numeric, but they represent factors
# Change the variable to factors for analysis
meso$city <- as.factor(meso$city)
meso$keep.side <- as.factor(meso$keep.side)
meso$habit.of.cigarette <- as.factor(meso$habit.of.cigarette)

meso_dummy <- model.matrix(diagnosis_label ~ ., data = meso)[,-1]
nearZeroVar(meso_dummy, freqCut = 95/5, saveMetrics = FALSE, names = TRUE)
table(meso$city)


# New data set with variables removed
remove.indices <- which(colnames(meso) == "diagnosis.method" | colnames(meso) == "class.of.diagnosis" | colnames(meso) == "type.of.MM")
meso2 <- meso[,-remove.indices]

# bin the city variable so that cities 5, 7, and 8 are one level
table(meso2$city)
meso2$city <- as.numeric(meso2$city) - 1
table(meso2$city)
meso2[which(meso2$city == 5 | meso2$city == 8),]$city <- 7
meso2$city <- factor(meso2$city)
table(meso2$city)

# Make the outcome variable a factor
meso2$diagnosis_label <- as.factor(meso2$diagnosis_label)

# Create a dataset with dummy variables to see what columns have low variance
# City 7 still has low variance, but combining it with another level seems questionable
meso2_dummy <- model.matrix(diagnosis_label ~ ., data = meso2)[,-1]
nearZeroVar(meso2_dummy, freqCut = 95/5, saveMetrics = FALSE, names = TRUE)

# No information rate
no_info_rate <- length(which(meso2$diagnosis_label == "Healthy")) / nrow(meso2)
no_info_rate

############################################################
# Cross Validated Logistic Regression (Model 1)
############################################################
set.seed(256)
logisticRegCV <- train(diagnosis_label ~ ., data = meso2,
                     method = "glm", trControl = trainControl(method = "repeatedcv",
                                                              number = 10,
                                                              repeats = 10, savePredictions = TRUE,
                                                              classProbs = TRUE))
as.data.frame(coef(logisticRegCV$finalModel))
# Results report an accuracy of 0.6945 and a Kappa statistics of 0.2087
model1_accuracy <- logisticRegCV$results$Accuracy
model1_kappa<- logisticRegCV$results$Kappa

# This loop using the confusion matrix from the model output to calculate sensitivity and specificity.
# Accuracy is also calculated and results in the same value as the model results ouput
logisticRegCV_accuracy <- array()
logisticRegCV_sens <- array()
logisticRegCV_spec <- array()
for (i in 1:100){
  logisticRegCV_accuracy[i] <- (logisticRegCV$resampledCM[i,1] + logisticRegCV$resampledCM[i,4]) / sum(logisticRegCV$resampledCM[i,1:4])
  logisticRegCV_sens[i] <- logisticRegCV$resampledCM[i,1] / sum(logisticRegCV$resampledCM[i,c(1,3)])
  logisticRegCV_spec[i] <- logisticRegCV$resampledCM[i,4] / sum(logisticRegCV$resampledCM[i,c(2,4)])
}
# Accuracy
mean(logisticRegCV_accuracy)
# Sensitivity
model1_sensitivity <- mean(logisticRegCV_sens)
# Specificity
model1_specificity <- mean(logisticRegCV_spec)

# Identify an optimal cutpoint
m1_pred_df <- data.frame(logisticRegCV$pred)
head(m1_pred_df)

m1_preds <- data.frame(prob = predict(logisticRegCV, type = "prob")[,2])
m1_preds$pred <- predict(logisticRegCV)
m1_preds$obs <- meso2$diagnosis_label
head(m1_preds)
optcut0 <- summary(optimal.cutpoints(X = "prob", status = "obs", data = m1_preds, 
                                     tag.healthy = "Healthy", methods = "MaxKappa"))
final_cut0 <- optcut0$MaxKappa$Global$optimal.cutoff$cutoff
final_cut0
m1_pred_df$new_pred_label <- as.factor(ifelse(m1_pred_df$Mesothelioma > final_cut0, "Mesothelioma", "Healthy"))

# Create binary variables to use in the calculation of the Brier score
m1_pred_df$brier <- ifelse(m1_pred_df$obs == "Healthy", 0, 1)
head(m1_pred_df)

# Calculate the brier score for model 1
brier_empty <- array()
for (i in 1:nrow(m1_pred_df)){
  brier_empty[i] <- (m1_pred_df$Mesothelioma[i] - m1_pred_df$brier[i])**2
  }
model1_brier <- mean(brier_empty)
model1_brier


# Calculate the accuracy, kappa, sensitivity, and specificity again using the new cutpoint
rep_accuracy0 <- array()
rep_sens0 <- array()
rep_spec0 <- array()
folds0 <- list()
kappa0 <- array()
count0 <- 1
for (j in 1:10){
  for(i in 1:10){
    if(j<10 & i<10){
      locator <- paste("Fold0", as.character(j), ".Rep0", as.character(i), sep = "")
    }else if(j >= 10 & i < 10){
      locator <- paste("Fold", as.character(j), ".Rep0", as.character(i), sep = "")
    }else if(j<10 & i >= 10){
      locator <- paste("Fold0", as.character(j), ".Rep", as.character(i), sep = "")
    }else{
      locator <- paste("Fold", as.character(j), ".Rep", as.character(i), sep = "")
    }
    df <- as.data.frame(m1_pred_df[which(m1_pred_df$Resample == locator), ])
    CM <- confusionMatrix(data = df$new_pred_label, reference = df$obs)
    rep_accuracy0[count0] <- CM$overall["Accuracy"]
    rep_sens0[count0] <- CM$byClass["Sensitivity"]
    rep_spec0[count0] <- CM$byClass["Specificity"]
    kappa0[count0] <- CM$overall["Kappa"]
    count0 = count0 +1
    folds0[count0] <- locator
  }
}
model1.1_accuracy <- mean(rep_accuracy0)
model1.1_sensitivity <- mean(rep_sens0)
model1.1_specificity <- mean(rep_spec0)
model1.1_kappa <- mean(kappa0)


# Using all the predicted data from the cross validation, examine calibration plot and ROC plot

# Calibration plot
calData <- calibration(obs ~ Mesothelioma, data = m1_pred_df, cuts = 10, class = "Mesothelioma")
xyplot(calData, auto.key = list(columns = 2))

# ROC Plot
mesoROC1 <- roc(m1_preds$obs, m1_preds$prob, class = "Mesothelioma")
model1_auc <- auc(mesoROC1)
model1_ci <- ci.auc(mesoROC1)
plot(mesoROC1, legacy.axes = TRUE)



############################################################
# Penalized Logistic Regression, ROC Method (Model 2)
############################################################

glmnGrid <- expand.grid(alpha = seq(0, 1, length = 11),
                        lambda = seq(0.01, 0.2, length = 10))
ctrl <- trainControl(method = "repeatedcv",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     repeats=10,
                     savePredictions = TRUE)
set.seed(546)
glmnFit <- train(x = data.matrix(meso2[, -c(which(colnames(meso2) == "diagnosis_label"))]),
                 y = meso2[, c(which(colnames(meso2) == "diagnosis_label"))],
                 method = "glmnet",
                 tuneGrid = glmnGrid,
                 metric = "ROC",
                 preProc = c("center", "scale"),
                 family = "binomial",
                 trControl = ctrl)
glmnFit$bestTune
# ROC value 0.6102
model2_roc <- mean(glmnFit$resample["ROC"][,1])
# Sensitivity 0.9421
model2_sensitivity <- mean(glmnFit$resample["Sens"][, 1])
# Specificity 0.1629
model2_specificity <- mean(glmnFit$resample["Spec"][, 1])


# The loop below allows me to get the accuracy for the model (even though it isn't necessary)
# the specificity and sensitivity match the model output
rep_accuracy <- array()
rep_sens <- array()
rep_spec <- array()
folds <- list()
count <- 1
for (j in 1:10){
  for(i in 1:10){
    if(j<10 & i<10){
      locator <- paste("Fold0", as.character(j), ".Rep0", as.character(i), sep = "")
    }else if(j >= 10 & i < 10){
      locator <- paste("Fold", as.character(j), ".Rep0", as.character(i), sep = "")
    }else if(j<10 & i >= 10){
      locator <- paste("Fold0", as.character(j), ".Rep", as.character(i), sep = "")
    }else{
      locator <- paste("Fold", as.character(j), ".Rep", as.character(i), sep = "")
    }
    
    df <- as.data.frame(glmnFit$pred[which(glmnFit$pred$alpha == glmnFit$bestTune$alpha & glmnFit$pred$lambda == glmnFit$bestTune$lambda & glmnFit$pred$Resample == locator),])
    CM <- confusionMatrix(data = df$pred, reference = df$obs)
    rep_accuracy[count] <- CM$overall["Accuracy"]
    rep_sens[count] <- CM$byClass["Sensitivity"]
    rep_spec[count] <- CM$byClass["Specificity"]
    count = count +1
    folds[count] <- locator
  }
}
coef(glmnFit$finalModel, s = glmnFit$bestTune$lambda)
model2_accuracy <- mean(rep_accuracy)

mean(rep_sens)
mean(rep_spec)

# Calculate the brier score
brier_data <- data.frame(glmnFit$pred)
brier_data$brier <- ifelse(brier_data$obs == "Healthy", 0, 1)
brier_empty3 <- array()
for (i in 1:nrow(brier_data)){
  brier_empty3[i] <- (brier_data$Mesothelioma[i] - brier_data$brier[i])**2
}
model2_brier <- mean(brier_empty3)

mesoROC2 <- roc(meso2$diagnosis_label, predict(glmnFit, type = "prob")[,2], class = "Mesothelioma")
model2_auc <- auc(mesoROC2)
model2_ci <- ci.auc(mesoROC2)
plot(mesoROC2, legacy.axes = TRUE)


############################################################
# Penalized Logistic Regression, Kappa Method (Model 3)
############################################################

ctrl2 <- trainControl(method = "repeatedcv",
                      classProbs = TRUE,
                      repeats=10,
                      savePredictions = TRUE)
set.seed(344)
glmnFit2 <- train(x = data.matrix(meso2[, -c(which(colnames(meso2) == "diagnosis_label"))]),
                 y = meso2[, c(which(colnames(meso2) == "diagnosis_label"))],
                 method = "glmnet",
                 tuneGrid = glmnGrid,
                 metric = "Kappa",
                 preProc = c("center", "scale"),
                 family = "binomial",
                 trControl = ctrl2)

glmnFit2$bestTune
mean(glmnFit2$resample$Accuracy)

coef(glmnFit2$finalModel, s=glmnFit2$bestTune$lambda)

# Determine the accuracy, sensitivity, specificity, and kappa of the model
rep_accuracy2 <- array()
rep_sens2 <- array()
rep_spec2 <- array()
rep_kappa2 <- array()
folds2 <- list()
count2 <- 1
for (j in 1:10){
  for(i in 1:10){
    if(j<10 & i<10){
      locator <- paste("Fold0", as.character(j), ".Rep0", as.character(i), sep = "")
    }else if(j >= 10 & i < 10){
      locator <- paste("Fold", as.character(j), ".Rep0", as.character(i), sep = "")
    }else if(j<10 & i >= 10){
      locator <- paste("Fold0", as.character(j), ".Rep", as.character(i), sep = "")
    }else{
      locator <- paste("Fold", as.character(j), ".Rep", as.character(i), sep = "")
    }
    
    df <- as.data.frame(glmnFit2$pred[which(glmnFit2$pred$alpha == glmnFit2$bestTune$alpha & glmnFit2$pred$lambda == glmnFit2$bestTune$lambda & glmnFit2$pred$Resample == locator),])
    CM <- confusionMatrix(data = df$pred, reference = df$obs)
    rep_accuracy2[count2] <- CM$overall["Accuracy"]
    rep_kappa2[count2] <- CM$overall["Kappa"]
    rep_sens2[count2] <- CM$byClass["Sensitivity"]
    rep_spec2[count2] <- CM$byClass["Specificity"]
    count2 = count2 +1
    folds2[count2] <- locator
  }
}
# Accuracy (matches the output of the model)
model3_accuracy <- mean(rep_accuracy2)
# Sensitivity
model3_sensitivity <- mean(rep_sens2)
# Specificity
model3_specificity <- mean(rep_spec2)
# Kappa (matches the output of the model)
model3_kappa <- mean(rep_kappa2)


# Find optimal cutpoint
model3_preds <- data.frame(prob = predict(glmnFit2, type = "prob")[,2])
model3_preds$pred <- predict(glmnFit2)
model3_preds$obs <- meso2$diagnosis_label
model3_preds

cut_data <- data.frame(glmnFit2$pred[which(glmnFit2$pred$alpha == glmnFit2$bestTune$alpha & glmnFit2$pred$lambda == glmnFit2$bestTune$lambda),])

optcut1 <- summary(optimal.cutpoints(X = "prob", status = "obs", data = model3_preds, 
                                     tag.healthy = "Healthy", methods = "MaxKappa"))
final_cut1 <- optcut1$MaxKappa$Global$optimal.cutoff$cutoff
final_cut1

# Create new variables for the calculation of the brier statistic
cut_data$new_pred_label <- as.factor(ifelse(cut_data$Mesothelioma > final_cut1, "Mesothelioma", "Healthy"))
cut_data$brier <- ifelse(cut_data$obs == "Healthy", 0, 1)
head(cut_data)

# Calculate brier statistics for both cutpoints
brier_empty4 <- array()
for (i in 1:nrow(cut_data)){
  brier_empty4[i] <- (cut_data$Mesothelioma[i] - cut_data$brier[i])**2
}
model3_brier <- mean(brier_empty4)

# Calculate accuracy, kappa, sensitivity, and specificity using new cutpoint
rep_accuracy3 <- array()
rep_sens3 <- array()
rep_spec3 <- array()
kappa3 <- array()
folds3 <- list()
count3 <- 1
for (j in 1:10){
  for(i in 1:10){
    if(j<10 & i<10){
      locator <- paste("Fold0", as.character(j), ".Rep0", as.character(i), sep = "")
    }else if(j >= 10 & i < 10){
      locator <- paste("Fold", as.character(j), ".Rep0", as.character(i), sep = "")
    }else if(j<10 & i >= 10){
      locator <- paste("Fold0", as.character(j), ".Rep", as.character(i), sep = "")
    }else{
      locator <- paste("Fold", as.character(j), ".Rep", as.character(i), sep = "")
    }
    df <- as.data.frame(cut_data[which(cut_data$Resample == locator), ])
    CM <- confusionMatrix(data = df$new_pred_label, reference = df$obs)
    rep_accuracy3[count3] <- CM$overall["Accuracy"]
    rep_sens3[count3] <- CM$byClass["Sensitivity"]
    rep_spec3[count3] <- CM$byClass["Specificity"]
    kappa3[count3] <- CM$overall["Kappa"]
    count3 <- count3 +1
    folds3[count3] <- locator
  }
}
# Accuracy
model3.1_accuracy <- mean(rep_accuracy3)
# Sensitivity
model3.1_sensitivity <- mean(rep_sens3)
# Specificity
model3.1_specificity <- mean(rep_spec3)
# Kappa
model3.1_kappa <- mean(kappa3)

mesoROC3 <- roc(model3_preds$obs, model3_preds$prob, class = "Mesothelioma")
model3_auc <- auc(mesoROC3)
model3_ci <- ci.auc(mesoROC3)
plot(mesoROC3, legacy.axes = TRUE)

############################################################
# Principal Component Analysis
############################################################

# Numeric variables to be used in PCA
testvars <- meso2[, c(14:15, 17:27, 32)]
colnames(testvars)
testsPCA <- prcomp(testvars, center = T, scale = T)
summary(testsPCA)

# Scree plot
fviz_eig(testsPCA)

# Loadings
testsPCA$rotation

# Graphical representation of components in the first two dimensions
# Not seeing any separation
fviz_pca_ind(testsPCA, label = "none", habillage = meso2$diagnosis_label,
             addEllipses = TRUE, ellipse.level = 0.95, palette = "Dark1", axes = c(2,6))
# Join original variables and the variables from PCA
pca_data <- data.frame(meso2[, -c(14:15, 17:27, 32)], testsPCA$x[, 1:14])


############################################################
# Cross Validated Logistic Regression, with PCA variables (Model 4)
############################################################
set.seed(221)
logisticRegCV_PCA <- train(diagnosis_label ~ ., data = pca_data,
                       method = "glm", trControl = trainControl(method = "repeatedcv",
                                                                number = 10,
                                                                classProbs = TRUE,
                                                                savePredictions = TRUE,
                                                                repeats = 10))
logisticRegCV_PCA$resample
logisticRegCV_PCA$results

cv_pca_df <- data.frame(logisticRegCV_PCA$pred)
head(cv_pca_df)
dim(cv_pca_df)
# Find optimal cutpoint

m4_preds <- data.frame(prob = predict(logisticRegCV_PCA, type = "prob")[,2])
m4_preds$pred <- predict(logisticRegCV_PCA)
m4_preds$obs <- meso2$diagnosis_label
head(m4_preds)


optcut_pca <- summary(optimal.cutpoints(X = "prob", status = "obs", data = m4_preds, 
                                     tag.healthy = "Healthy", methods = "MaxKappa"))
final_cut_pca <- optcut_pca$MaxKappa$Global$optimal.cutoff$cutoff
final_cut_pca
cv_pca_df$new_pred_label <- as.factor(ifelse(cv_pca_df$Mesothelioma > final_cut_pca, "Mesothelioma", "Healthy"))

# Create new variables for brier score calculation
cv_pca_df$brier <- ifelse(cv_pca_df$obs == "Healthy", 0, 1)
head(cv_pca_df)


rep_accuracy_cvPCA <- array()
rep_sens_cvPCA <- array()
rep_spec_cvPCA <- array()
kappa_cvPCA <- array()
count_cvPCA <- 1
for (j in 1:10){
  for(i in 1:10){
    if(j<10 & i<10){
      locator <- paste("Fold0", as.character(j), ".Rep0", as.character(i), sep = "")
    }else if(j >= 10 & i < 10){
      locator <- paste("Fold", as.character(j), ".Rep0", as.character(i), sep = "")
    }else if(j<10 & i >= 10){
      locator <- paste("Fold0", as.character(j), ".Rep", as.character(i), sep = "")
    }else{
      locator <- paste("Fold", as.character(j), ".Rep", as.character(i), sep = "")
    }
    
    df <- as.data.frame(cv_pca_df[which(cv_pca_df$Resample == locator),])
    CM <- confusionMatrix(data = df$pred, reference = df$obs)
    rep_accuracy_cvPCA[count_cvPCA] <- CM$overall["Accuracy"]
    rep_sens_cvPCA[count_cvPCA] <- CM$byClass["Sensitivity"]
    rep_spec_cvPCA[count_cvPCA] <- CM$byClass["Specificity"]
    kappa_cvPCA[count_cvPCA] <- CM$overall["Kappa"]
    count_cvPCA = count_cvPCA +1
  }
}
model4_accuracy <- mean(rep_accuracy_cvPCA)
model4_sensitivity <- mean(rep_sens_cvPCA)
model4_specificity <- mean(rep_spec_cvPCA)
model4_kappa <- mean(kappa_cvPCA)


brier_empty_pca_oldcut <- array()
for (i in 1:nrow(cv_pca_df)){
  brier_empty_pca_oldcut[i] <- (cv_pca_df$Mesothelioma[i] - cv_pca_df$brier[i])**2
}
model4_brier <- mean(brier_empty_pca_oldcut)




rep_accuracy_cvPCA_cut <- array()
rep_sens_cvPCA_cut <- array()
rep_spec_cvPCA_cut <- array()
kappa_cvPCA_cut <- array()
count_cvPCA_cut <- 1
for (j in 1:10){
  for(i in 1:10){
    if(j<10 & i<10){
      locator <- paste("Fold0", as.character(j), ".Rep0", as.character(i), sep = "")
    }else if(j >= 10 & i < 10){
      locator <- paste("Fold", as.character(j), ".Rep0", as.character(i), sep = "")
    }else if(j<10 & i >= 10){
      locator <- paste("Fold0", as.character(j), ".Rep", as.character(i), sep = "")
    }else{
      locator <- paste("Fold", as.character(j), ".Rep", as.character(i), sep = "")
    }
    
    df <- as.data.frame(cv_pca_df[which(cv_pca_df$Resample == locator),])
    CM <- confusionMatrix(data = df$new_pred_label, reference = df$obs)
    rep_accuracy_cvPCA_cut[count_cvPCA_cut] <- CM$overall["Accuracy"]
    rep_sens_cvPCA_cut[count_cvPCA_cut] <- CM$byClass["Sensitivity"]
    rep_spec_cvPCA_cut[count_cvPCA_cut] <- CM$byClass["Specificity"]
    kappa_cvPCA_cut[count_cvPCA_cut] <- CM$overall["Kappa"]
    count_cvPCA_cut = count_cvPCA_cut +1
    }
}
model4.1_accuracy <- mean(rep_accuracy_cvPCA_cut)
model4.1_sensitivity <- mean(rep_sens_cvPCA_cut)
model4.1_specificity <- mean(rep_spec_cvPCA_cut)
model4.1_kappa <- mean(kappa_cvPCA_cut)

mesoROC4 <- roc(m4_preds$obs, m4_preds$prob, class = "Mesothelioma")
model4_auc <- auc(mesoROC4)
model4_ci <- ci.auc(mesoROC4)
plot(mesoROC4, legacy.axes = TRUE)

############################################################
# Penalized Regression, Kappa Metric, with PCA variables (Model 5)
############################################################
set.seed(843)
glmnFit2_PCA <- train(x = data.matrix(pca_data[,-which(colnames(pca_data)== "diagnosis_label"),]),
                  y = pca_data[, which(colnames(pca_data) == "diagnosis_label")],
                  method = "glmnet",
                  tuneGrid = glmnGrid,
                  metric = "Kappa",
                  preProc = c("center", "scale"),
                  family = "binomial",
                  trControl = ctrl2)


glmnFit2_PCA$results
coef(glmnFit2_PCA$finalModel, s=glmnFit2_PCA$bestTune$lambda)
glmnFit2_PCA$finalModel$tuneValue
glmnFit2_PCA$bestTune

# Create data set that has cv predictions with best tuning parameters
pen_pca_data <- data.frame(glmnFit2_PCA$pred[which(glmnFit2_PCA$pred$alpha == glmnFit2_PCA$bestTune$alpha & glmnFit2_PCA$pred$lambda == glmnFit2_PCA$bestTune$lambda),])
head(pen_pca_data)

# Find the accuracy, kappa, sensitivity, and specificity of model
rep_accuracy_pen_pca <- array()
rep_sens_pen_pca <- array()
rep_spec_pen_pca <- array()
kappa_pen_pca <- array()
count_pen_pca <- 1
for (j in 1:10){
  for(i in 1:10){
    if(j<10 & i<10){
      locator <- paste("Fold0", as.character(j), ".Rep0", as.character(i), sep = "")
    }else if(j >= 10 & i < 10){
      locator <- paste("Fold", as.character(j), ".Rep0", as.character(i), sep = "")
    }else if(j<10 & i >= 10){
      locator <- paste("Fold0", as.character(j), ".Rep", as.character(i), sep = "")
    }else{
      locator <- paste("Fold", as.character(j), ".Rep", as.character(i), sep = "")
    }
    
    df <- as.data.frame(pen_pca_data[which(pen_pca_data$Resample == locator),])
    CM <- confusionMatrix(data = df$pred, reference = df$obs)
    rep_accuracy_pen_pca[count_pen_pca] <- CM$overall["Accuracy"]
    rep_sens_pen_pca[count_pen_pca] <- CM$byClass["Sensitivity"]
    rep_spec_pen_pca[count_pen_pca] <- CM$byClass["Specificity"]
    kappa_pen_pca[count_pen_pca] <- CM$overall["Kappa"]
    count_pen_pca = count_pen_pca +1
  }
}
model5_accuracy <- mean(rep_accuracy_pen_pca)
model5_sensitivity <- mean(rep_sens_pen_pca)
model5_specificity <- mean(rep_spec_pen_pca)
model5_kappa <- mean(kappa_pen_pca)


# Find optimal cutpoint
model5_preds <- data.frame(prob = predict(glmnFit2_PCA, type = "prob")[,2])
model5_preds$pred <- predict(glmnFit2_PCA)
model5_preds$obs <- meso2$diagnosis_label
model5_preds


optcut_pen_pca <- summary(optimal.cutpoints(X = "prob", status = "obs", data = model5_preds, 
                                     tag.healthy = "Healthy", methods = "MaxKappa"))
final_cut_pen_pca <- optcut_pen_pca$MaxKappa$Global$optimal.cutoff$cutoff
final_cut_pen_pca

# Create new prediction outcome
pen_pca_data$new_pred_label <- as.factor(ifelse(pen_pca_data$Mesothelioma > final_cut_pen_pca, "Mesothelioma", "Healthy"))

# Create new variables for Brier calculation
pen_pca_data$brier <- ifelse(pen_pca_data$obs == "Healthy", 0, 1)

# Calculate brier scores
brier_empty_pen_pca_old <- array()
for (i in 1:nrow(pen_pca_data)){
  brier_empty_pen_pca_old[i] <- (pen_pca_data$Mesothelioma[i] - pen_pca_data$brier[i])**2
}
model5_brier <- mean(brier_empty_pen_pca_old)

# Accuracy, sensitivity, specificity and kappa for new cutpoint
rep_accuracy_pen_pca_cut <- array()
rep_sens_pen_pca_cut <- array()
rep_spec_pen_pca_cut <- array()
kappa_pen_pca_cut <- array()
count_pen_pca_cut <- 1
for (j in 1:10){
  for(i in 1:10){
    if(j<10 & i<10){
      locator <- paste("Fold0", as.character(j), ".Rep0", as.character(i), sep = "")
    }else if(j >= 10 & i < 10){
      locator <- paste("Fold", as.character(j), ".Rep0", as.character(i), sep = "")
    }else if(j<10 & i >= 10){
      locator <- paste("Fold0", as.character(j), ".Rep", as.character(i), sep = "")
    }else{
      locator <- paste("Fold", as.character(j), ".Rep", as.character(i), sep = "")
    }
    
    df <- as.data.frame(pen_pca_data[which(pen_pca_data$Resample == locator),])
    CM <- confusionMatrix(data = df$new_pred_label, reference = df$obs)
    rep_accuracy_pen_pca_cut[count_pen_pca_cut] <- CM$overall["Accuracy"]
    rep_sens_pen_pca_cut[count_pen_pca_cut] <- CM$byClass["Sensitivity"]
    rep_spec_pen_pca_cut[count_pen_pca_cut] <- CM$byClass["Specificity"]
    kappa_pen_pca_cut[count_pen_pca_cut] <- CM$overall["Kappa"]
    count_pen_pca_cut = count_pen_pca_cut +1
  }
}
model5.1_accuracy <- mean(rep_accuracy_pen_pca_cut)
model5.1_sensitivity <- mean(rep_sens_pen_pca_cut)
model5.1_specificity <- mean(rep_spec_pen_pca_cut)
model5.1_kappa <- mean(kappa_pen_pca_cut)

mesoROC5 <- roc(model5_preds$obs, model5_preds$prob, class = "Mesothelioma")
model5_auc <- auc(mesoROC5)
model5_ci <- ci.auc(mesoROC5)
plot(mesoROC5, legacy.axes = TRUE)


############################################################
# Clustering
############################################################

# PAM
gower.meso <- daisy(meso2[,-33], metric = "gower")
gower.matrix <- as.matrix(gower.meso)

# Most similar patients
meso2[which(gower.matrix == min(gower.matrix[gower.matrix != min(gower.matrix)]), arr.ind = TRUE)[1,],]

# Most dissimilar clients
meso2[which(gower.matrix == max(gower.matrix[gower.matrix != min(gower.matrix)]), arr.ind = TRUE)[1,],]

asw <- numeric(0)
for (k in 1:9){
  asw[k] <- pam(gower.meso, k+1)$silinfo$avg.width
}
k.best <- which.max(asw)
cat("silhouette-optimal number of clusters:", k.best +1, "\n")
plot(2:10, asw, type = "o", main = "pam() Clustering Assessment",
     xlab = "k (# of clusters)", ylab = "average silhouette width")
axis(1, k.best, paste("best", k.best, sep = "\n"), col = "red", col.axis = "red")


k <- 2
pam_fit <- pam(gower.meso, diss = TRUE, k)
pam_results <- meso2 %>%
  mutate(cluster = pam_fit$clustering) %>%
  group_by(cluster) %>%
  do(the_summary = summary(.))
pam_results$the_summary





pam.meso2 <- pam(meso2[, -33], k=2)

##Cluster visualization
fviz_cluster(object = pam.meso2, 
             data=meso2.cluster,
             ellipse.type = "convex",
             palette = "jco",
             geom = "point",
             repel = TRUE,
             ggtheme = theme_bw(),
             axis = c(2,3) )

library(Rtsne)
tsne_obj <- Rtsne(gower.meso, is_distance = TRUE)
tsne_data <- tsne_obj$Y %>%
  data.frame() %>%
  setNames(c("X", "Y")) %>%
  mutate(cluster = factor(pam_fit$clustering))

ggplot(aes(x = X, y = Y), data = tsne_data) +
  geom_point(aes(color = cluster)) +
  labs(title = "Dimension Reduction using Rtsne()", caption = "T-Distributed Stochastic Neighbor Embedding")


meso2_cluster <- data.frame(meso2)
colnames(meso2_cluster)
meso2_cluster$cluster <- pam_fit$clustering

############################################################
# CV Logistic Regression with Cluster Variable (Model 6)
############################################################
set.seed(256)
logisticCV.cluster <- train(diagnosis_label ~ ., data = meso2_cluster,
                       method = "glm", trControl = trainControl(method = "repeatedcv",
                                                                number = 10,
                                                                repeats = 10, 
                                                                savePredictions = TRUE,
                                                                classProbs = TRUE))


model6_kappa <- logisticCV.cluster$results$Kappa



logistic_cluster_accuracy <- array()
logistic_cluster_sens <- array()
logistic_cluster_spec <- array()
for (i in 1:100){
  logistic_cluster_accuracy[i] <- (logisticCV.cluster$resampledCM[i,1] + logisticCV.cluster$resampledCM[i,4]) / sum(logisticCV.cluster$resampledCM[i,1:4])
  logistic_cluster_sens[i] <- logisticCV.cluster$resampledCM[i,1] / sum(logisticCV.cluster$resampledCM[i,c(1,3)])
  logistic_cluster_spec[i] <- logisticCV.cluster$resampledCM[i,4] / sum(logisticCV.cluster$resampledCM[i,c(2,4)])
}
model6_accuracy <- mean(logistic_cluster_accuracy)
model6_sensitivity <- mean(logistic_cluster_sens)
model6_specificity <- mean(logistic_cluster_spec)


cluster.df <- data.frame(logisticCV.cluster$pred)
head(cluster.df)

m6_preds <- data.frame(prob = predict(logisticCV.cluster, type = "prob")[,2])
m6_preds$pred <- predict(logisticCV.cluster)
m6_preds$obs <- meso2$diagnosis_label
head(m6_preds)
optcut.clust <- summary(optimal.cutpoints(X = "prob", status = "obs", data = m6_preds, 
                                     tag.healthy = "Healthy", methods = "MaxKappa"))
final_cut_cluster <- optcut.clust$MaxKappa$Global$optimal.cutoff$cutoff
final_cut_cluster
cluster.df$new_pred_label <- as.factor(ifelse(cluster.df$Mesothelioma > final_cut_cluster, "Mesothelioma", "Healthy"))
cluster.df$brier <- ifelse(cluster.df$obs == "Healthy", 0, 1)
head(cluster.df)

brier_empty_cluster <- array()
for (i in 1:nrow(cluster.df)){
  brier_empty_cluster[i] <- (cluster.df$Mesothelioma[i] - cluster.df$brier[i])**2
}
model6_brier <- mean(brier_empty_cluster)




rep_accuracy_clust <- array()
rep_sens_clust <- array()
rep_spec_clust <- array()
folds_clust <- list()
kappa_clust <- array()
count_clust <- 1
for (j in 1:10){
  for(i in 1:10){
    if(j<10 & i<10){
      locator <- paste("Fold0", as.character(j), ".Rep0", as.character(i), sep = "")
    }else if(j >= 10 & i < 10){
      locator <- paste("Fold", as.character(j), ".Rep0", as.character(i), sep = "")
    }else if(j<10 & i >= 10){
      locator <- paste("Fold0", as.character(j), ".Rep", as.character(i), sep = "")
    }else{
      locator <- paste("Fold", as.character(j), ".Rep", as.character(i), sep = "")
    }
    df <- as.data.frame(cluster.df[which(cluster.df$Resample == locator), ])
    CM <- confusionMatrix(data = df$new_pred_label, reference = df$obs)
    rep_accuracy_clust[count_clust] <- CM$overall["Accuracy"]
    rep_sens_clust[count_clust] <- CM$byClass["Sensitivity"]
    rep_spec_clust[count_clust] <- CM$byClass["Specificity"]
    kappa_clust[count_clust] <- CM$overall["Kappa"]
    count_clust = count_clust +1
    folds_clust[count_clust] <- locator
  }
}
model6.1_accuracy <- mean(rep_accuracy_clust)
model6.1_sensitivity <- mean(rep_sens_clust)
model6.1_specificity <- mean(rep_spec_clust)
model6.1_kappa <- mean(kappa_clust)

mesoROC6 <- roc(m6_preds$obs, m6_preds$prob, class = "Mesothelioma")
model6_auc <- auc(mesoROC6)
model6_ci <- ci.auc(mesoROC6)
plot(mesoROC6, legacy.axes = TRUE)




# Predictions to create calibration plots
model1_pred <- data.frame(predict(logisticRegCV, newdata = meso2[,-33], type = "prob"))
model2_pred <- data.frame(predict(glmnFit, testX = meso2[, -33], type = "prob"))
model3_pred <- data.frame(predict(glmnFit2, testX = meso2[, -33], type = "prob"))
model4_pred <- data.frame(predict(logisticRegCV_PCA, newdata = pca_data[,-which(colnames(pca_data)== "diagnosis_label"),], type = "prob"))
model5_pred <- data.frame(predict(glmnFit2_PCA, testX = pca_data[,-which(colnames(pca_data)== "diagnosis_label"),], type = "prob"))
model6_pred <- data.frame(predict(logisticCV.cluster, newdata = meso2_cluster, type = "prob"))

test_models <- data.frame("CV_Log" = model1_pred$Mesothelioma, "PLR_ROC"= model2_pred$Mesothelioma, "PLR_Kappa"=model3_pred$Mesothelioma, "Diagnosis" = meso2$diagnosis_label)
test_models2 <- data.frame("CV_Log_PCA" = model4_pred$Mesothelioma, "PLR_PCA" = model5_pred$Mesothelioma, "CV_Log_Cluster" = model6_pred$Mesothelioma, "Diagnosis" = meso2$diagnosis_label)

cal.models1 <- calibration(Diagnosis ~ CV_Log + PLR_ROC + PLR_Kappa, data = test_models, cuts = 10, class = "Mesothelioma")
cal.models2 <- calibration(Diagnosis ~ CV_Log_PCA + PLR_PCA + CV_Log_Cluster, data = test_models2, cuts = 10, class = "Mesothelioma")
xyplot(cal.models1, auto.key = list(columns = 3))
xyplot(cal.models2, auto.key = list(columns = 3))



model1_vals <- c(model1_accuracy, model1_sensitivity, model1_specificity, model1_kappa, model1_brier)
model1.1_vals <- c(model1.1_accuracy, model1.1_sensitivity, model1.1_specificity, model1.1_kappa, model1_brier)
model2_vals <- c(model2_accuracy, model2_sensitivity, model2_specificity, NA, model2_brier)
model3_vals <- c(model3_accuracy, model3_sensitivity, model3_specificity, model3_kappa, model3_brier)
model3.1_vals <- c(model3.1_accuracy, model3.1_sensitivity, model3.1_specificity, model3.1_kappa, model3_brier)
model4_vals <- c(model4_accuracy, model4_sensitivity, model4_specificity, model4_kappa, model4_brier)
model4.1_vals <- c(model4.1_accuracy, model4.1_sensitivity, model4.1_specificity, model4.1_kappa, model4_brier)
model5_vals <- c(model5_accuracy, model5_sensitivity, model5_specificity, model5_kappa, model5_brier)
model5.1_vals <- c(model5.1_accuracy, model5.1_sensitivity, model5.1_specificity, model5.1_kappa, model5_brier)
model6_vals <- c(model6_accuracy, model6_sensitivity, model6_specificity, model6_kappa, model6_brier)
model6.1_vals <- c(model6.1_accuracy, model6.1_sensitivity, model6.1_specificity, model6.1_kappa, model6_brier)

final.matrix <- matrix(c(model1_vals, model1.1_vals,  model2_vals, model3_vals, model3.1_vals, model4_vals, model4.1_vals, model5_vals, model5.1_vals, model6_vals, model6.1_vals), ncol = 5, byrow = T)
colnames(final.matrix) <- c("Accuracy", "Sensitivity", "Specificity", "Kappa", "Brier")
rownames(final.matrix) <- c("CV Logistic", "CV Logistic2", "Penalized ROC", "Penalized Logistic", 
                            "Penalized Logistic2", "Logistic PCA", "Logistic PCA2", "Pnlzd Logistic PCA", 
                            "Pnlzd Logistic PCA2", "CV Logistic Cluster", "CV Logistic Cluster2")
final.df <- as.data.frame(final.matrix)
final.df$Model <- rownames(final.matrix)
final.df$OptCutPoint <- c("No", "Yes", "No", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes")
final.df

final.df_OCP <- final.df[c(2,5,7,9,11),]

ggplot(data = final.df[which(final.df$OptCutPoint == "Yes"),], aes(x = reorder(Model, Kappa), y = Kappa))+
  geom_bar(stat = "identity", position = "dodge", fill = "turquoise4") + 
  geom_text(aes(label = round(Kappa, 4)), position = position_dodge(width = 0.2), hjust = -0.25, size = 3) +
  coord_flip() + 
  theme_minimal() + 
  ylim(0,0.3) + 
  xlab("Model") + 
  theme(axis.text.y = element_text(angle = 20, hjust = 1)) +
  theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5)) + 
  labs(title = "Kappa Statistic", subtitle = "Using Optimal Cut Points")

final.df
ggplot(data = final.df[which(final.df$OptCutPoint == "Yes" | final.df$Model == "Penalized ROC"),], aes(x = reorder(Model, Accuracy), y = Accuracy))+
  geom_bar(stat = "identity", position = "dodge") + 
  geom_text(aes(label = round(Accuracy, 4)), position = position_dodge(width = 0.2), hjust = -0.25, size = 3) +
  coord_flip() + 
  theme_minimal() + 
  ylim(0,1) + 
  xlab("Model") + 
  theme(axis.text.y = element_text(angle = 20, hjust = 1)) +
  theme(plot.title = element_text(hjust = 0.5)) + 
  labs(title = "Overall Accuracy")

gather.df <- gather(final.df, key = "statistic", value = "value", Accuracy, Sensitivity, Specificity, Kappa, Brier, OptCutPoint)
gather.df2 <- gather.df[-which(gather.df$statistic == "OptCutPoint" | gather.df$statistic == "Brier"),]
gather.df2 <- gather.df2[-which(gather.df2$Model == "Pnlzd Logistic PCA" | gather.df2$Model == "Penalized Logistic" | gather.df2$Model == "Logistic PCA" | gather.df2$Model == "CV Logistic Cluster" | gather.df2$Model == "CV Logistic"),]

ggplot(gather.df2, aes(fill = statistic, x = Model, y = as.numeric(value)))+
  geom_bar(position = "dodge", stat = "identity")+
  scale_y_continuous() +
  theme(axis.text.y = element_text(angle = 20, hjust = 1)) + 
  coord_flip() + 
  labs(title = "Results", y = "Value")+
  theme(plot.title = element_text(hjust = 0.5))

ggplot(data = final.df[which(final.df$OptCutPoint == "No" | final.df$Model == "Penalized ROC"),], aes(x = reorder(Model, -Brier), y = Brier))+
  geom_bar(stat = "identity", position = "dodge", fill = "turquoise4") + 
  geom_text(aes(label = round(Brier, 4)), position = position_dodge(width = 0.2), hjust = -0.25, size = 3) +
  coord_flip() + 
  theme_minimal() + 
  ylim(0,0.25) + 
  xlab("Model") + 
  theme(axis.text.y = element_text(angle = 20, hjust = 1)) +
  theme(plot.title = element_text(hjust = 0.5)) + 
  labs(title = "Brier Score")


roc_df <- data.frame(ROC = c(model1_auc, model2_auc, model3_auc, model4_auc, model5_auc, model6_auc))
names <- c("CV Logistic","Penalized ROC", "Penalized Logistic", "Logistic PCA","Pnlzd Logistic PCA", "CV Logistic Cluster")
roc_df$Model <- names
roc_df

ggplot(data = roc_df, aes(x = reorder(Model, ROC), y = ROC))+
  geom_bar(stat = "identity", position = "dodge") + 
  geom_text(aes(label = round(ROC, 4)), position = position_dodge(width = 0.2), hjust = -0.25, size = 3) +
  coord_flip() + 
  theme_minimal() + 
  ylim(0,1) + 
  xlab("Model") + 
  theme(axis.text.y = element_text(angle = 20, hjust = 1)) +
  theme(plot.title = element_text(hjust = 0.5)) + 
  labs(title = "AUC Values")

varImp(logisticCV.cluster)
cluster.impo <- varImp(logisticCV.cluster, scale = FALSE)
plot(cluster.impo, top = 10, main = "CV Logistic Regression with Clustering")
