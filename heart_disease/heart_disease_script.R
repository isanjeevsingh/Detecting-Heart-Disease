# References 
# 1. PR Curve and ROC: http://pages.cs.wisc.edu/~jdavis/davisgoadrichcamera2.pdf
# 2. Data Pre-processing:   http://topepo.github.io/caret/data-splitting.html
# 3. Data Source: https://archive.ics.uci.edu/ml/datasets/Statlog+%28Heart%29
# 4. ROCR Package: https://www.analyticsvidhya.com/blog/2015/11/beginners-guide-on-logistic-regression-in-r/
# 5. PRROC Package: https://cran.r-project.org/web/packages/PRROC/vignettes/PRROC.pdf
# 6. Correlation analysis: https://www.kaggleusercontent.com/kf/4266451/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..5FDp6QpjeHWrz8-8Yj-MhA.gfM5xOslX8-vtzgW9HqDZfmUVOQc242qe-HdjedXb24oJ_mWfSBaAJkNqis2ImYRG4kkTTqZQWxpfEHF5UFDt_jdY2_2F3JE87-GgwQfYmmHhfaKguK6CbJ8gxPF3Bcbjs2vNU_Sgkotu3sTYyJhHMkEoIQ6Hlhw6ivzVSSrWeYzHsuJJEoL_7ydwYgedcrioWNBWHePGJFYu4zoxLnIRuSa5OQ0lJP7OLisbfVppC9Bsu0tx15DTpkiFP0RcZMiQtfTLjSi2xYbKweJsbnMsCr_vWA2oIYfO4KsSLLz31XMFxqCeaLDSs5UuSHUShp8Gx8aGtYk3DP6fzDDZ2wd_chUyjqtpbMw4xhcrnFEvQW6pniCgjNyKgQbJunhWBCgZsUCc6UyKkF1yrCLZJVXPQ_es0552oCHT2tIfpPnOzyJk0XNq_1JCo_ybffemgdn5GoXmQoUIHdrOalvz5tep99vqAywR8V4nFqX_P8QSfzYrbpnFp-GeJd5btlP2-B5ldhUOCQiOztuft6T1EP4xXdOh7b6IqZsAUEeEnnmGig3YMcHjpyQoIQe9zagwUNl7YpKYyd6EiHdRahWAdYPa1NsIf30R8PepqFw9PTfS0KoOMwkku2q5vN7wrruafoFdfb1TIkEbqCnUaJ-xRkA-CNOi_mtNpK6LJu5n6U9ycY.CKO_Bq9Sz_A5OW_Z6-ylRA/__results__.html#
# 7. Ordinal Variable: https://stackoverflow.com/questions/41943789/how-does-r-handle-ordinal-predictors-in-lm
# 8. Categorical Variable Encoding: https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02
# 9. Similar work: https://towardsdatascience.com/heart-disease-prediction-73468d630cfc
# 10. Literature on heart disease: https://en.wikipedia.org/wiki/Cardiovascular_disease
# 11. ggpubr: http://www.sthda.com/english/articles/24-ggpubr-publication-ready-plots/81-ggplot2-easy-way-to-mix-multiple-graphs-on-the-same-page/
# 12. ggroc: https://rdrr.io/cran/pROC/man/ggroc.html
set.seed(22)

## LOAD LIBRARIES
library("caret")
library("corrplot")
library("ggplot2")
library("pROC")
library("PRROC")
library("ROCR")
library("xgboost")
library("rjags")
library("dplyr")
library("coda")
library("knitr")
library("kableExtra")
library("ggpubr")
library("gridExtra")
library("grid")"

## HELPER FUNCTIONS

# ROC and PRC
calc_rocprc <- function(prob, label){
  index_class2 <- label == 1
  index_class1 <- label == 0
  pr <- pr.curve(prob[index_class2], prob[index_class1], curve = TRUE)
  roc <- roc.curve(prob[index_class2], prob[index_class1], curve = TRUE)
  print(pr); print(roc); 
  plot(pr) 
  plot(roc)
}

# Function to calculate pairwise correlation [7]
getPairCorrelation <- function(corrMatrix){
  featureName <- colnames(corrMatrix)
  nFeature <- length(featureName)
  
  # set lower triangle of matrix to NA (these values are all redundant)
  corrMatrix[lower.tri(corrMatrix, diag = TRUE)] <- NA
  
  # convert matrix to data frame
  featurePair <- data.frame(feature1 = rep(featureName, nFeature), feature2 = rep(featureName, each = nFeature), coef = as.vector(corrMatrix))
  # remove NAs
  featurePair <- featurePair[!is.na(featurePair$coef), ]
  # calculate absolute value of correlation coefficient
  featurePair$coefAbs <- abs(featurePair$coef)
  # order by coefficient
  featurePair <- featurePair[order(featurePair$coefAbs, decreasing = TRUE), ]
  
  featurePair
} 

## LOAD DATA
columns=c( "age", "sex", "angina", "resting_bp", "ser_chol", "BSgt120"
           ,"restecg", "max_hr", "exang", "oldpeak", "slope", "vessels", "thal", "y")
setwd("~/Documents/Statistics/classification/heart_disease")
dat = read.csv(file="heart.dat", sep=" ", col.names = columns, header=FALSE)
nrow(dat) # Test number of rows


# Assign factors to the data
dat$sex <- factor(dat$sex)
levels(dat$sex) <- c("female", "male")

dat$angina <- factor(dat$angina)
levels(dat$angina)=c("typical","atypical","nonang","asymp")

dat$BSgt120 <- factor(dat$BSgt120)
levels(dat$BSgt120) <- c("false", "true")

dat$restecg <- factor(dat$restecg)
levels(dat$restecg ) <- c("normal","stt","hypertrophy")

dat$exang <- factor(dat$exang)
levels(dat$exang) <- c("no","yes")

dat$slope <- factor(dat$slope, order = TRUE)
levels(dat$slope)=c("upsloping","flat","downslop")

dat$thal <- factor(dat$thal)
levels(dat$thal)=c("normal","fixed","reversable")

dat$y <- ifelse(dat$y==1, 0, 1)

summary(dat)

# PREPARE TRAIN AND TEST
# Balanced 74% split. list = False avoids returning data as a list
train_ind <- createDataPartition(dat$y, p = .75, list = FALSE, times = 1) 
train.X <- dat[train_ind, ][,-ncol(dat)]
train.y <- dat[train_ind, ]$y
test.X  <- dat[-train_ind, ][,-ncol(dat)]
test.y  <- dat[-train_ind, ]$y

# validate the split
round(table(train.y)/sum(table(train.y))*100, 1) # 44.3% Yes
round(table(test.y)/sum(table(test.y))*100, 1) # 44.8% Depositors
nrow(train.X); length(train.y); nrow(test.X); length(test.y) # Check the split data

# DATA PREPROCESSING

# Center data
preProcValues <- preProcess(train.X, method = c("center", "scale"))
train.X.T <- predict(preProcValues, train.X)
test.X.T <- predict(preProcValues, test.X)

# Test the scaled data
numeric_cols <- c("age", "resting_bp" , "ser_chol", "max_hr", "oldpeak")
colMeans(train.X.T[numeric_cols]) # test column means are zero
apply(train.X.T[numeric_cols], 2, sd) # test column sd is one
colMeans(test.X.T[numeric_cols]) # test column means are zero
apply(test.X.T[numeric_cols], 2, sd) # test column sd is one

# Fit a baseline model and extract the design matrix without the intercept column
glm.fit = glm(train.y ~ . + resting_bp * sex 
              ,data=train.X.T, family="binomial")
train.design.matrix = model.matrix(glm.fit)[,-1] # Get the design matrix
test.design.matrix <- model.matrix(test.y ~ .+ resting_bp * sex, data=test.X.T, family="binomial")[,-1]

# PERFORM EDA

# List of covariates 
names(as.data.frame(train.design.matrix)) # 51 c


#1. Correlation: Response vs. Covariates
covariates <- names(as.data.frame(train.design.matrix)) 
corr <- data.frame(covariate = covariates, coef = rep(NA, length(covariates)))
for (icov in 1:length(covariates)){
  corr$coef[icov] <- cor(train.design.matrix[, icov], train.y)
  print(covariates[icov]);print(corr$coef[icov])
}

corr.order <- corr[order(corr$coef, decreasing = FALSE), ]

ggplot(corr.order, aes(x = factor(covariate, levels = covariate), y = coef)) + 
  geom_bar(stat = "identity") + 
  coord_flip() + 
  xlab("Feature") + 
  ylab("Correlation Coefficient")


#2. Correlation: Between Covariates

# calculate correlation matrix
corrMatrix <- cor(train.design.matrix)
corrplot(corrMatrix, method = "color", type = "upper") # plot matrix

# Pairwise correlation table
pairCorr <- getPairCorrelation(corrMatrix)    
kable(pairCorr[1:10, ]) %>% kable_styling(full_width = FALSE) #using kable_styling

# Correlation distribution
ggplot(pairCorr, aes(coef)) + geom_histogram(binwidth = 0.1) + xlab("Correlation Coefficient")
boxplot(pairCorr$coef)


# Get correlation of numeric data that is centered
Cor=cor(train.design.matrix)
corrplot(Cor, type="upper", method="ellipse")
corrplot(Cor, type="lower", method="number", col="black",
         add=TRUE, diag=FALSE, tl.pos="n", cl.pos="n")

# MODELLING


# 1. BASELINE MODEL
summary(glm.fit)
plot(residuals(glm.fit, type="deviance"))
plot(fitted(glm.fit), train.y)

# Training Data Results
glm.prob.train = predict(glm.fit, type="response")
plot(glm.prob.train, jitter(train.y)) # Plot of prob w.r.t. labels
100*(1-mean(as.numeric(glm.prob.train > 0.29) != train.y)) # Train Accuracy
table(train.y, glm.prob.train > 0.29) # confusion matrix
calc_rocprc(glm.prob.train, train.y) # ROC & AUC

# Test Data Results
glm.prob.test = predict(glm.fit, newdata=test.X.T, type="response")
plot(glm.prob.test, jitter(test.y)) # Plot of prob w.r.t. labels
100*(1-mean(as.numeric(glm.prob.test > 0.30) != test.y)) # Test Accuracy
table(test.y, glm.prob.test > 0.30) # confusion matrix
calc_rocprc(glm.prob.test, test.y) # ROC & AUC

100*(1-mean(as.numeric(glm.prob.test > 0.2) != test.y)) # Test Accuracy
table(test.y, glm.prob.test > 0.2) # confusion matrix

# 2. FIT BAYESIAN MODEL THROUGH JAGS

mod_string = " model {
    for (i in 1:length(y)) {
        y[i] ~ dbern(p[i])
        logit(p[i]) = int + b[1]*age[i] + b[2]*sexmale[i] + b[3]*anginaatypical[i] 
                          + b[4]*anginanonang[i] + b[5]*anginaasymp[i] 
                          + b[6]*resting_bp[i]       
                          + b[7]*ser_chol[i] + b[8]*BSgt120true[i] + b[9]*restecgstt[i]
                          + b[10]*restecghypertrophy[i] + b[11]*max_hr[i] + b[12]*exangyes[i] 
                          + b[13]*oldpeak[i] + b[14]*slope.L[i] + b[15]*slope.Q[i]
                          + b[16]*vessels[i] + b[17]*thalfixed[i] + b[18]*thalreversible[i]
                          + b[19]*sexmale_resting_bp[i]  
    }
    
    int ~ dnorm(0.0, 1.0/25.0)
    for (j in 1:20) {
        b[j] ~ ddexp(0.0, sqrt(2.0)) # has variance 1.0
    }
}"

train.jags <- cbind(train.design.matrix , y=train.y)
train.jags.df <- as.data.frame(train.jags)
train.jags.df <- train.jags.df %>% rename(  "sexmale_resting_bp" = "sexmale:resting_bp") # using dplyr
train.jags.list = as.list(train.jags.df)
params = c("int", "b")
mcmc.model = jags.model(textConnection(mod_string), data=train.jags.list, n.chains=3)
update(mcmc.model, 1e3)
mcmc.model.sim = coda.samples(model=mcmc.model,
                              variable.names=params,
                              n.iter=5e3)
mcmc.model.csim = as.mcmc(do.call(rbind, mcmc.model.sim))

# Convergence diagnostics
plot(mcmc.model.sim, ask=TRUE) # Remove b4, b8, b9 , b14, b19, b6
gelman.diag(mcmc.model.sim)
autocorr.diag(mcmc.model.sim)
autocorr.plot(mcmc.model.sim)
effectiveSize(mcmc.model.sim)
HPDinterval(mcmc.model.sim)

# calculate DIC
dic1 = dic.samples(mcmc.model, n.iter=1e3)
summary(mcmc.model.sim)


# Select top 20 features
(pm_coef = colMeans(mcmc.model.csim))
sort(abs(pm_coef), decreasing = TRUE)
plot(sort(abs(pm_coef[1:51]), decreasing = TRUE), type="l")
barplot(sort(abs(pm_coef[1:51]), decreasing = TRUE))

# Prediction 

# Training
pm_Xb = pm_coef["int"] + train.design.matrix %*% pm_coef[1:20]
phat = 1.0 / (1.0 + exp(-pm_Xb))
head(phat)
plot(phat, jitter(train.y))
(tab0.28 = table(phat > 0.28, train.y))
sum(diag(tab0.28)) / sum(tab0.28)

(tab0.2 = table(phat > 0.2, train.y))
sum(diag(tab0.2)) / sum(tab0.2)

calc_rocprc(phat, train.y) # ROC & AUC

# Test
pm_Xb = pm_coef["int"] + test.design.matrix %*% pm_coef[1:20]
phat = 1.0 / (1.0 + exp(-pm_Xb))
head(phat)
plot(phat, jitter(test.y))
(tab0.28 = table(phat > 0.28, test.y))
sum(diag(tab0.28)) / sum(tab0.28)

(tab0.2 = table(phat > 0.2, test.y))
sum(diag(tab0.2)) / sum(tab0.2)

calc_rocprc(phat, test.y) # ROC & AUC

# Remove covariates based on the posterior estimates 
rm_cov = c( "chest_painnon_anginal", "BSgt120true", "ecgstt", "slope.L","thalfixed", "resting_bp")
# FIT Reduced model
reduced.mod.string = " model {
    for (i in 1:length(y)) {
        y[i] ~ dbern(p[i])
        logit(p[i]) = int + b[1]*age[i] 
                          + b[2]*sexmale[i] 
                          + b[3]*chest_painatypical[i] 
                          + b[4]*chest_painasymptomatic[i]      
                          + b[5]*cholestoral[i] 
                          + b[6]*ecghypertrophy[i] 
                          + b[7]*heart_rate[i] 
                          + b[8]*anginayes[i] 
                          + b[9]*oldpeak[i] 
                          + b[10]*slope.Q[i]
                          + b[11]*vessels1[i] 
                          + b[12]*vessels2[i] 
                          + b[13]*vessels3[i]
                          + b[14]*thalreversable[i]    
    }
    
    int ~ dnorm(0.0, 1.0/25.0)
    for (j in 1:14) {
        b[j] ~ dnorm(0.0, 1.0/25.0) # noninformative for logistic regression
    }
} "

params = c("int", "b")
reduced.train.design.matrix=train.design.matrix[,!colnames(train.design.matrix) %in% rm_cov ]
reduced.train.data  = cbind(reduced.train.design.matrix , y=train.y)
reduced.train.jags.df <- as.data.frame(reduced.train.data)
reduced.train.jags.list = as.list(reduced.train.jags.df)
reduced.mcmc.model = jags.model(textConnection(reduced.mod.string), data=reduced.train.jags.list, n.chains=3)
update(reduced.mcmc.model, 1e3)
reduced.mcmc.model.sim = coda.samples(model=reduced.mcmc.model,
                                      variable.names=params,
                                      n.iter=10e3)
reduced.mcmc.model.csim = as.mcmc(do.call(rbind, reduced.mcmc.model.sim))
plot(reduced.mcmc.model.sim, ask=TRUE)
gelman.diag(reduced.mcmc.model.sim)
autocorr.diag(reduced.mcmc.model.sim)
autocorr.plot(reduced.mcmc.model.sim)
effectiveSize(reduced.mcmc.model.sim)
summary(reduced.mcmc.model.sim)
HPDinterval(reduced.mcmc.model.csim)

# Prediction 
(pm_coef = colMeans(reduced.mcmc.model.csim))
pm_Xb = pm_coef["int"] + reduced.train.design.matrix %*% pm_coef[1:14]
phat = 1.0 / (1.0 + exp(-pm_Xb))
head(phat)
plot(phat, jitter(train.y))
(tab0.5 = table(phat > 0.5, train.y))
sum(diag(tab0.5)) / sum(tab0.5)

(tab0.2 = table(phat > 0.2, train.y))
sum(diag(tab0.2)) / sum(tab0.2)

calc_rocprc(phat, train.y) # ROC & AUC


# APPENDIX


# Attributes types
# -----------------
# Real: 1,4,5,8,10,12
# Ordered:11,
# Binary: 2,6,9
# Nominal:7,3,13
# 
# Variable to be predicted
# ------------------------
# Absence (1) or presence (2) of heart disease 