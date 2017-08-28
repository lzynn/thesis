install.packages("glmnet")
install.packages("flare")

setwd("F:/master/OnlineNewsPopularity")
#setwd("/Volumes/CLASS/master/OnlineNewsPopularity")
#setwd("/Mathematics/Graduates/ziyi.liu/Downloads/master/")
data <- read.csv("OnlineNewsPopularity.csv") #url, timedelta are non-predictive
summary(data) #something data point are wrong

sum(data$n_unique_tokens > 1)
which.max(data$n_unique_tokens) # data$n_unique_tokens[31038] = 701, we can't get a rate like that. So as this point other two rates, remove!
data <- data[-31038, ]

#check data channel
#channelthing <- data[,14:19]
#chansum <- rowSums(channelthing)
#sum(chansum == 1)
#sum(chansum == 0)
#sum(chansum > 1)  # no article in mutiple channel, some are in no channel

#channelthing$data_channel_is_none = chansum == 0
#channel <- factor(apply(channelthing, 1, function(x) which(x == 1)), 
#                  labels = colnames(channelthing)) 

# check the weekday
#weekdaything <- data[,32:38]
#weekday <- factor(apply(weekdaything, 1, function(x) which(x == 1)), 
#                  labels = colnames(weekdaything)) 

#data <- data[,-c(1, 14:19, 32:38)]
#data$weekday <- weekday
#data$channel <- channel

for(i in 2:61)
{
  f_name <- names(data)[i]
  filename <- paste0('feature/', f_name, '.png')
  png(file = filename, width = 480, height = 480, units = "px")
  hist(data[,i], xlab = f_name, main = f_name)
  dev.off()
}

no_norm <- data[, c(2, 14:19, 32:39)]
#no_norm[, -1] <- lapply(no_norm[, -1], factor)
be_norm <- data[, -c(1, 2, 14:19, 32:39)]
mean_norm <- colMeans(be_norm)
sd_norm <- sqrt(colSums((t(t(be_norm) - mean_norm))^2)/39642)
after_norm <- t((t(be_norm)-mean_norm)/sd_norm)

data1 <- cbind.data.frame(no_norm, after_norm)

# Something from orginal paper
# e1071 for svm
# randomForest for rf

D1 <- 1400
popular <- as.numeric(data$shares >= D1)
png(file = 'feature/popular.png', width = 480, height = 480, units = "px")
hist(data$popular, xlab = 'popular', main = 'popular')
dev.off()
train <- data[1:9000,c(3:60,62)]
test <- data[9001:10000,c(3:60,62)]

library(e1071)
m1l <- svm(formula = popular ~ ., data = train, kernel = 'radial', type = 'C-classification')
m2l <- svm(formula = popular ~ ., data = train, kernel = 'linear', type = 'C-classification', cost = 2)

pred1 <- predict(m1l, test)
mean((as.integer(pred1)-1-as.integer(test[,59]))^2)

pred2 <- predict(m2l, test)
mean((as.integer(pred2)-1-as.integer(test[,59]))^2)

m3l <- svm(formula = popular ~ ., data = train, kernel = 'linear', type = 'C-classification', cost = 2^2)
pred3 <- predict(m3l, test)
mean((as.integer(pred3)-1-as.integer(test[,59]))^2)

m4l <- svm(formula = popular ~ ., data = train, kernel = 'linear', type = 'C-classification', cost = 2^3)
pred4 <- predict(m4l, test)
mean((as.integer(pred4)-1-as.integer(test[,59]))^2)

svm_tune <- tune.svm(popular ~ ., data = train, kernel = "radial", cost=2^(0:6))


# My work
# linear

linearm <- lm(shares ~ ., data = data1)
step(linearm)
step(linearm, k = log(39643))
# after BIC step, it goes to lm(formula = shares ~ n_tokens_title + num_hrefs + average_token_length + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + global_subjectivity + avg_negative_polarity + timedelta + data_channel_is_entertainment, data = data1)

data2 <- data1[, c(1, 3, 16, 21, 25, 33:36, 44, 53, 60)]

cv_linear <- function(data, form, cv_num, seed = 250)
{
  set.seed(seed)
  n <- dim(data)[1]
  cv_cut <- sample(1:n)
  errors <- rep(0, cv_num)
  for(i in 1:cv_num)
  {
    test <- data[cv_cut[round((i - 1)/cv_num*n + 1):round(i/cv_num*n)],]
    train <- data[-cv_cut[round((i - 1)/cv_num*n + 1):round(i/cv_num*n)],]
    cv_lm <- lm(form, data = train)
    pred <- predict(cv_lm, test)
    errors[i] <- mean((test$shares - pred)^2)
  }
  return(mean(errors))
}

cv_linear(data = data2, form = shares ~ ., cv_num = 10) # error: 0.9803159

matrix_data <- as.matrix(data2)
library(glmnet)
cv_lasso <- cv.glmnet(matrix_data[, -12], matrix_data[, 12], alpha = 1)
plot(cv_lasso)
mean(cv_lasso$cvm) #0.9839963

cv_lasso <- function(data, form, cv_num, seed = 250)
{
  set.seed(seed)
  n <- dim(data)[1]
  cv_cut <- sample(1:n)
  errors <- rep(0, cv_num)
  for(i in 1:cv_num)
  {
    test <- as.matrix(data[cv_cut[round((i - 1)/cv_num*n + 1):round(i/cv_num*n)],])
    train <- as.matrix(data[-cv_cut[round((i - 1)/cv_num*n + 1):round(i/cv_num*n)],])
    cv_lasso <- glmnet(train[, -12], train[, 12])
    pred <- predict(cv_lasso, test[, -12])
    errors[i] <- mean((test[, 12] - pred)^2)
  }
  return(mean(errors))
}

cv_lasso(data = data2, form = shares ~ ., cv_num = 10) # error: 0.9838633

library(flare)
cv_sqlasso <- slim(matrix_data[, -12], matrix_data[, 12], method = "lq", nlambda = 40, lambda.min.value = sqrt(log(11)/39643), q = 2)

cv_sqlasso <- function(data, cv_num, method, q, seed = 250)
{
  set.seed(seed)
  n <- dim(data)[1]
  cv_cut <- sample(1:n)
  errors <- rep(0, cv_num)
  for(i in 1:cv_num)
  {
    test <- as.matrix(data[cv_cut[round((i - 1)/cv_num*n + 1):round(i/cv_num*n)],])
    train <- as.matrix(data[-cv_cut[round((i - 1)/cv_num*n + 1):round(i/cv_num*n)],])
    cv_lasso <- slim(train[, -12], train[, 12], method = method, nlambda = 40, lambda.min.value = sqrt(log(11)/39643), q = q)
    pred <- predict(cv_lasso, newdata = test[, -12], lambda.idx = c(40:40), Y.pred.idx = c(1:1))
    pred <- as.data.frame(pred)
    errors[i] <- mean((test[, 12] - pred[, 1])^2)
  }
  return(mean(errors))
}lq_cv <- cv_sqlasso(data = data2, "lq", 2, cv_num = 10) # error: 0.9823111
lad_cv <- cv_sqlasso(data = data2, "lq", 1, cv_num = 10) # error: 0.983656

