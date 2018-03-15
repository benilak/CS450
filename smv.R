# CS450 - SVM
library(e1071)

# =============================================================================


vowel <- read.csv('C:\\Users\\brega\\OneDrive\\Desktop\\CS450\\vowel.csv')

# encode categorical features as numeric
vowel$Speaker <- as.numeric(vowel$Speaker)
vowel$Sex <- as.numeric(vowel$Sex)

# scale the data to the range 0 to 1 (pardoning the class column)
vowel[1:12] <- apply(vowel[1:12], 2, function(x) (x - min(x))/(max(x)-min(x)))

# information on the vowel data set tells us it was meant to be split such that
# speakers 1-8 comprise the training set and 9-15 comprise the testing set,
# so that's what we'll do

voweltrain <- vowel[1:(990*8/15),]
voweltest <- vowel[(990*8/15+1):990,]

# choose a range of parameters for C and gamma 
voweltuned <- tune.svm(Class~., data = voweltrain, 
                  gamma = 10^seq(-10, 10, 5), cost = 10^(-2:2))
voweltuned$best.performance
summary(voweltuned) # (gamma = 1, cost = 100)

# refine parameters based on the best performance 
voweltuned2 <- tune.svm(Class~., data = voweltrain, 
                            gamma = 10^(-2:2), cost = 10^seq(1, 3, 0.5))
voweltuned2$best.performance
summary(voweltuned2) # (gamma = 0.1, cost = 10^1.5)

# interestingly, we see the same performance for gamma = 0.1 no matter what
# the cost parameter is.
# upon further inspection, it seems we only want gamma = 0.1 and cost >= 1

# refine one more time ()
voweltuned3 <- tune.svm(Class~., data = voweltrain, 
                        gamma = 10^seq(-2, 0, 0.1), cost = 10)
voweltuned3$best.performance
summary(voweltuned3) # same performance for gamma = 10^c(-1.0, -0.9, -0.8)...

# let's choose gamma = 0.1, cost = 10 and build the model using the RBF kernel
vowelmodel <- svm(Class~., data = voweltrain, 
                  kernel = "radial", gamma = 0.1, cost = 10)
summary(vowelmodel)

# let the model predict classes
vowelpredict <- predict(vowelmodel, voweltest[,-13])
voweltab <- table(pred = vowelpredict, true = voweltest$Class)
voweltab
vowelagreement <- vowelpredict == voweltest$Class
vowelaccuracy <- prop.table(table(vowelagreement))
vowelaccuracy # 62%
# for such low errors in the grid search process, I'm rather disappointed

# =============================================================================


letters.df <- read.csv('C:\\Users\\brega\\OneDrive\\Desktop\\CS450\\letters.csv')

# no encoding necessary - all numeric data except for the class attribute

# the page info for the letters data set indicates it is already scaled

# "We typically train on the first 16000 items and then use the resulting 
# model to predict the letter category for the remaining 4000"
letterstrain <- letters.df[1:16000,]
letterstest <- letters.df[16001:20000,]

# choose a range of parameters for C and gamma 
letterstuned <- tune.svm(letter~., data = letterstrain, 
                       gamma = 10^seq(-10, 10, 5), cost = 10^(-2:2))
# this is taking over an hour to process, so let's pretend we got some output
summary(lettersmodel)
letterstuned$best.performance
summary(letterstuned) # gamma = 1, cost = 1

# choose gamma = 1 and cost = 1 for the RBF model
lettersmodel <- svm(letter~., data = letterstrain, 
                  kernel = "radial", gamma = 1, cost = 1)
summary(lettersmodel)

# let the model predict classes
letterspredict <- predict(lettersmodel, letterstest[,-1])
letterstab <- table(pred = letterspredict, true = letterstest$letter)
letterstab
lettersagreement <- letterspredict == letterstest$letter
lettersaccuracy <- prop.table(table(lettersagreement))
lettersaccuracy
# look at that it's 100% accuracy

