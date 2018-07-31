dataset = read.csv("50_Startups.csv")



#Encoding categorical data
dataset$State = factor(dataset$State,
                       levels = c('New York','California','Florida'),
                       labels = c(1,2,3))

library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

#fitting multiple linear regression to the training set
regressor = lm(formula = Profit ~ ., 
               data = training_set)

#Predicting the test set results
y_pred = predict(regressor, newdata = test_set)

#Building the optimal model using backward elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State, 
               data = dataset)
summary(regressor)

#Removing states
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend, 
               data = dataset)
summary(regressor)

#Removing Administration
regressor = lm(formula = Profit ~ R.D.Spend  + Marketing.Spend, 
               data = dataset)
summary(regressor)

#Removing Marketing spend
regressor = lm(formula = Profit ~ R.D.Spend, 
               data = dataset)
summary(regressor)