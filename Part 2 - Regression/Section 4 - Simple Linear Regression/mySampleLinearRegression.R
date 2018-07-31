#Simple Linear Regression

dataset = read.csv('Salary_Data.csv')

#splittin data into the Training set and the Test set
library(caTools)
set.seed(123)
#Split the dependent variable, Salary in this case
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)


#Fitting Simple Linear Regression to the Training set
#f1 to get information about something
#Salary is proportional to Years of Experience
regressor = lm(formula = Salary ~ YearsExperience, 
               training_set)

#Summary(object) give us informations about the object.
#if we see *** this means it's highly statistical significant
# the lower the P value is, the more significant the independent variable
#is going to be.
# Below 5% percent, independent variable is significant
# over 5%, indepedent variable is less significant

#predicting the test set results
y_pred = predict(regressor, newdata = test_set)

#Visualising the Training set results
# install.packages('ggplot2')
library(ggplot2)
ggplot() + 
  #plotting observation points
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') + 
  ggtitle('Salary vs Experience (Training set)')+
  xlab('Years of experience') +
  ylab('Salary')


#Visualising the test set results
ggplot() + 
  #plotting observation points
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') + 
  ggtitle('Salary vs Experience (Training set)')+
  xlab('Years of experience') +
  ylab('Salary')
