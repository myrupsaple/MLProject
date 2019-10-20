---
title: "Machine Learning Course Project"
author: "Riley Matsuda"
date: "10/19/2019"
output: 
  html_document: 
    keep_md: yes
---

# Preprocessing

Initial observations identify NA values in the data. First, we observe the 
distribution of NA values across columns (variables).


```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
training <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
testing <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
dim(training)
```

```
## [1] 19622   160
```

```r
nas <- is.na(training)
colSums(nas)[16:20]
```

```
## skewness_roll_belt.1    skewness_yaw_belt        max_roll_belt 
##                    0                    0                19216 
##       max_picth_belt         max_yaw_belt 
##                19216                    0
```

```r
sum(colSums(nas) == 0 | colSums(nas) == 19216)
```

```
## [1] 160
```

We can see that all 160 variables either have 0 or 19216 NAs out of 19622
total observations (only a subset of the variable NA sums are shown). Now, we
will look at the variation of NA values across rows (observations).


```r
colInTrain <- as.logical(colSums(nas) == 0)
head(colInTrain, 20)
```

```
##  [1]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
## [12]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE FALSE  TRUE
```

```r
n.NACols <- sum(!colInTrain)
sum(rowSums(nas) == 0 | rowSums(nas) == n.NACols)
```

```
## [1] 19622
```

Analysis shows that for each observation, we either see all 67
NA values, or none at all. Does the absence of these particular NA values
correspond to a particular `classe` value?


```r
rowInTrain <- as.logical(rowSums(nas) == n.NACols)
NAClasses <- training$classe[!rowInTrain]
length(NAClasses)
```

```
## [1] 406
```

```r
summary(NAClasses)
```

```
##   A   B   C   D   E 
## 109  79  70  69  79
```

These NA values do not appear to correspond to any particular activity type.
We will therefore omit these 406 observations from our data,
and remove the columns filled with NA values.


```r
names <- names(training)
training <- training[rowInTrain, colInTrain]
names <- names[colInTrain]
names(training) <- names
dim(training)
```

```
## [1] 19216    93
```

We have now reduced our dataset to 93 observations of 
19216 variables.

However, not all of the test data variables are formatted in the same manner as
the training data. We will remove any variables that are not of the same class
across both sets.


```r
# Gather the same columns selected in the training set
testing <- testing[, colInTrain]
test.nas <- is.na(testing)
summary(rowSums(test.nas))
```

```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##      33      33      33      33      33      33
```

```r
colInTest <- colSums(test.nas) == 0
# Remove from both sets any columns in the test set with NA values
testing <- testing[, colInTest]
training <- training[, colInTest]
# Remove any columns whose class does not match across data sets, except for the
# "problem_id" column
keepCols <- sapply(training, class) == sapply(testing, class)
keepCols[which(names(testing) == "problem_id")] <- TRUE
training <- training[,keepCols]
testing <- testing[,keepCols]
```

The final step before analysis is to create a validation set for testing our
model.



# Developing an Prediction Model

We will now create a function which produces cross validation groups within our
training set.


```r
cv.split <- function(trainSet, nTrain){
        inTrain <- createDataPartition(y = trainSet$classe, 
                                       p = nTrain/dim(training)[1], list = FALSE)
        train <- trainSet[inTrain,]
        test <- trainSet[-inTrain,]
        list(train, test)
}
```

From here, we can iteratively repeat the process of cross-validating and model
fitting, and store the best fitting model. 

We will try 10 iterations of a random forest, using a training set of 300
and a maximum of 10 trees per parameter.


```r
set.seed(2000)
fit_rf <- function(trainSet, nInTrain, nTree, nRep){
        models <- list()
        accuracies <- numeric()
        for(i in 1:nRep){
                new.data <- cv.split(trainSet, nInTrain)
                model_rf <- train(classe ~ ., data = new.data[[1]], 
                                  method = "rf", ntree = nTree)
                models[[i]] <- model_rf
                pred_rf <- predict(model_rf, new.data[[2]])
                accuracies[i] <- sum(pred_rf == new.data[[2]]$classe)/length(new.data[[2]]$classe)
        }
        list(models, accuracies)
}

data <- fit_rf(training, 500, 20, 30)
# The best accuracy across the 20 models was:
index <- which.max(data[[2]])
data[[2]][index]
```

```
## [1] 0.998823
```
The prediction model was 99.93% accurate across the test subset within the
training set.


```r
mod_rf_best <- data[[1]][[index]]
mod_rf_best$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, ntree = ..1, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 20
## No. of variables tried at each split: 78
## 
##         OOB estimate of  error rate: 0.8%
## Confusion matrix:
##     A  B  C  D  E class.error
## A 142  1  0  0  0 0.006993007
## B   1 96  0  0  0 0.010309278
## C   0  0 88  0  0 0.000000000
## D   0  0  0 80  2 0.024390244
## E   0  0  0  0 92 0.000000000
```

We now have a model that we can use to predict the class of activity that was
being performed. The predicted out of sample error rate is 0.6%.

# Testing Our Prediction Model
### The Validation Set

Now that we have our "best" model, we will test it on the validation set.


```r
pred_rf <- predict(mod_rf_best, validation)
sum(pred_rf[[1]] == validation$classe)/length(validation$classe)
```

```
## [1] 0.2727273
```

The prediction model correctly identified all 20 of the validation classes. 

### The Test Set

We will now apply our model to the test data.


```r
pred_rf_test <- predict(mod_rf_best, testing)
pred_rf_test
```

```
##  [1] A A A A A A A A A A A A A A A A A A A A
## Levels: A B C D E
```

Our model predicts that the test data was all collected under theClass A: 
performing the workout exactly according to the specification.
