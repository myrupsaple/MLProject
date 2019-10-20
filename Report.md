---
title: "Machine Learning Course Project"
author: "Riley Matsuda"
date: "10/19/2019"
output: 
  html_document: 
    keep_md: yes
---

# Preprocessing and Cleaning


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
dim(training); dim(testing)
```

```
## [1] 19622   160
```

```
## [1]  20 160
```

```r
summary(colSums(is.na(training)))
```

```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##       0       0       0    8047   19216   19216
```

```r
summary(colSums(is.na(testing)))
```

```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##     0.0     0.0    20.0    12.5    20.0    20.0
```

Initial observations of the datasets indicate a lot of variables with NA values.
We will remove all of the columns containing NA values. Furthermore, some of the
column classes in the training set do not match those in the test set. We will
remove those columns as well, with the exception of the test set's "problem_id"
column. In addition to removing these variables, the variables with near-zero
variation will be removed. The first five columns are labels that would not be 
expected to have any correlation to the data collected, so they will also be
removed.


```r
colToRemove <- (colSums(is.na(training)) != 0 | colSums(is.na(testing) != 0))
colToRemove <- !(sapply(training, class) == sapply(testing, class))
colToRemove[which(names(testing) == "problem_id")] <- FALSE
colToRemove[nearZeroVar(training)] <- TRUE
colToRemove[1:5] <- TRUE
names_train <- names(training)[!colToRemove]
names_test <- names(testing)[!colToRemove]
training <- training[!colToRemove]
testing <- testing[!colToRemove]
names(training) <- names_train
names(testing) <- names_test
dim(training); dim(testing)
```

```
## [1] 19622    51
```

```
## [1] 20 51
```

The data in both sets has been reduced to 51 dimensions from the original 160.

# Prediction Model Development

Before creating any prediction models, the training data will be subsetted into
training and validation sets.


```r
inValid <- createDataPartition(y = training$classe, p = 0.2, list = FALSE)
sub_train <- training[-inValid, ]
sub_valid <- training[inValid, ]
```

### Random Forest

The random forest model will be run on the subsetted training data. The number
of trees is limited to 50 to reduce runtime.


```r
set.seed(2000)
mod_rf <- train(classe ~ ., method = "rf", data = sub_train, ntree = 50,
                trControl = trainControl(method = "cv", number = 3))
mod_rf$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, ntree = 50, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 50
## No. of variables tried at each split: 26
## 
##         OOB estimate of  error rate: 0.28%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 4463    1    0    0    0 0.0002240143
## B    5 3028    3    1    0 0.0029634508
## C    0    8 2728    1    0 0.0032882718
## D    0    0   12 2559    1 0.0050544323
## E    1    1    0   10 2873 0.0041594454
```

```r
pred_rf <- predict(mod_rf, sub_valid)
cm_rf <- confusionMatrix(pred_rf, sub_valid$classe)
cm_rf
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1115    2    0    0    0
##          B    0  758    3    0    2
##          C    0    0  682    3    0
##          D    0    0    0  641    4
##          E    1    0    0    0  716
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9962          
##                  95% CI : (0.9937, 0.9979)
##     No Information Rate : 0.2842          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9952          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   0.9974   0.9956   0.9953   0.9917
## Specificity            0.9993   0.9984   0.9991   0.9988   0.9997
## Pos Pred Value         0.9982   0.9934   0.9956   0.9938   0.9986
## Neg Pred Value         0.9996   0.9994   0.9991   0.9991   0.9981
## Prevalence             0.2842   0.1935   0.1744   0.1640   0.1839
## Detection Rate         0.2839   0.1930   0.1737   0.1632   0.1823
## Detection Prevalence   0.2844   0.1943   0.1744   0.1642   0.1826
## Balanced Accuracy      0.9992   0.9979   0.9973   0.9971   0.9957
```

Based on the cross-validation data, the random forest model has an expected out 
of sample errror rate of 0.23%.

### Boosting

The generalized boosted model will be run on the subsetted training data.


```r
mod_gbm <- train(classe ~ ., method = "gbm", data = sub_train, verbose = FALSE,
                 trControl = trainControl(method = "cv", number = 3))
mod_gbm$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 50 predictors of which 50 had non-zero influence.
```

```r
pred_gbm <- predict(mod_gbm, sub_valid)
cm_gbm <- confusionMatrix(pred_gbm, sub_valid$classe)
cm_gbm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1110   12    0    0    0
##          B    5  735    9    0    3
##          C    0   10  676    4    2
##          D    1    3    0  638    7
##          E    0    0    0    2  710
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9852          
##                  95% CI : (0.9809, 0.9888)
##     No Information Rate : 0.2842          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9813          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9946   0.9671   0.9869   0.9907   0.9834
## Specificity            0.9957   0.9946   0.9951   0.9966   0.9994
## Pos Pred Value         0.9893   0.9774   0.9769   0.9831   0.9972
## Neg Pred Value         0.9979   0.9921   0.9972   0.9982   0.9963
## Prevalence             0.2842   0.1935   0.1744   0.1640   0.1839
## Detection Rate         0.2827   0.1872   0.1721   0.1625   0.1808
## Detection Prevalence   0.2857   0.1915   0.1762   0.1653   0.1813
## Balanced Accuracy      0.9952   0.9809   0.9910   0.9937   0.9914
```

The boosted model has an expected out of sample error rate of 1.3%, which is
slightly higher than the random forest model. We will use the random forest
model on our test data.

# Applying Our Model to the Test Set

```r
pred_final <- predict(mod_rf, testing)
pred_final
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

This output represents our predicted activity classes for the test data.
