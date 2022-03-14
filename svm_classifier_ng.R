# function to construct and train support vector machine (SVM) classifier
train <- function() {
  
  # necessary packages
  library(caret)
  library(e1071)
  library(pROC)
  
  # read in beta values and classification assignments from discovery cohort
  bVals = as.data.frame(t(read.csv("beta_values.csv", row.names=1)))
  clusters = read.csv("subgroups.csv", row.names=1)
  mlDat = data.frame(clusters = as.character(clusters$x), bVals)
  mlDat$clusters = as.factor(mlDat$clusters)
  
  # slice data into training (75%) and testing (25%) partitions
  set.seed(1234)
  intrain <- createDataPartition(y = mlDat$clusters, p= 0.75, list = FALSE)
  training <- mlDat[intrain,]
  testing <- mlDat[-intrain,]
  
  # train SVM model using training partition with 10-fold cross validation
  trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
  svm_Linear <- train(clusters ~., data = training, method = "svmLinear",
                      trControl=trctrl,
                      preProcess = c("center", "scale"),
                      tuneLength = 10)
  
  # test SVM model using testing partition
  test_pred <- predict(svm_Linear, newdata = testing)
  confusionMatrix(test_pred, testing$clusters)
  
  # save SVM classifier
  saveRDS(svm_Linear, file="svm_linear_classifier.rds")
}

train()