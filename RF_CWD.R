#####################################################################################################################################
#
# Name: RF_CWD (Random Forests of Coarse Woody Debris)
# Author: Gustavo Lopes Queiroz
# Created: October, 2018
#
# Input: CSV table containing rows of image-objects and columns of attributes to be used in training and classification. One of the
#              columns must be called 'ClassID' which will be used as the reference class for training and testing purposes.
#
# Description: Divides an input table into training and testing datasets, trains a Random Forests (RF) classifier using the
#              training dataset and applies it to the testing dataset, assessing the classification accuracy of CWD objects
#
# Functions: Each function performs different accuracy tests by incrementally changing the training parameters and datasets
#
# Output: CSV tables containing different accuracy metrics depending on the functions used
#
#####################################################################################################################################

library(splitstackshape)
require(randomForest)
library('scales')

#Input CSV file
f <- file.choose()
d <- read.csv(f)

#Verification Area
verification = askYesNo("Would you like to input a separate dataset to use as verification? (If not the dataset will be split into testing and verification)", default = TRUE)
if(verification == TRUE){
  f <- file.choose()
  dV <- read.csv(f)
}

#Limit classes to: Logs(1), snags(2), water(3), dirt(4) and other(5) by merging classes larger than 5 into class 5
d$ClassID[d$ClassID>4 & d$ClassID<9] = 5
if(verification == TRUE){
  dV$ClassID[dV$ClassID>4 & dV$ClassID<9] = 5
}

#Create a folder to place the output files
outFolder = paste(dirname(f),"/Outputs",sep = "")
if(!dir.exists(outFolder))
{
  dir.create(outFolder)
}
#Output CSV files
filePred <- paste(outFolder,"/","rf_prediction.csv",sep = "")
fileConfusion <- paste(outFolder,"/","rf_confusion.csv",sep = "")
filePrediction <- paste(outFolder,"/","rf_classPrediction.csv",sep = "")
fileImportance <- paste(outFolder,"/","rf_attImportance.csv",sep = "")
fileAttNum <- paste(outFolder,"/","rf_attNumber.csv",sep = "")
fileSampleNum <- paste(outFolder,"/","rf_sampleNumber.csv",sep = "")
fileElevTest <- paste(outFolder,"/","rf_elevTest.csv",sep = "")
fileAccTest <- paste(outFolder,"/","rf_accuracyTest.csv",sep = "")
fileMultiPred <- paste(outFolder,"/","rf_multiPred.csv",sep = "")

#Set the arbitrary random seed to make test reproducible
set.seed(682)

#Number of iterations for multiple tests
multiIter = 100

#Mode function
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

#--RUN RANDOM FORESTS AND ASSESS ACCURACY OF CWD-------------------------------------------------------------------------------------
#
runRF <- function( badCols = c(), resetSample = TRUE, printResults = FALSE, saveImportance = FALSE, savePrediction = FALSE, dubiousToCWD=FALSE, dubiousFractionCWD=0.3 ){
  
  if(verification !=TRUE){
    stratD <- stratified(d, "ClassID", 0.8, bothSets = TRUE)
    dTrain <- stratD[[1]]
    dTest <- stratD[[2]]
  }else{
    dTrain <- stratified(d, "ClassID", 0.8, bothSets = FALSE)
    dTest <- stratified(dV, "ClassID", 0.8, bothSets = FALSE)
  }
  
  if(dubiousToCWD)
  {
    dTest$ClassID[dTest$ClassID>10] <- dTest$ClassID[dTest$ClassID>10] - (runif(length(dTest$ClassID[dTest$ClassID>10]))<dubiousFractionCWD)*10
    dTest$ClassID[dTest$ClassID>10] <- 5
  }
  
  #Remove unknown and irrelevant samples from test (Class ID 9, 11 or 12)
  dTrain <- dTrain[!dTrain$ClassID>8,]
  if(!savePrediction)
  {
    dTest <- dTest[!dTest$ClassID>8,]
  }
  else
  {
    dTest <- dV
  }

  #Get the class of train and test objects ('y' vector)
  dTrainClass <- factor(dTrain$ClassID)
  dTestClass <- factor(dTest$ClassID)

  #Remove unwanted columns from predictor matrix
  cnames <- names(d)
  dropCols <- c(badCols, "OBJECTID", "FID", "FID_1", "FID_2", "ClassID", "Type", "AssignedClass")
  keepCols <- cnames[!(cnames %in% dropCols)]
  #Get the predictor matrix ('x' matrix)
  dTestPredictors <- subset(dTest, select= keepCols)
  dTrainPredictors <- subset(dTrain, select= keepCols)

  #Train classifier with training subset
  rf <- randomForest(dTrainPredictors, y=dTrainClass, ntree=100, keep.forest=TRUE, importance=TRUE)

  #Predict classes of testing subset
  pred <- predict(rf, newdata=dTestPredictors, type="class")
  
  #Put actual vs predicted classes side by side in a table
  results = data.frame(actual = as.numeric(dTest$ClassID), predicted = as.numeric(pred), area = as.numeric(dTest$AreaPxl), 
                       union = "-", log = "-", snag = "-", stringsAsFactors=FALSE)
  actualPredicted <- cbind(dTest$ClassID, pred)

  #Confusion matrix actual vs predicted
  confusionMatrix <- table(pred,dTest$ClassID)

  #Accuracy of compiled CWD
  results$union[results$actual < 3 & results$predicted < 3] = "tp"
  results$union[results$actual > 2 & results$predicted < 3] = "fp"
  results$union[results$actual < 3 & results$predicted > 2] = "fn"
  tp <- sum(results$area[results$union == "tp"])
  fp <- sum(results$area[results$union == "fp"])
  fn <- sum(results$area[results$union == "fn"])
  comp <- tp/(tp+fn)
  corr <- tp/(tp+fp)

  #Accuracy of Logs
  results$log[results$actual == 1 & results$predicted == 1] = "tp"
  results$log[results$actual != 1 & results$predicted == 1] = "fp"
  results$log[results$actual == 1 & results$predicted != 1] = "fn"
  tpL <- sum(results$area[results$log == "tp"])
  fpL <- sum(results$area[results$log == "fp"])
  fnL <- sum(results$area[results$log == "fn"])
  compL <- tpL/(tpL+fnL)
  corrL <- tpL/(tpL+fpL)  

  #Accuracy of Snags
  results$snag[results$actual == 2 & results$predicted == 2] = "tp"
  results$snag[results$actual != 2 & results$predicted == 2] = "fp"
  results$snag[results$actual == 2 & results$predicted != 2] = "fn"
  tpS <- sum(results$area[results$snag == "tp"])
  fpS <- sum(results$area[results$snag == "fp"])
  fnS <- sum(results$area[results$snag == "fn"])
  compS <- tpS/(tpS+fnS)
  corrS <- tpS/(tpS+fpS)  

  if(printResults){
    cat("-----\n*** RANDOM FORESTS CWD IMAGE-OBJECT CLASSIFICATION ***\n", sep = '')
    #print(keepCols)
    cat("N| Training: ", nrow(dTrain), "; Testing: ",nrow(dTest),"\n", sep = '')
    cat("N| Attributes: ", length(keepCols), "\n", sep = '')
    cat("-----\n")
    print(confusionMatrix)
    cat("-----\n")
    cat("True positives: ",tp," | ",sep = '')
    cat("False positives: ",fp," | ",sep = '')
    cat("False negatives: ",fn,"\n",sep = '')
    cat("Completeness: ",percent(comp)," | ",sep = '')
    cat("Correctness: ",percent(corr)," | ",sep = '')
    cat("Omission: ",percent(1-comp)," | ",sep = '')
    cat("Comission: ",percent(1-corr),"\n",sep = '')
    cat("-----\n")
  }

  if(saveImportance){
    write.csv(actualPredicted, file = filePred)
    attImportance <- importance(rf)
    write.csv(attImportance , file = fileImportance)
  }
  
  if(savePrediction)
  {
    dPrediction = dTest
    dPrediction$Prediction = pred
    write.csv(dPrediction , file = filePrediction)
    write.csv(confusionMatrix, file = fileConfusion)
  }

  return(c(comp,corr,compL,corrL,compS,corrS))
}
#
#------------------------------------------------------------------------------------------------------------------------------------

#--MULTIPLE RUNS OF RF, RETURN AVERAGE ACCURACY AND STANDARD DEVIATION---------------------------------------------------------------
#
multipleRF <- function( iter, badCols = c() ){
  accList <- matrix(NA, nrow=iter, ncol=12)
  for(i in c(1:iter)){
    acc <- runRF(badCols,TRUE,FALSE,FALSE)
    #cat(i," of ",iter,"\n",sep = '')
    accList[i,] <- acc
  }
  layout(matrix(c(1,2), 1, 2, byrow = TRUE))
  hist(accList[,1], main="Completeness All Histogram")
  hist(accList[,2], main="Correctness All Histogram")
  return(c(mean(accList[,1]),mean(accList[,2]),mean(accList[,3]),mean(accList[,4]),mean(accList[,5]),mean(accList[,6]),
           sd(accList[,1]),  sd(accList[,2]),  sd(accList[,3]),  sd(accList[,4]),  sd(accList[,5]),  sd(accList[,6])))
}
#
#------------------------------------------------------------------------------------------------------------------------------------

#--RUN RF MULTIPLE TIMES AND AVERAGE ATTR. IMPORTANCE MATRIX-------------------------------------------------------------------------
#
importanceTest <- function(){
  AttImpMatr <- list()
  for(i in c(1:multiIter)){
    runRF(c(),TRUE,FALSE,TRUE)
    attImp <- read.csv(fileImportance)
    AttImpMatr[[i]] <- attImp
  }
  matSize <- dim(AttImpMatr[[1]])
  joinMatrices <- matrix(unlist(AttImpMatr), ncol=length(AttImpMatr) )
  meanOfMatrices <- matrix(rowMeans(joinMatrices),matSize[1],matSize[2])
  write.csv(meanOfMatrices, file = fileImportance)
  cat("Number of RF importance matrices averaged: ", length(AttImpMatr))
}
#
#------------------------------------------------------------------------------------------------------------------------------------

#--DROP BAD ATTR. ONE BY ONE AND MEASURE IMPACT IN ACCUR.----------------------------------------------------------------------------
#
attributeNumTest <- function() {
  badCols <- c("PolySelfIn", "CurvByLen", "BorConCHM", "SkewBlue", "EllipticFi", "SkewGreen", "Roundness", "Perimeter", "SkewRed", "BorConDSM", "RectFit", "LengthPxl", "MinDSM", "SkewNDVI", "LenByWidMl", "WidthMl", "MaxDSM", "BorIndex", "MeaInBorDS", "MeanDSM", "MeaOutBorD", "NumPxls", "AreaPxl", "StdDevBlue", "SkewDSM", "SkewCHM", "StdDevNIR", "MeanStdDev", "StdDevNDVI", "StdDevRed", "MinCHM", "StdDevGree", "RatioGreen", "MeanGreen", "WidthPoly", "Brightness", "DifNeiNIR", "MeanRed", "MyBrightne", "Compactnes", "StdDevCHM", "BorConNDVI", "Redness", "RatioRed", "ShapeIndex", "DifNeiNDVI", "MeanBlue", "MaxNDVI", "BorConBlue", "MaxCHM", "DifNeiGree", "MeaOutBorN", "DifNeiBlue", "StdDevDSM", "RatioBlue", "MeanNIR", "BorConGree", "DifNeiRed", "BorConRed", "Blueness", "BorLenByLe", "MeaOutBorC", "LenByWidPo", "Asymmetry", "MinNDVI", "MeaInBorCH", "Density", "MeanCHM", "Greenness", "MeaInBorND", "MeanNDVI")
  accList <- matrix(NA, nrow=length(badCols), ncol=12)
  for(n in c(1:(length(badCols)-1))){
    acc <- multipleRF(multiIter, badCols[1:n])
    cat(n," of ",length(badCols),"\n",sep = '')
    print(acc)
    accList[n,] <- acc
  }
  cat(accList)
  write.csv(accList, file = fileAttNum)
}
#
#------------------------------------------------------------------------------------------------------------------------------------

#--CHM VERSUS DSM--------------------------------------------------------------------------------------------------------------------
#
elevationTest <- function(){
  elevResults <- matrix(nrow = 6, ncol = 12)
  
  #Run with all spectral, spatial and elevation attributes
  badCols <-c()
  elevResults[1,] <- multipleRF(multiIter, badCols)
  
  #Run with all spectral, spatial and CHM attributes (remove DSM)
  elevResults[2,] <- multipleRF(multiIter, c(badCols, "BorConDSM", "MaxDSM", "MeanDSM", "MeaInBorDS", "MeaOutBorD", "MinDSM", "SkewDSM", "StdDevDSM"))
  
  #Run with all spectral, spatial and CHM attributes (remove CHM)
  elevResults[3,] <- multipleRF(multiIter, c(badCols, "BorConCHM", "MaxCHM", "MeanCHM", "MeaInBorCH", "MeaOutBorC", "MinCHM", "SkewCHM", "StdDevCHM"))
  
  #Run with all spectral and spatial attributes (remove elevation data)
  badCols <- c(badCols, "BorConCHM", "MaxCHM", "MeanCHM", "MeaInBorCH", "MeaOutBorC", "MinCHM", "SkewCHM", "StdDevCHM")
  badCols <- c(badCols, "BorConDSM", "MaxDSM", "MeanDSM", "MeaInBorDS", "MeaOutBorD", "MinDSM", "SkewDSM", "StdDevDSM")
  elevResults[4,] <- multipleRF(multiIter, badCols)
  
  #Run with all elevation attributes (remove spectral and spatial data)
  badCols <- c("PolySelfIn", "CurvByLen", "BorConCHM", "SkewBlue", "EllipticFi", "SkewGreen", "Roundness", "Perimeter", "SkewRed", "BorConDSM", "RectFit", "LengthPxl", "MinDSM", "SkewNDVI", "LenByWidMl", "WidthMl", "MaxDSM", "BorIndex", "MeaInBorDS", "MeanDSM", "MeaOutBorD", "NumPxls", "AreaPxl", "StdDevBlue", "SkewDSM", "SkewCHM", "StdDevNIR", "MeanStdDev", "StdDevNDVI", "StdDevRed", "MinCHM", "StdDevGree", "RatioGreen", "MeanGreen", "WidthPoly", "Brightness", 
               "DifNeiNIR", "MeanRed", "MyBrightne", "Compactnes", "StdDevCHM", "BorConNDVI", "Redness", "RatioRed", "ShapeIndex", "DifNeiNDVI", "MeanBlue", "MaxNDVI", "BorConBlue", "MaxCHM", "DifNeiGree", "MeaOutBorN", "DifNeiBlue", "StdDevDSM", "RatioBlue", "MeanNIR", "BorConGree", "DifNeiRed", "BorConRed", "Blueness", "BorLenByLe", "MeaOutBorC", "LenByWidPo", "Asymmetry", "MinNDVI", "MeaInBorCH", "Density", "MeanCHM", "Greenness", "MeaInBorND", "MeanNDVI")
  goodCols <- c("BorConDSM", "MaxDSM", "MeanDSM", "MeaInBorDS", "MeaOutBorD", "MinDSM", "SkewDSM", "StdDevDSM")
  goodCols <- c(goodCols, "BorConCHM", "MaxCHM", "MeanCHM", "MeaInBorCH", "MeaOutBorC", "MinCHM", "SkewCHM", "StdDevCHM")
  badCols <- badCols[!(badCols %in% goodCols )]
  elevResults[5,] <- multipleRF(multiIter, badCols)
  
  #Write test param
  elevResults[6,] <- c("Test sample size",multiIter,"","","","","","","","","","")
  
  #Write results in CSV
  write.csv(elevResults, file = fileElevTest)
}
#
#------------------------------------------------------------------------------------------------------------------------------------

#--TEST OPTIMAL SAMPLE N BY GRADUALLY REDUCING TRAINING DATASET----------------------------------------------------------------------
#
trainingNumTest <- function(){
  resampleIter = 100
  multiIter = 1
  verification <<- TRUE
  originalD <- d
  dSize <- 0.999
  reductionFac <- 0.9
  accList3 <- matrix(NA, nrow=100, ncol=13)
  n <- 1
  while( dSize > 0.005 )
  {
    accList2 <- matrix(NA, nrow=resampleIter, ncol=12)
    for(i in c(1:resampleIter)){
      stratD <- stratified(originalD, "ClassID", 0.8, bothSets = TRUE)
      d <<- stratified(stratD[[1]], "ClassID", dSize, bothSets = FALSE)
      dV <<- stratD[[2]]
      
      acc <- multipleRF(multiIter,c())
      accList2[i,] <- acc
    }
    accList3[n,] <- c(ceiling(nrow(d)*0.8),
                      mean(accList2[,1]),mean(accList2[,2]),mean(accList2[,3]),mean(accList2[,4]),mean(accList2[,5]),mean(accList2[,6]), 
                      sd(accList2[,1]),sd(accList2[,2]),sd(accList2[,3]),sd(accList2[,4]),sd(accList2[,5]),sd(accList2[,6]) )
    cat(accList3[n,],"\n",sep = ' ')
    n <- n+1

    dSize <- dSize * reductionFac 
  }
  write.csv(accList3, file = fileSampleNum )
}
#
#------------------------------------------------------------------------------------------------------------------------------------

#--RUN MULTIPLE ITERATIONS OF RF AND GET THE MODE OF PREDICTIONS FOR THE TEST SET----------------------------------------------------
#
multiPrediction <- function(iter) {
  predList <- matrix(NA, nrow=nrow(dV), ncol=iter)
  for(i in c(1:iter)){
    filePrediction <<- paste("rf_classPrediction_",i,".csv", sep="") 
    runRF(savePrediction = TRUE)
    dR <- read.csv(filePrediction)
    predList[,i] = dR$Prediction
    unlink(filePrediction, recursive = FALSE)
  }
  
  predMode <- matrix(NA, nrow=nrow(dV), ncol=1)
  for(i in c(1:nrow(dV))){
    predMode[i] = getmode(predList[i,])
  }
  
  write.csv(predMode, file = fileMultiPred)
}
#
#------------------------------------------------------------------------------------------------------------------------------------

# FUNCTION CALLS

runRF(printResults = TRUE, saveImportance = TRUE)
#write.csv(multipleRF(100), file = fileAccTest)
#multiPrediction(100)
#write.csv(multipleRF(100,badCols), file = fileAccTest)
#elevationTest()
#attributeNumTest()
#trainingNumTest()
