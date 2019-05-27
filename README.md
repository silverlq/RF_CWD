RF_CWD
==============

Random forests (RF) classification of coarse woody debris (CWD) image-objects (segments of aerial images)
--------------

**Author:** Gustavo Lopes Queiroz

**Reference:** This repository serves as supplementary material to the paper of Queiroz et al. (2019) entitled: "Mapping Coarse Woody Debris with Random Forest Classification of Centimetric Aerial Imagery"

**Created:** October, 2018

**Published:** May, 2019

**Input:** CSV table containing rows of image-objects and columns of attributes to be used in training and classification. One of the columns must be called 'ClassID' which will be used as the reference class for training and testing purposes. It is possible to input a second verification table to be used as the testing set.
			 
**Description:** Divides an input table into training and testing datasets, trains a Random Forests (RF) classifier using the training dataset and applies it to the testing dataset, assessing the classification accuracy of CWD objects

**Functions:** Each function performs different accuracy tests by incrementally changing the training parameters and datasets

**Output:** CSV tables containing different accuracy metrics depending on the functions used
