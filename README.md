# Allstate

In this program, we will take a look at the Allstate Kaggle competition. The script explores the data, 
models an xgboost model and tunes the parameters. There are two scripts included in this a python and an R script.

## Getting Started

This competition can be found at the following link
https://www.kaggle.com/c/allstate-claims-severity

### Prerequisites

What things you need to install the software and how to install them

###Python
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor```
```
###R
```
library(readr)
library(rpart)
library(lattice)
library(Matrix)
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)
library(PCAmixdata)
library(hydroGOF)
library(reshape)
library(onehot)
library(dplyr)
```


## Deployment

Both scripts essentially accomplish the same thing. They go through and explore the data the model and tune the paramters. 
The Python script takes a bit more data manipulation inorder to label and one hot encode the categorical variables.
## Built With



## Authors

###Python
@Ned Hulseman

###R
@James Now


