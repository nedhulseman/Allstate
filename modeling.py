import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


#######################
# @Ned Hulseman       #
# @12/8/2017          #
# @Allstate Kaggle    #
#######################

allstate = pd.read_csv('C:\\Users\\nedhu\\Desktop\\nedhulseman.com\\allstate\\train.csv')

############################################ Initial Exploration ##################################################
###################################################################################################################
print(allstate.shape) #188318, 132
print(list(allstate))
for col in allstate:
    print(allstate[col].dtype)
# @col 0: id variable
# @col 1:116 cat1:cat116
# @ col 117:130 cont1:cont14
# @col 131 loss
# object and float64 types



##################################
# dives variables into 3 sets    #
# loss/log_loss                  #
# categorical @cat               #
# continuous @cont               #
##################################
loss=allstate['loss']
log_loss=np.log(loss)
cat=allstate.select_dtypes(include=[object])
cont=allstate.iloc[:, 117:131]


num_levels=[]
for i in cat:
    levels=len(cat[i].unique())
    num_levels.append(levels)
###################################################################################################################
###################################################################################################################


############################################### loss Exploration ##################################################
###################################################################################################################
num_bins=20
n, bins, patches = plt.hist(loss, num_bins, facecolor='blue', alpha=0.5)

log_loss=np.log(loss)
n, bins, patches = plt.hist(np.log(loss), num_bins, facecolor='blue', alpha=0.5)
#Perhaps log of the y variable may be a good idea... could improve predicitona ccuracy
###################################################################################################################
###################################################################################################################




############################################### Cat exploration ###################################################
###################################################################################################################

for col in cat:
    print(cat[col].unique())
#cat 75 is last  2-level cat


for col in cat: #looks at level distribution
    plt.figure()
    sns.countplot(x=col, data=cat)


###################################################################################################################
###################################################################################################################

############################################### cont exploration ###################################################
###################################################################################################################
for col in cont:
    n, bins, patches = plt.hist(allstate[col])
    plt.title(col)
    plt.show()
    
  
###################################################################################################################
###################################################################################################################


cat=cat.values #changes type to a numpy.ndarray for ease of use
cont=cont.values #changes type to a numpy.ndarray


#################################################################################
# formats the categorical variables in a way so that it can be input into xgboost
# the label encoder changes the strings inputs to numbers

encoded_cat=None
for i in range(0, cat.shape[1]):
    label_encoder = LabelEncoder()#instantiates labelEncoder
    feature=label_encoder.fit_transform(cat[:, i]) #changes strings to numbers
    feature=feature.reshape(cat.shape[0], 1) #reshape from (0,5) to ()
    onehot_encoder=OneHotEncoder(sparse=False) #instantiates onehotencoder
    feature=onehot_encoder.fit_transform(feature) #turns levels to columns
    if encoded_cat==None:
        encoded_cat=feature
    else:
        encoded_cat=np.concatenate((encoded_cat, feature), axis=1)
        
X=np.concatenate((encoded_cat, cont), axis=1)



seed=3
test_size=.3

X_train, X_test, y_train, y_test = train_test_split(X, log_loss, test_size=test_size, random_state=seed)


model=XGBRegressor(learning_rate=0.08, 
                   max_depth=10, 
                   objective='reg:linear', 
                   nthread=3, 
                   gamma=0.2, 
                   subsample=0.9,
                   n_estimators=100,
                   )
model.fit(X_train, y_train)
print(model)
y_pred=model.predict(X_test)

def mae(predicted, actual, logscale=False):
    if logscale == True:
        predexp=np.exp(predicted)
        actualexp=np.exp(actual)
        return np.mean(np.abs(predexp - actualexp))
    else:
        return np.mean(np.abs(predicted - actual))

print(mae(y_pred, y_test, True))


#Plotting Variable Importance
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.title('Variable Importance')
