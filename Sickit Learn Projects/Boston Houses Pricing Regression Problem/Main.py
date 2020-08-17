# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 19:41:14 2020

@author: Mahdi
"""

#%% Importing Libraries

# Good for addresses
from pathlib import Path

# Good for addresses too
import os

# Good for plotting
import matplotlib.pyplot as plt

# The same code as above
from matplotlib import pyplot

# Famous Numpy
import numpy as np

# Good for loading data
import pickle

# Excellent for loading data
from pandas import read_csv

# Scatter_Matrix plotting tool of pandas
from pandas.plotting import scatter_matrix

# Seaborn has got handy tools in plotting too
import seaborn as sns


################ SKLEARN ################





         ### Pre-Processings ###

# Min-Max Scaler... Pretty simple
from sklearn.preprocessing import MinMaxScaler

# Standardization (Makes data to have mean=0 and std=1)
from sklearn.preprocessing import StandardScaler

# Normalizer (Good for sparse data with a lot of 0)
from sklearn.preprocessing import Normalizer

# Makes it binary :)
from sklearn.preprocessing import Binarizer





         ### Feature Selection ###
         
# Univariate Selection : select k-best scored based on chi2 scoring method
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2         

# Recursive Feature Selection
from sklearn.feature_selection import RFE

# The mighty PCA
from sklearn.decomposition import PCA

# ??? Test if Ensemble methods like Random Forest and Extra Forests can
#     be used as feature selectors or not!!    
from sklearn.ensemble import ExtraTreesClassifier





         ### Model Selection (For Data Splitting) ###

# Good for splitting train and test sets
from sklearn.model_selection import train_test_split

# Importing cross_val_score to be able to score more than "model.score" that
# is for the "model" itself and only measures accuracy by 
# "correct_predictions / total samples"
from sklearn.model_selection import cross_val_score

# You can do K-Folds on data rather than "train_test_split"
# to devide it more than one part and make cross_vals better and better
from sklearn.model_selection import KFold

# Leave One Out (k=1 in k-folds-cross-val)
from sklearn.model_selection import LeaveOneOut

# Shuffler
from sklearn.model_selection import ShuffleSplit






         ### Model Selection (parameter optimization) ###

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV






         ### Performance Metrics (Classification) ###
         
# The accuracy metric
from sklearn.metrics import accuracy_score         
         
# Confusion matrix is F1_score , recall, precision story
# It's for multiclass
from sklearn.metrics import confusion_matrix

# The best classification metric ever
from sklearn.metrics import classification_report         




         ### Performance Metrics (Regression) ###
         
# The best regressor metric : MSE
from sklearn.metrics import mean_squared_error




         ### Classification Algorithms ###


from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier





         ### Regression Algorithms ###
         
from sklearn.linear_model import LinearRegression         
      

from sklearn.linear_model import Ridge
    

from sklearn.linear_model import Lasso


from sklearn.linear_model import ElasticNet


from sklearn.neighbors import KNeighborsRegressor


from sklearn.tree import DecisionTreeRegressor


from sklearn.svm import SVR


from sklearn.neural_network import MLPRegressor





         ### Ensembles (Classification) ###

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier    

from sklearn.ensemble import VotingClassifier



         ### Ensembles (Regression) ###
         
from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import VotingRegressor




         ### Pipelines ###
from sklearn.pipeline import Pipeline

from sklearn.pipeline import FeatureUnion

#%% Address Definings

dataset_address = 'E:\\My Projects\\2019 Works\\Datasets\\For Begginers\\Housing'

code_directory = 'E:\\My Projects\\2019 Works\\Python\\Spyder\\Real Work Tests\Sickit Learn Projects\\Boston Houses Pricing Regression Problem'

#%% Data Information (Feature info mostly)

'''
Attribute Information:

    1. CRIM      per capita crime rate by town
    2. ZN        proportion of residential land zoned for lots over 
                 25,000 sq.ft.
    3. INDUS     proportion of non-retail business acres per town
    4. CHAS      Charles River dummy variable (= 1 if tract bounds 
                 river; 0 otherwise)
    5. NOX       nitric oxides concentration (parts per 10 million)
    6. RM        average number of rooms per dwelling
    7. AGE       proportion of owner-occupied units built prior to 1940
    8. DIS       weighted distances to five Boston employment centres
    9. RAD       index of accessibility to radial highways
    10. TAX      full-value property-tax rate per $10,000
    11. PTRATIO  pupil-teacher ratio by town
    12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks 
                 by town
    13. LSTAT    % lower status of the population
    14. MEDV     Median value of owner-occupied homes in $1000's


Number of Attributes: 13 continuous attributes (including "class"
                         attribute "MEDV"), 
                        1 binary-valued attribute(River-Sided).
                        
Relevant Information:

    Concerns housing values in suburbs of Boston.

Number of Instances: 506


'''

#%% Loading Dataset

# Going to dataset address
os.chdir(dataset_address)

# Selecting names for attributes (features)
names = ['CRIM','ZN','INDUS','CHAS','NOX' , 'RM' , 'AGE' , 'DIS' , 'RAD' , 'TAX' , 'PTRATIO' , 'B' , 'LSTAT' , 'MEDV']

# See what's inside the directory
f = os.listdir()

#Loading the dataset
for i,item in enumerate(f):
    if f[i] == 'housing.data':
        dataset = read_csv(f[i] , delim_whitespace = True  ,names=names)

# Let's get back to our code's directory
os.chdir(code_directory)

#%% Prepare data for learning

# Getting rid of explanations and to have only numbers
array = dataset.values

# Input
X = array[:,0:13]

# Target
Y = array[:,13]

# Setting Test_Sets's size (Usually 33% of the dataset)
test_size = 0.33

#  Seeding the random selection to be repeatable
seed = 7

# Seperating Training_Sets and Test_Sets
xtrain , xtest , ytrain , ytest = train_test_split(X , Y , test_size = test_size , random_state = seed)

# Seeing shapes of each
print(f'X_Train Shape : {xtrain.shape} \nY_Train Shape : {ytrain.shape} \nX_Test Shape : {xtest.shape} \nY_Test Shape : {ytest.shape}')

#%% Machine Learning
# An empty list of models
models = []

# Appending every algorithm with a nick name to the list to make a tuple of 
# the nick name and the algorithm itself
models.append(('LinR',LinearRegression()))
models.append(('LASSO',Lasso()))
models.append(('KNN',KNeighborsRegressor()))
models.append(('Tree',DecisionTreeRegressor()))
models.append(('Elastic',ElasticNet()))
models.append(('SVM',SVR()))
models.append(('MLP',MLPRegressor(max_iter = 1000)))
models.append(('RForest',RandomForestRegressor()))
models.append(('GBR',GradientBoostingRegressor()))
models.append(('ExtraTR',ExtraTreesRegressor()))
models.append(('ADABoost',AdaBoostRegressor()))




# Two lists to store "results" and "names of the algorithms" inside
results = []
names = []

# Pay attention that we shouldn't use enumerate since we don't want indexs 
# (numbers) of each element in the model but we want the elements themselves.
for name , model in models:
    
    kfold = KFold(n_splits = 10 , random_state = 7)
    
    # In here, we passed 'xtrain' , 'ytrain' in order to train over these two
    # and then be able to test our predictions. It's too much better to first
    # train on the data and then test instead of giving all the data in cross
    # validation.
    cv_result = cross_val_score(model , xtrain , ytrain , scoring = 'neg_mean_squared_error' , cv = kfold )
    
    results.append(cv_result)
    names.append(name)
    
# Printing the results    
for i in range(len(names)):
    
    print(f'\n The {names[i]} Algorithm had : {results[i].mean() } % as mean of the results of CV_Scoring based on "NegMSE" and {results[i].std() } % as std of the results \n')
    
#%% Some other tasks on it

# Plotting the box plot of results
fig = plt.figure()

ax = fig.add_subplot(111)

pyplot.boxplot(results)

ax.set_xticklabels(names)


# F1_Score of each model
for name , model in models:
    tmp = model
    tmp.fit(xtrain , ytrain)
    predictions = tmp.predict(xtest)
    
    print(f'\n MSE Report of {name} Algorithm :\n {mean_squared_error(ytest , predictions)} \n')


# An algorithm to review results sorted by the amount of MSE in an ascending 
# mode
resulterman = []
for ind , item in enumerate(results):
    resulterman.append([names[ind] , abs(results[ind].mean())])

# Sorting
sorter =  sorted(resulterman , key=lambda x:x[-1])

result_presenter_1 = []
for ind , item in enumerate(sorter):
    result_presenter_1.append([sorter[ind][0] , sorter[ind][1]])

## Save this result_presenter cause t's the best
filename = 'result_presenter_1'
f = open(filename,'wb')
pickle.dump(result_presenter_1 , f)
f.close()

#%% Now , a little bit of preprocessings and then calculating metrics

## First of all, sklearn.pipeline.Pipeline() should contain a tuple, that might
 # contain a list whithin itself.

## Then , I have to mention that if you want to mix a few feature extraction 
# mehtods together, you have to use FeatureUnion module. otherwise, pipeline
# is enough for one feature extractor.


# Now , we are going to first do some data prepration and then some feature
# extraction on the data


# An empty workspace for data prepration
data_works = []

# For scaling data in between 0 and 1
data_works.append(('Scale' , StandardScaler()))

# For normalizing the data to have 0 mean and 1 std
data_works.append(('normal' , Normalizer()))




# An empty workspace for features
feature_works = []

# Mighty PCA
feature_works.append (('PCA', PCA(n_components = 9)))

# # Select K Best based on scoring_funcion chi2 (FOR CLASSIFICATION ONLY)
# feature_works.append (('SelectBest' , SelectKBest(score_func = chi2 , k=6)))


# An empty workspace for models
modeling = []

# Appending every algorithm with a nick name to the list to make a tuple of 
# the nick name and the algorithm itself
modeling.append(('LinR',LinearRegression()))
modeling.append(('LASSO',Lasso()))
modeling.append(('KNN',KNeighborsRegressor()))
modeling.append(('Tree',DecisionTreeRegressor()))
modeling.append(('Elastic',ElasticNet()))
modeling.append(('SVM',SVR()))
modeling.append(('MLP',MLPRegressor(max_iter = 1000)))
modeling.append(('RForest',RandomForestRegressor()))
modeling.append(('GBR',GradientBoostingRegressor()))
modeling.append(('ExtraTR',ExtraTreesRegressor()))
modeling.append(('ADABoost',AdaBoostRegressor()))


# Doing a pipeline babe !!




# An empty space for pipeline1 to hold all procedures
pipelines = []

# The first approach is : .append( 'Name of procedure' , Pipeline[all you need])
# First, we want Scaling for data preparation, PCA for feature extraction, and
# all algorithms on the data
# Here, we need the whole (('name',algo)) tuple from "modeling" list, so we do
# as follows
for i,x in enumerate(modeling) :
    pipelines.append(('Scaled,PCA,All-Algos' , Pipeline([data_works[0] , feature_works[0] , modeling[i]])))
    # pipelines.append(('Scaled, All-Algos' , Pipeline([data_works[0] , modeling[i]])))


# Now, Normalizing with PCA for all algos
for i,x in enumerate(modeling) :
    pipelines.append(('Normal,PCA,All-Algos' , Pipeline([data_works[1] , feature_works[0] , modeling[i]])))
    # pipelines.append(('Normal,All-Algos' , Pipeline([data_works[1] , modeling[i]])))




# An empty place for results
results = []

# An empty place for names
names = []


for name , model in pipelines :
    
    kfold = KFold(n_splits = 10 , random_state = 7)
    cv_result = cross_val_score(model , xtrain , ytrain , cv=kfold , scoring = 'neg_mean_squared_error')
    results.append(cv_result)
    names.append(name)
    
    

# An empty list to create algorithm names since we can't have their names...
# it's inside a complecated tuple and we have to have all 11 algorithms' names
# to be repeated for 4 times.
Algo_names = []

for i in range(2):
    for name , model in modeling:
        Algo_names.append(name)
    
    
# Printing resluts    
for i in range(len(names)):
    
    print(f'\n The {names[i]} with Algorithm {Algo_names[i]} had : {results[i].mean() } % as mean of the results of CV_Scoring based on "NegMSE" and {results[i].std() } % as std of the results \n')


# Box plotting results
fig = plt.figure()
fig.suptitle('Pipline Algorithms')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()



# Compare a little on Mean Squared Metric (Just like the cross val... But this
# time on real test data)
counter = 0
index_counter = -1
for name , model in pipelines:
    tmp = model
    tmp.fit(xtrain , ytrain)
    predictions = tmp.predict(xtest)
    counter +=1
    index_counter +=1
    print(f'\n {counter} MSE of {name} Algorithm {Algo_names[index_counter]} :\n {mean_squared_error(ytest , predictions)} \n')



# An algorithm to review results sorted by the amount of MSE in an ascending 
# mode
resulterman = []
for ind , item in enumerate(results):
    resulterman.append([names[ind] + ' ' + Algo_names[ind] , abs(results[ind].mean())])

# Sorting
sorter =  sorted(resulterman , key = lambda x:x[-1])

result_presenter_2 = []
for ind , item in enumerate(sorter):
    result_presenter_2.append([sorter[ind][0] , sorter[ind][1]])


## Save this result_presenter cause t's the best
filename = 'result_presenter_2'
f = open(filename,'wb')
pickle.dump(result_presenter_2 , f)
f.close()

#%% Now, let's find out which procedures were the bests


# Without pipeline procedures
filename = 'result_presenter_1'
f=open(filename,'rb')
no_pipelines = pickle.load(f)
f.close()

# With pipeline procedures
filename = 'result_presenter_2'
f=open(filename,'rb')
with_pipelines = pickle.load(f)
f.close()

#%% So , the best algo is selected.

# Extra Tree without anything else

# So , now we want to know how many trees are suitable for our 
# Extra-Tree-Regressor

# Select only one searcher

### These search algorithms would use cross-val to estimate best parameters.
### By fitting the algo to the train data, you can get best params and by
### predicting the algo on test data.
model = ExtraTreesRegressor()

kfold = KFold(n_splits = 10 )

grid1 = dict(n_estimators = np.array(range(100,1000,10)))

# grid2 = dict(n_estimators = np.random.randn())

searcher1 = GridSearchCV(estimator = model , param_grid = grid1 , scoring = 'neg_mean_squared_error' , cv = kfold)

# searcher2 = RandomizedSearchCV(estimator = model , param_distributions = grid2 , scoring = 'neg_mean_sqared_error' , n_iter = 200)

search_results1 = searcher1.fit(xtrain , ytrain)

# search_results2 = searcher2.fit(xtrain , ytrain)

print(f'\n Best Score is:\n {search_results1.best_score_}\n')

print(f'\n Best number of trees is :\n{search_results1.best_estimator_.n_estimators}\n')

print(f'\n Like above, this command will also show best parameter (number of trees):\n{search_results1.best_params_}')

mean = search_results1.cv_results_['mean_test_score']
stds = search_results1.cv_results_['std_test_score']
params = search_results1.cv_results_['params']

for x,y,z in zip(mean , stds , params):
    print(f'\n\n\n mean:{x}\n std:{y}\n parameter:{z}')


ytrue , ypredict = ytest , search_results1.predict(xtest)

final_eval = mean_squared_error(ytrue , ypredict)

print(f'\ninal_eval = {final_eval}\n')

#%% Using the optimized parameter in our model to estimate the data

## we wanna test that "final_eval" is actually the test score we can get by
## applying the main algorithm on the data itself.

finalmodel = ExtraTreesRegressor(n_estimators = search_results1.best_estimator_.n_estimators)

kfold = KFold(n_splits = 10) #, random_state = 7)

cross_val = cross_val_score(finalmodel , xtrain , ytrain , scoring = 'neg_mean_squared_error' , cv=kfold)

print(f'\n mean of results on training data : {cross_val.mean()}\n')

finalmodel.fit(xtrain , ytrain)

predicttion = finalmodel.predict(xtest)

final_res = mean_squared_error(ytest , predicttion)

print(f'final_score = {final_res}')



print('\n Yes... We can understand that Gridsearch.CV.fit on the training data and .predict on the test data is somehow close to fit and predict the main model with best parameter on the data.\n')
