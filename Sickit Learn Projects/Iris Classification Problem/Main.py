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

# Good for splitting train and test sets
from sklearn.model_selection import train_test_split

# Logistic Regression function from sklearn
from sklearn.linear_model import LogisticRegression

# Importing cross_val_score to be able to score more than "model.score" that
# is for the "model" itself and only measures accuracy by 
# "correct_predictions / total samples"
from sklearn.model_selection import cross_val_score

# You can do K-Folds on data rather than "train_test_split"
# to devide it more than one part and make cross_vals better and better
from sklearn.model_selection import KFold

# Confusion matrix is F1_score , recall, precision story
# It's for multiclass
from sklearn.metrics import confusion_matrix

# The best classification metric ever
from sklearn.metrics import classification_report

# The accuracy metric
from sklearn.metrics import accuracy_score

# Importing some other training algorithms :

from sklearn.svm import SVC

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

#%% Address Definings

dataset_address = 'E:\\My Projects\\2019 Works\\Datasets\\For Begginers\\IRIS'

code_directory = 'E:\\My Projects\\2019 Works\\Python\\Spyder\\Real Work Tests\Sickit Learn Projects\\Iris Classification Problem'

#%% Data Information (Feature info mostly)

'''

Attribute Information:
   1. sepal length in cm
   2. sepal width in cm
   3. petal length in cm
   4. petal width in cm
   5. class: 
      -- Iris Setosa
      -- Iris Versicolour
      -- Iris Virginica

Number of Instances: 150 (50 in each of three classes)

Class Distribution: 33.3% for each of 3 classes.

'''




#%% Loading Dataset

# Going to dataset address
os.chdir(dataset_address)

# Selecting names for attributes (features)
names = ['sepal length','sepal width','petal length','petatl width','class']

# See what's inside the directory
f = os.listdir()

#Loading the dataset
for i,item in enumerate(f):
    if f[i] == 'iris.data':
        dataset = read_csv(f[i],names=names)

# Let's get back to our code's directory
os.chdir(code_directory)

#%% Prepare data for learning

# Getting rid of explanations and to have only numbers
array = dataset.values

# Input
X = array[:,0:4]

# Target
Y = array[:,4]

# Setting Test_Sets's size (Usually 33% of the dataset)
test_size = 0.33

#  Seeding the random selection to be repeatable
seed = 7

# Seperating Training_Sets and Test_Sets
xtrain , xtest , ytrain , ytest = train_test_split(X , Y , test_size = test_size , random_state = seed)

# Seeing shapes of each
print(f'X_Train Shape : {xtrain.shape} \nY_Train Shape : {ytrain.shape} \nX_Test Shape : {xtest.shape} \nY_Test Shape : {ytest.shape}')

#%% Machine Learning Models Practice

# Selecting the model to fit on the training data
model = LogisticRegression(max_iter = 1000)

# Fitting the model on the data
model.fit(xtrain,ytrain)

# Calculating accuracy manualy
# Prediction (Hypothesis)
predictions = model.predict(xtest)

# To get the number of correct predictions
correct_num = 0
for index , state in enumerate(ytest):
    if state == predictions[index]:
        correct_num += 1
# Total number of samples     
total_num = len(ytest)

# Accuracy is : 
print(f'Manually Computed Accuracy : {(correct_num / total_num) * 100} %')

# Sklearn Accuracy
result_score = model.score(xtest,ytest)

# Printing accuracy
print(f'Sklearn Computed Accuracy : {result_score * 100} %')

# Obtaining and printing model parameters
result_parameters = model.get_params(model)


# Observing a few predictions: 
print (f'First 20 Predicted Cases: \n {predictions[0:6]} \n')

# Observing the related real targets
print (f'First 20 Cases\' Real Targets: \n {predictions[0:6]} \n')

### These were the "model" itself functions and metrics, not any other 
### external library or something.

#%% K-Fold Cross Val

## Most of the times it's much better to do a k-fold-cross-validation rather 
## than a simple train , test split... 

# Number of folds
num_folds = 10

# Kfold settings
kfold = KFold(n_splits = num_folds , random_state = seed)

# New Model
model2 = LogisticRegression(max_iter = 1000)

# cross_val on the model instead of model.score()
# Remember that you have to pass X and Y not xtrain and ytrain since it's 
# doing a new kind of segmentation.
cv_results = cross_val_score(model2 , X , Y , cv = kfold)

# We have 10 pairs of cv-scores since we had 10 folds. 1-fold means to split
# the dataset into one trainset and one testset, then gain cv-scores. 2-fold
# means that we once split data into trainset and testset and gain one cv-score
# and then, for the second time, we reverse their places with each other so the
# last trainset becomes new testset and last testset becomes new trainset.
# for 3-folds, we have 3 groups that two are for train and third for test. then
# they would change their places until every one of them has been a testset once.
# It's the same routine for k-folds

# Printing mean of results and std of them
print(f'CV-Accuracy mean percent :{cv_results.mean() * 100} % \n')
print(f'CV-Accuracy std percent :{cv_results.std() * 100} % \n')


#%% A little on metrics

# After fitting the "model" in first example, come here :)

# Calculating confusion matrix: 
conf = confusion_matrix(ytest , predictions)

print (f"Confusion Matrix : \n {conf} \n")


# Classification Report (Best Classification Metric)

report = classification_report(ytest , predictions)
print(f"Classification Report : \n {report} \n")


#%% Manual Pipeline Making Procedure. 
   # Automatic pipelines are for each model like below! As you can see below, 
   # the pipeline is runing for each model and does a series of steps like 
   # scaling the data, feature extraction etc on the data and then feed it 
   # to the final model. Then using this procedure to do an automatic algorithm.
   # But, since we didn't have scaling and feature extraction and etc, we just
   # created the list of models and set the training. 
   # If you used scaling and feature extraction and etc and used pipelines, the
   # rest of the procedure is identical to ours. It is necessary to import the
   # library first by "from sklearn.pipeline import Pipeline" code and then use
   # Pipeline if you need it.
   
# pipelines = []
# pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
# pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO',Lasso())])))
# pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN',ElasticNet())])))
# pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',KNeighborsRegressor())])))
# pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',DecisionTreeRegressor())])))
# pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))

# An empty list of models
models = []

# Appending every algorithm with a nick name to the list to make a tuple of 
# the nick name and the algorithm itself
models.append(('LR',LogisticRegression(max_iter = 1000)))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))
models.append(('MLP',MLPClassifier(max_iter = 1000)))


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
    cv_result = cross_val_score(model , xtrain , ytrain , scoring = 'accuracy' , cv = kfold )
    
    results.append(cv_result)
    names.append(name)
    
# Printing the results    
for i in range(len(names)):
    
    print(f'\n The {names[i]} Algorithm had : {results[i].mean() * 100} % as mean of the results and {results[i].std() * 100} % as std of the results \n')
    

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
    print(f'\n Accuracy of {name} Algorithm :\n {accuracy_score(ytest , predictions) * 100} % \n')
    print(f'\n Classification Report of {name} Algorithm :\n {classification_report(ytest , predictions)} \n')



