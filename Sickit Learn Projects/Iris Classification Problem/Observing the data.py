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

#%% Peak at data

print(dataset.head(20))

#%% Data Types

print(dataset.dtypes)
        
#%% Data Dimension

print(f'Dataset dimension is = {dataset.shape}')
        
        
#%% Numerical information on the dataset

print(dataset.describe())
    
#%% Scatter Plot Matrix

# With pandas
scatter_matrix(dataset)

# With seaborn
sns.pairplot(dataset)

# All attributes in relation with "class" for classification possibility checking
sns.pairplot(dataset,hue='class')

#%% Check on "class" attribute's classes

print(dataset.groupby('class').size())


#%% Correlations on data by numbers   

print(dataset.corr(method = 'pearson'))


#%% Visualizing correlation bar

# Using pyplot

# Creating main figure with its name
fig = plt.figure('Pyplot Correlation Visualization')

# Settings for subplot in figure
ax = fig.add_subplot(111)

# Settings for correlation showing in figure
cax = ax.matshow(dataset.corr(method = 'pearson'), vmin =-1, vmax = 1, interpolation = 'none')

# Applying settings to figure's colorbar
fig.colorbar(cax)

# Arranging an array for making x-axis and y-axis sectioning to start from 0
# not 1 and make it possible to name each section.
# If you want to name your sections, you must do all lines of codes in here
# related to ax.
ticks = np.arange(0,4,1)

# Fixing the sections numbering
ax.set_xticks(ticks)

# Fixing the sections numbering
ax.set_yticks(ticks)

# Setting x-axis names
ax.set_xticklabels(names)

# Setting y-axis names
ax.set_yticklabels(names)




# Using seaborn library as sns
plt.figure('Seaborn Correlation Visualization')
sns.heatmap(dataset.corr(method='pearson'), vmin = -1, vmax = 1, annot=True)



#%% Dataset skew (Larger negative values --> Left skewed data)
                #(Larger positive values --> Right skewed data)
                #(Close to zero values --> Less skewed data)

print(dataset.skew())


#%% Histogram of data

dataset.hist()

#%% Density Plotting

dataset.plot(kind='density', subplots = True, layout = (2,2), title = names[0:4] , sharex= False, sharey = False)

#%% Box plot (Red line is for Median (middle of the data))
            #(Box is between 25th percentile and 75th percentile of data (middle 50 percentiles))
            #(Dots outside the box are outliers)
            
dataset.plot(kind='box', subplots = True, layout = (2,2) , sharex= False, sharey = False)


