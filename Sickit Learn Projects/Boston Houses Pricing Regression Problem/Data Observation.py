# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 21:24:22 2020

@author: Mahdi

This template is all about loading data, visualizations and getting statistical
numbers out of it.

The examole is done by IRIS dataset.
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
        dataset = read_csv(f[i] , delim_whitespace = True ,names=names)

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
    
#%% Scatter Plot Matrix (Run line by line)

# With pandas
scatter_matrix(dataset)

# With seaborn
sns.pairplot(dataset)

# All attributes in relation with "class" for classification possibility checking
sns.pairplot(dataset,hue='MEDV')

#%% Check on "class" attribute's classes

print(dataset.groupby('MEDV').size())


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



#%% Dataset skew (Larger negative values --> Left skewed data
                #                           , meaning it's left taled and the
                #                             concentration of data is on the
                #                             right)
                #(Larger positive values --> Right skewed data 
                #                           , meaning it's right taled and the
                #                             concentration of data is on the
                #                             left)
                #(Close to zero values --> Less skewed data)

print(dataset.skew())


#%% Histogram of data

dataset.hist()

#%% Density Plotting

dataset.plot(kind='density', subplots = True, layout = (7,2), title = names[0:4] , sharex= False, sharey = False)

#%% Box plot (Red line is for Median (middle of the data))
            #(Box is between 25th percentile and 75th percentile of data (middle 50 percentiles))
            #(Dots outside the box are outliers)
            
dataset.plot(kind='box', subplots = True, layout = (2,7) , sharex= False, sharey = False)
