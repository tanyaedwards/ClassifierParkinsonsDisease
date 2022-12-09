#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 16:54:56 2020

@author: tanyaedwards
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from matplotlib.colors import ListedColormap
from sklearn.model_selection import cross_val_score


# 0 = Healthy
# 1 = Parkinson's Disease

#plot correlation matrix to explore dependency between features
def plot_correlation(data):
    rcParams['figure.figsize'] = 15, 20
    fig = plt.figure()
    sns.heatmap(data.corr(), annot=True, fmt=".2f")
    plt.show()
    fig.savefig('Figures/corr_matrix.png')
    
#Plot features densities depending on the outcome values
def plot_densities(data):
    rcParams['figure.figsize'] = 15, 20
    # create two dataframes based on class type
    class_0 = data[data['class'] == 0]
    class_1 = data[data['class'] == 1]

    # create figure
    col_names = list(data.columns)
    n_subplots = len(col_names[1:])
    fig, axs = plt.subplots(n_subplots, 1)
    fig.suptitle('Densities of Speech Features for Healthy and Unhealthy Patients')
    plt.subplots_adjust(left = 0.25, right = 0.9, bottom = 0.1, top = 0.95,
                        wspace = 0.2, hspace = 0.9)
    
    # plot density curves
    for col in col_names[1:]: 
        ax = axs[col_names.index(col)-1]
        class_0[col].plot(kind='density', ax=ax, subplots=True, 
                                    sharex=False, color="red", legend=True,
                                    label=col + ' for Class = 0')
        class_1[col].plot(kind='density', ax=ax, subplots=True, 
                                     sharex=False, color="green", legend=True,
                                     label=col + ' for Class = 1')
        ax.set_xlabel(col + ' values')
        ax.set_title(col + ' density')
        ax.grid('on')
    plt.show()
    fig.savefig('Figures/density_curves.png')

#Plotting example boundary for k with two features
def plot_boundaries(X, y):
    #Test, train and split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 41)

    #colour maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold  = ListedColormap(['#FF0000', '#0000FF'])

    rcParams['figure.figsize'] = 8, 5
    n_neighbors = 7
    step_size   = .02  
    fig = plt.figure()
    i=1
    for weights in ['uniform', 'distance']:
        KNN_classifier = KNeighborsClassifier(n_neighbors, weights=weights)
        KNN_classifier.fit(X_train, y_train)

        # Assigning colour to each mesh area
        x_min, x_max = X[:, 0].min() -0.1, X[:, 0].max() +0.1
        y_min, y_max = X[:, 1].min() -0.1, X[:, 1].max() +0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size),
                             np.arange(y_min, y_max, step_size))
        Z = KNN_classifier.predict(np.c_[xx.ravel(), yy.ravel()])

        # Plot mesh
        Z = Z.reshape(xx.shape)
        #fig = plt.figure()
        plt.subplot(1,2,i)
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot feature points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)   
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        #plt.title("Healthy/Unhealthy classification (k = %i, weights = '%s')" % (n_neighbors, weights))
        plt.title("k = %i, weights = '%s'" % (n_neighbors, weights))
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()    
        i += 1
    fig.savefig('weights.png')


    
#THE DATA (Parkinson's Disease)
Parkinson_df_master = pd.read_csv('Data/pd_speech_features.csv')
print(Parkinson_df_master.head())


#CHOOSING FEATURES
Parkinson_df = Parkinson_df_master.iloc[:,1:755]
#Parkinson_df['class'] = Parkinson_df_master['class']
col_class = Parkinson_df.pop('class')
Parkinson_df.insert(0, 'class', col_class)
corr_df = Parkinson_df.corr().abs().reset_index()
corr_best = corr_df[(corr_df['class'] > 0.30)]

#Plot correlation matrix of 10 features
Parkinson_df_best = Parkinson_df[corr_best['index'].tolist()[:10]]
plot_correlation(Parkinson_df_best) 

#Features_df = Parkinson_df.iloc[:,2:100]
corr_best_list = corr_best['index'].tolist()[1:]
Features_df = Parkinson_df[corr_best_list]
Labels_df = Parkinson_df['class']

# plot density curves
#dens_best = corr_df[(corr_df['class'] > 0.35)]
dens_best = corr_df.sort_values('class')
dens_list = dens_best['index'].tolist()[-5:]
dens_list = dens_list[::-1]
Parkinson_df_dens = Parkinson_df[dens_list]
plot_densities(Parkinson_df_dens)

#Normalize Data
scaler = MinMaxScaler()
Features_array = scaler.fit_transform(Features_df)
Features_df = pd.DataFrame(Features_array)


#BUILD MODEL
#TEST_TRAIN_SPLIT
train_datap, test_datap, train_labels, test_labels = train_test_split(Features_df, Labels_df, 
                                                                    test_size=0.2, random_state=1)



#Finding best K
cv_scores_k = []
cv_scores = []
for k in range(1,81,2):
    KNN_classifier = KNeighborsClassifier(n_neighbors = k, weights='distance')
    KNN_classifier.fit(train_datap, train_labels)
       
    scores = cross_val_score(KNN_classifier, train_datap, train_labels, cv=10, scoring='accuracy')
    cv_scores_k.append([scores.mean(), k])
    cv_scores.append(scores.mean())
    
#print(cv_scores_k)
missclass_error = [1 - x for x in cv_scores]
#print(missclass_error)

#k_best_test = max(score_test_list)
k_best = max(cv_scores_k)
print('Best K from Cross Validation: ', k_best)

plt.figure()
plt.plot(range(1,81,2), missclass_error)
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.title('Parkinson\'s Speech Classifier Accuracy')
plt.show()

#train best fit model
KNN_classifier_best = KNeighborsClassifier(n_neighbors = k_best[-1])
KNN_classifier_best.fit(train_datap, train_labels)


#MODEL ACCURACY (TEST DATA)
score_test = KNN_classifier_best.score(test_datap, test_labels)
print('Score Test: {}'.format(score_test))

predict = KNN_classifier_best.predict(test_datap)
print('WITH Best K')
print('')
print(confusion_matrix(test_labels,predict))
print('')
print(classification_report(test_labels,predict))

plot_confusion_matrix(KNN_classifier_best, test_datap, test_labels)
print(len(test_datap))

#PREDICT PATIENT READINGS


#PLOT distance 2 class plot
Features_array = np.array(Features_df.iloc[:, [1,2]])
Labels_df_ = Parkinson_df[['class']]
Labels_array = np.array(Labels_df_.iloc[:, 0])
plot_boundaries(Features_array, Labels_array)
