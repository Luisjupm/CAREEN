# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:36:37 2023

@author: Digi_2
"""

import argparse
import joblib

import os
import subprocess
import sys

import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from matplotlib import pyplot as plt
import pickle
import seaborn as sns
current_directory=os.path.dirname(os.path.abspath(__file__))
temp_folder=os.path.join(current_directory,'..','temp')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--te',type=str,help='Path for the test file')
    parser.add_argument('--tr',type=str,help='Path for the train file')
    parser.add_argument('--o',type=str,help='Path for the output file')
    parser.add_argument('--f',type=str,help='Path for the features')
    parser.add_argument('--ne',type=str,help='Path for the number of estimators')
    parser.add_argument('--c',type=str,help='Path for the criterion')
    parser.add_argument('--md',type=str,help='Path for the maximum depth of the threes')
    parser.add_argument('--ms',type=str,help='Path for the minimum number of simples required to Split the internal node')
    parser.add_argument('--mns',type=str,help='Path for the minimum number of samples required to be a leaf node')
    parser.add_argument('--mwf',type=str,help='Path for the minimum weight fraction of the total sum of weights')
    parser.add_argument('--mf',type=str,help='Path for the maximum number of features')
    parser.add_argument('--bt',type=str,help='Path for the bootstrap')
    parser.add_argument('--s',type=str,help='Path for the scoring')
    parser.add_argument('--nj',type=str,help='Path for the number of jobs')
    
    args=parser.parse_args()  
    
    
    test_file=args.te
    print("Test file located in " + test_file)
    train_file=args.tr
    print("Train file located in " + train_file)
    output_directory=args.o
    print("Output directory is " + output_directory)
    features2include=args.f
    print("Features to include = " + features2include)
    n_estimators=args.ne
    print("Number of estimators = " + n_estimators)
    criterion=args.c
    print("Criterion chosen = " + criterion)
    max_depth=args.md
    print("Maximum depth of the tree = " + max_depth)
    min_samples_split=args.ms
    print("Minimum number of samples required to split an internal node = " + min_samples_split)
    min_samples_leaf=args.mns
    print("Minimum number of samples required to be at a leaf node = " + min_samples_leaf)
    min_weight_fraction_leaf=args.mwf
    print("Minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node = " + min_weight_fraction_leaf)
    max_features=args.mf
    print("Number of features to consider = " + max_features)
    bootstrap=args.bt
    print("Bootstrap = " + bootstrap)
    scoring=args.s
    print("Scoring = " + scoring)
    n_jobs=args.nj
    print("Number of jobs = " + n_jobs)
    
    
   
    labels2include=['Classification']

    #Store in a Pandas dataframe the content of the file
    pcd_training=pd.read_csv(train_file,delimiter=' ')
    #Store in a Pandas dataframe the content of the file
    pcd_testing=pd.read_csv(test_file,delimiter=' ')
    #Clean the dataframe, and drop all the line that contains a NaN (Not a Number) value.
    pcd_training.dropna(inplace=True)
    pcd_testing.dropna(inplace=True)
    #Create training and testing
    labels_training=pcd_training[labels2include]
    
    with open(output_directory + "\\features.txt", "r") as file:
        features2include = [line.strip().split(',') for line in file]    
    features=pcd_training[features2include[0]]

    #features_train = MinMaxScaler().fit_transform(features)
    features_training=features
    labels_testing=pcd_testing[labels2include]
    features=pcd_testing[features2include[0]]
    #features_evaluation = MinMaxScaler().fit_transform(features)
    features_testing=features
    X_test=features_testing
    y_test=labels_testing.to_numpy()

    X_train=features_training
    y_train=labels_training.to_numpy()
    
    #*******************BEST MODEL***************************************
    best_conf = {'ne' : 0, 'md' : 0, 'mf': 0, 'bt':0} 
    best_f1 = 0
    
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    min_samples_split = int(min_samples_split)
    min_samples_leaf = int(min_samples_leaf)
    min_weight_fraction_leaf=int(min_weight_fraction_leaf)
    n_jobs=int(n_jobs)
    
    for ne, md, mf, bt, c, ms, mns, mwf in list(itertools.product([n_estimators], [max_depth], [max_features], [bootstrap], [criterion], [min_samples_split], [min_samples_leaf], [min_weight_fraction_leaf])):
        bt = bt.lower() == 'true'
        if not bt:
            obb = False
        else:
            obb = True
    
        rf_classifier = RandomForestClassifier(
            n_estimators=ne,
            max_depth=md,
            max_features=mf,
            random_state=42,
            oob_score=obb,
            bootstrap=bt,
            criterion=c,
            min_samples_split=ms,
            min_samples_leaf=mns,
            min_weight_fraction_leaf=mwf,
            # scoring=s,
            n_jobs=-1
            )
    rf_classifier.fit(X_train,y_train.ravel())
    test_rf_predictions = rf_classifier.predict(X_test) 
    # Compute metrics and update best model
    acc = accuracy_score(y_test.ravel(), test_rf_predictions)
    f1 = f1_score(y_test.ravel(), test_rf_predictions, average='weighted')
    # Update best configuration
    if f1 > best_f1:                                                
       best_conf['ne'] = ne
       best_conf['md'] = md
       best_conf['mf'] = mf
       best_conf['bt'] = bt
       best_f1 = f1
    if obb==True:
      print('\tne: {}, md: {}, mf: {},  bt: {}- acc: {} f1: {} oob_score: {}'.format(ne, md,mf,bt, acc, f1, rf_classifier.oob_score_))
    else:
      print('\tne: {}, md: {}, mf: {},  bt: {}- acc: {} f1: {}'.format(ne, md,mf,bt, acc, f1))  
  
    print('Best parameters: ne: {}, md: {}, mf: {},  bt: {}'.format(best_conf['ne'], best_conf['md'], best_conf['mf'],best_conf['bt']))
    
    #*******************IMPORTANCE***************************************
    print('Parameters used for the final Random Forest classificator: ne: {}, md: {}, bt: {}'.format(best_conf['ne'], best_conf['md'],best_conf['bt']))
    if best_conf['bt']==False:
      rf_classifier = RandomForestClassifier(n_estimators = best_conf['ne'],max_depth= best_conf['md'],bootstrap=best_conf['bt'],n_jobs=n_jobs,random_state=42)
    else:
      rf_classifier = RandomForestClassifier(n_estimators = best_conf['ne'],max_depth= best_conf['md'],bootstrap=best_conf['bt'],n_jobs=n_jobs,random_state=42,oob_score=True)
    rf_classifier.fit(X_train,y_train.ravel())

    fi = pd.DataFrame({'feature': list(X_train),'importance': rf_classifier.feature_importances_}).sort_values('importance', ascending = True)
    fig, ax = plt.subplots(figsize=(15,20))
    ax.barh(fi['feature'], fi['importance'])
    fi.to_csv(output_directory+'/Feature_importance.csv', encoding = 'utf-8-sig') 
    plt.savefig(output_directory+'/Feature_importance.jpg')

    #*******************CONFUSION MATRIX***************************************
    
    fig, ax = plt.subplots(figsize=(5,5))
    test_rf_predictions = rf_classifier.predict(X_test)  
    sns.heatmap(confusion_matrix(y_test.ravel(),test_rf_predictions), annot=True,cmap='Blues',fmt='d')
    plt.savefig(output_directory+'/Confusion_matrix.jpg')
    #*******************MATRIZ CLASSIFICATION***************************************
    print(classification_report(y_test, test_rf_predictions,digits=3))

    #ESCRITURA MODELO FINAL
    pcd_testing['Predictions']=test_rf_predictions
    pcd_testing[['X','Y','Z','Predictions']].to_csv(os.path.join(output_directory, 'predictions.txt'), index=None)
    pickle.dump(rf_classifier, open(output_directory+"/random_forest.pkl", 'wb'))
    
    
if __name__=='__main__':
	main()