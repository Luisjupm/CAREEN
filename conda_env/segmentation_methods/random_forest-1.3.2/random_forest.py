# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:36:37 2023

@author: Digi_2
"""

import argparse

from pandas import pd
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--te',type=str,help='Path for the test file')
    parser.add_argument('--tr',type=str,help='Path for the train file')
    parser.add_argument('--o',type=str,help='Path for the output file')
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

    #Store in a Pandas dataframe the content of the file on the google drive
    pcd_train=pd.read_csv(test_file,delimiter=' ')
    pcd_train
    #Store in a Pandas dataframe the content of the file on the google drive
    pcd_evaluation=pd.read_csv(train_file,delimiter=' ')
    pcd_evaluation
    #Clean the dataframe, and drop all the line that contains a NaN (Not a Number) value.
    pcd_train.dropna(inplace=True)
    pcd_evaluation.dropna(inplace=True)
    #Create training and testing
    labels_train=pcd_train[labels2include]
    features=pcd_train[features2include]
    #features_train = MinMaxScaler().fit_transform(features)
    features_train=features
    labels_evaluation=pcd_evaluation[labels2include]
    features=pcd_evaluation[features2include]
    #features_evaluation = MinMaxScaler().fit_transform(features)
    features_evaluation=features
    X_test=features_evaluation
    y_test=labels_evaluation.to_numpy()

    X_train=features_train
    y_train=labels_train.to_numpy()
    
    for ne, md, mf, bt, c, ms, mns, mwf, s, nj in list(itertools.product(n_estimators, max_depth, max_features, bootstrap, criterion, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, scoring, n_jobs)):
          if bt==False:
            obb=False
            rf_classifier = RandomForestClassifier(
                                                    n_estimators = ne,
                                                    max_depth=md,
                                                    n_jobs=n_jobs,
                                                    max_features=mf,
                                                    random_state=42,
                                                    bootstrap=bt,
                                                    criterion=c,
                                                    min_samples_split=ms,
                                                    min_samples_leaf=mns,
                                                    min_weight_fraction_leaf=mwf,
                                                    scoring=s,
                                                    n_jobs=nj                 
                                                    )
          else:
            obb=True
            rf_classifier = RandomForestClassifier(
                                                    n_estimators = ne,
                                                    max_depth=md,
                                                    n_jobs=n_jobs,
                                                    max_features=mf,
                                                    random_state=42,
                                                    oob_score=True,
                                                    bootstrap=bt,
                                                    criterion=c,
                                                    min_samples_split=ms,
                                                    min_samples_leaf=mns,
                                                    min_weight_fraction_leaf=mwf,
                                                    scoring=s,
                                                    n_jobs=nj                 
                                                    )

if __name__=='__main__':
	main()