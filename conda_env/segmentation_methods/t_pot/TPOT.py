# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:29:32 2023

@author: Digi_2
"""

import argparse

import os
import subprocess
import sys

# from sklearn.externals import joblib  # For scikit-learn versions < 0.23
# For newer versions of scikit-learn, use:
import joblib

from sklearn.ensemble import *
from sklearn.kernel_approximation import *
from sklearn.naive_bayes import *
from sklearn.neural_network import *

import pandas as pd
import numpy

from tpot import TPOTClassifier
from tpot import *
import tpot

#%% DEFINING INPUTS OF CMD
current_directory=os.path.dirname(os.path.abspath(__file__))
temp_folder=os.path.join(current_directory,'..','temp')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--te',type=str,help='Path for the test file')
    parser.add_argument('--tr',type=str,help='Path for the train file')
    parser.add_argument('--o',type=str,help='Path for the output file')
    parser.add_argument('--f',type=str,help='Path for the features')
    parser.add_argument('--ge',type=str,help='Path for the generations')
    parser.add_argument('--ps',type=str,help='Path for the population size')
    parser.add_argument('--mr',type=str,help='Path for the mutation rate')
    parser.add_argument('--cr',type=str,help='Path for the crossover rate')
    parser.add_argument('--cv',type=str,help='Path for the cross validation')
    parser.add_argument('--mtm',type=str,help='Path for the max time mins')
    parser.add_argument('--metm',type=str,help='Path for the max evaluation time mins')
    parser.add_argument('--ng',type=str,help='Path for the number of generations without improvement')
    parser.add_argument('--s',type=str,help='Path for the scoring')
    
    args=parser.parse_args()  

    
    test_file=args.te
    print("Test file located in " + test_file)
    train_file=args.tr
    print("Train file located in " + train_file)
    output_directory=args.o
    print("Output directory is " + output_directory)
    features2include=args.f
    print("Features to include = " + features2include)
    generations=args.ge
    print("Value chosen for generations = " + generations)
    population_size=args.ps
    print("Value chosen for pupulation size = " + population_size)
    mutation_rate=args.mr
    print("Value chosen for mutation rate = " + mutation_rate)
    crossover_rate=args.cr
    print("Value chosen for crossover rate = " + crossover_rate)
    cv=args.cv
    print("Value chosen for cross validation = " + cv)
    max_time_mins=args.mtm
    print("Value chosen for max time mins = " + max_time_mins)
    max_eval_time_mins=args.metm
    print("Value chosen for max eval time mins = " + max_eval_time_mins)
    early_stop=args.ng
    print("Value chosen for early stop = " + early_stop)
    scoring=args.s
    print("Scoring chosen = " + scoring)
  
   
    #Store in a Pandas dataframe the content of the file
    pcd_training=pd.read_csv(train_file,delimiter=' ')
    
    #Store in a Pandas dataframe the content of the file 
    pcd_testing=pd.read_csv(test_file,delimiter=' ')
    
    labels2include= ['Classification']
    #Clean the dataframe, and drop all the line that contains a NaN (Not a Number) value.
    pcd_training.dropna(inplace=True)
    pcd_testing.dropna(inplace=True)
    #Create training and testing
    labels_train=pcd_training[labels2include]

    with open(output_directory + "\\features.txt", "r") as file:
        features2include = [line.strip().split(',') for line in file]    
    features=pcd_training[features2include[0]]
    features_train=features
    labels_evaluation=pcd_testing[labels2include]
    features=pcd_testing[features2include[0]]
    features_evaluation=features
    X_test=features_evaluation
    y_test1=labels_evaluation.to_numpy()

    X_train=features_train
    y_train=labels_train.to_numpy()
    # Assuming y is a column vector or a 2D array
    y_train_reshaped = np.ravel(y_train)
    
    pipeline_optimizer = TPOTClassifier(
                                        generations=int(generations),
                                        population_size=int(population_size),
                                        mutation_rate=float(mutation_rate),
                                        crossover_rate=float(crossover_rate),
                                        scoring=scoring,
                                        cv=int(cv),
                                        max_time_mins=int(max_time_mins),
                                        max_eval_time_mins=int(max_eval_time_mins),
                                        early_stop=int(early_stop),
                                        random_state=None,
                                        verbosity=2,
                                        n_jobs=1,
                                        use_dask=False
                                        )
       

    pipeline_optimizer.fit(X_train, y_train_reshaped)
    
    # Serialize the best pipeline (model) into a .pkl file
    joblib.dump(pipeline_optimizer.fitted_pipeline_,os.path.join(output_directory, 'best_pipeline.pkl'))  # Replace 'fitted_pipeline_' with the appropriate attribute
    
    # Load the pickled model
    loaded_model = joblib.load(os.path.join(output_directory, 'best_pipeline.pkl'))

    # Assuming 'new_data' contains the data you want to predict on
    y_pred = loaded_model.predict(X_test)
    # y_pred_str = map(str, y_pred)

    pcd_testing_subset = pcd_testing[['X', 'Y', 'Z']].copy()
    pcd_testing_subset['Predictions'] = y_pred
    # Saving the DataFrame to a CSV file
    pcd_testing_subset.to_csv(os.path.join(output_directory, 'predictions.txt'), index=False)

if __name__=='__main__':
	main()