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
# test_file = 'C:\\Users\\Digi_2\\Documents\\GitHub\\CAREEN\\temp_folder_for_results\\Machine_Learning\\INPUT_classification\\input_class_test.txt'
# train_file = 'C:\\Users\\Digi_2\\Documents\\GitHub\\CAREEN\\temp_folder_for_results\\Machine_Learning\\INPUT_classification\\input_class_train.txt'
# output_directory = 'C:\\Users\\Digi_2\\Documents\\GitHub\\CAREEN\\temp_folder_for_results\\Machine_Learning\\OUTPUT'

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
    
    
    # generations=1
    # population_size=20
    # mutation_rate=0.9
    # crossover_rate=0.1
    # scoring="balanced_accuracy"
    # cv=2
    # max_time_mins=60
    # max_eval_time_mins=10
    # early_stop=2
    
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

    with open(output_directory + "\\features_file.txt", "r") as file:
        features2include = [line.strip().split(',') for line in file]    
    features=pcd_training[features2include[0]]
    #features_train = MinMaxScaler().fit_transform(features)
    features_train=features
    labels_evaluation=pcd_testing[labels2include]
    features=pcd_testing[features2include[0]]
    #features_evaluation = MinMaxScaler().fit_transform(features)
    features_evaluation=features
    X_test=features_evaluation
    y_test1=labels_evaluation.to_numpy()

    X_train=features_train
    y_train=labels_train.to_numpy()
    
    # # Asegúrate de que las características (X) sean de tipo float64
    # X_train = X_train1.astype('float64')
    # X_test = X_test1.astype('float64')
    
    # # Asegúrate de que las etiquetas (y) sean de tipo int32
    # y_train = y_train1.astype('int32')
    # y_test = y_test1.astype('int32') 
    
    pipeline_optimizer = TPOTClassifier(
                                        generations=generations,
                                        population_size=population_size,
                                        mutation_rate=mutation_rate,
                                        crossover_rate=crossover_rate,
                                        scoring=scoring,
                                        cv=cv,
                                        max_time_mins=max_time_mins,
                                        max_eval_time_mins=max_eval_time_mins,
                                        early_stop=early_stop,
                                        random_state=None,
                                        verbosity=2,
                                        n_jobs=1,
                                        use_dask=True
                                        )
       
    # y_train = y_train.astype(int)
    # y_test = y_test.astype(int)
    # y_test = labels_train.to_numpy().astype(int)
    # y_train = labels_train.to_numpy().astype(int)

    pipeline_optimizer.fit(X_train, y_train)
    
    # y_test = y_test1.to_numpy().ravel()
    
    # print(pipeline_optimizer.score(X_test, y_test))
    # Serialize the best pipeline (model) into a .pkl file
    joblib.dump(pipeline_optimizer.fitted_pipeline_,os.path.join(output_directory, 'best_pipeline.pkl'))  # Replace 'fitted_pipeline_' with the appropriate attribute
    
    # Load the pickled model
    loaded_model = joblib.load(os.path.join(output_directory, 'best_pipeline.pkl'))

    # Assuming 'new_data' contains the data you want to predict on
    y_pred = loaded_model.predict(X_test)
    y_pred_str = map(str, y_pred)
    with open(os.path.join(output_directory, 'predictions.txt'), "w") as output_file:
        output_file.write(' '.join(y_pred_str))
    

if __name__=='__main__':
	main()