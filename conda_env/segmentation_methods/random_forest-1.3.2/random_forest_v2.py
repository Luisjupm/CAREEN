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
import yaml

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
import pickle
import matplotlib.pyplot as plt
from yellowbrick.classifier import ConfusionMatrix

def main():
    # Import all the parameters form the CMD
    parser = argparse.ArgumentParser()
    parser.add_argument('--i',type=str,help='Yaml configuration file')
    parser.add_argument('--o',type=str,help='Output_directory')    
    args=parser.parse_args() 
    
    # Read the configuration from the YAML file for the set-up
    with open(args.i, 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    test_file= config_data.get('INPUT_POINT_CLOUD_TESTING')
    train_file= config_data.get('INPUT_POINT_CLOUD_TRAINING')
    output_directory= config_data.get('OUTPUT_DIRECTORY')
    features2include_path= config_data.get('INPUT_FEATURES')
    n_estimators=config_data['CONFIGURATION']['ne']
    criterion=config_data['CONFIGURATION']['c']
    max_depth=config_data['CONFIGURATION']['md']
    min_samples_split=config_data['CONFIGURATION']['ms']
    min_samples_leaf=config_data['CONFIGURATION']['mns']
    min_weight_fraction_leaf=config_data['CONFIGURATION']['mwf']
    max_features=config_data['CONFIGURATION']['mf']
    bootstrap=config_data['CONFIGURATION']['bt']
    scoring=config_data['CONFIGURATION']['s']
    n_jobs=config_data['CONFIGURATION']['nj']  
    
    # There are an issue with the f1 score. This score only accepts 0 and 1 lables. So if you introduce 3 and 4 labels, for example, throws and error
    if scoring=="f1":
        scoring="f1_macro"
    
    # test_file=args.te
    print("Test file located in " + test_file)
    # train_file=args.tr
    print("Train file located in " + train_file)
    # output_directory=args.o
    print("Output directory is " + output_directory)
    # features2include=args.f
    print("Features to include = " + features2include_path)
    # n_estimators=args.ne
    print("Number of estimators = " + str(n_estimators))
    # criterion=args.c
    print("Criterion chosen = " + str(criterion))
    # max_depth=args.md
    print("Maximum depth of the tree = " + str(max_depth))
    # min_samples_split=args.ms
    print("Minimum number of samples required to split an internal node = " + str(min_samples_split))
    # min_samples_leaf=args.mns
    print("Minimum number of samples required to be at a leaf node = " + str(min_samples_leaf))
    # min_weight_fraction_leaf=args.mwf
    print("Minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node = " + str(min_weight_fraction_leaf))
    # max_features=args.mf
    print("Number of features to consider = " + str(max_features))
    # bootstrap=args.bt
    print("Bootstrap = " + str(bootstrap))
    # scoring=args.s
    print("Scoring = " + str(scoring))
    # n_jobs=args.nj
    print("Number of jobs = " + str(n_jobs))

   
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


    #*******************MATRIZ CLASSIFICATION***************************************
    print(classification_report(y_test, test_rf_predictions,digits=3))

    #ESCRITURA MODELO FINAL
    pcd_testing['Predictions']=test_rf_predictions
    pcd_testing[['X','Y','Z','Predictions']].to_csv(os.path.join(output_directory, 'predictions.txt'), index=None)
    pickle.dump(rf_classifier, open(output_directory+"/random_forest.pkl", 'wb'))
    
    # Load the pickled model 
    loaded_model = joblib.load(os.path.join(output_directory, 'random_forest.pkl'))
    
    # Prediction
    y_pred = loaded_model.predict(X_test)
    
    # Save the model to a file
    joblib.dump(loaded_model, os.path.join(output_directory, 'model.pkl'))
    
    #*******************CONFUSION MATRIX***************************************
    
    # Create the confusion matrix
    cm= ConfusionMatrix(loaded_model, cmap='Blues')
    cm.score (X_test,y_test)
    cm.show(outpath=os.path.join(output_directory, 'confusion_matrix.png'))  # Save the confusion matrix to a file
    
    # Create the classification report
    report=classification_report(y_test, y_pred)
    # Write the report to a file
    with open(os.path.join(output_directory,'classification_report.txt'), 'w') as file:
        file.write(report)
    
    # Create the final point cloud with a layer of predictions
    pcd_testing_subset = pcd_testing[['X', 'Y', 'Z']].copy()
    pcd_testing_subset['Predictions'] = y_pred
    # Saving the DataFrame to a CSV file
    pcd_testing_subset.to_csv(os.path.join(output_directory, 'predictions.txt'), index=False)
    
    
if __name__=='__main__':
	main()