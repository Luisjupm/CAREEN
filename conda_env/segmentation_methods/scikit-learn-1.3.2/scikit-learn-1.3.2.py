# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 19:42:08 2023

@author: LuisJa
"""
import pandas as pd

import sklearn

from sklearn.pipeline import *
from sklearn.ensemble import *
from sklearn.ensemble import *
from sklearn.kernel_approximation import *
from sklearn.naive_bayes import *
from sklearn.neural_network import *
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
from tpot import TPOTClassifier
from tpot import *
import tpot

import matplotlib.pyplot as plt
from yellowbrick.classifier import ConfusionMatrix

import joblib

import argparse

import os
import yaml

def main():
    
    # Import all the parameters form the CMD
    parser = argparse.ArgumentParser()
    parser.add_argument('--i',type=str,help='Yaml configuration file')
    parser.add_argument('--o',type=str,help='Output_directory')    
    args=parser.parse_args() 
    
    #Read the configuration from the YAML file for the set-up
    with open(args.i, 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    algo =  config_data.get('ALGORITHM')
    if algo=='Prediction':
        
        input_file= config_data.get('INPUT_POINT_CLOUD')
        output_directory= config_data.get('OUTPUT_DIRECTORY')
        features2include= config_data['CONFIGURATION']['f']   
        pkl_file=config_data['CONFIGURATION']['p']   
       
        
        print("The input file is taken from " + str(input_file))  
        print("The features are taken from " + str(features2include)) 
        print("The pkl file is taken from " + str(pkl_file)) 
        print("Output directory is " + str(output_directory))
        
        
        # Store in a Pandas dataframe the content of the file
        pcd_testing=pd.read_csv(input_file,delimiter=' ')
        with open(features2include, "r") as file:
            f2i = [line.strip().split(',') for line in file]    
        X_test= pcd_testing[f2i[0]].ffill()
        
      
        # Load the model from the file
        loaded_model = joblib.load(pkl_file)
      
        # Prediction
        y_pred = loaded_model.predict(X_test)     
        
        # Create the final point cloud with a layer of predictions
        pcd_testing_subset = pcd_testing[['X', 'Y', 'Z']].copy()
        pcd_testing_subset['Predictions'] = y_pred
        # Saving the DataFrame to a CSV file
        pcd_testing_subset.to_csv(os.path.join(output_directory, 'predictions.txt'), index=False)
    
if __name__=='__main__':
	main()