# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 19:42:08 2023

@author: LuisJa
"""
import pandas as pd

import sklearn

from sklearn.pipeline import *
from sklearn.ensemble import *

import joblib

import argparse

import os

#%% DEFINING INPUTS OF CMD
current_directory=os.path.dirname(os.path.abspath(__file__))
temp_folder=os.path.join(current_directory,'..','temp')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--i',type=str,help='Path for the input file')
    parser.add_argument('--o',type=str,help='Path for the output file')
    parser.add_argument('--f',type=str,help='Path for the features')
    parser.add_argument('--p',type=str,help='Path for the pkl file')
    
    args=parser.parse_args()  
    
    input_file=args.i
    print("The input file is taken from " + input_file)  
    features2include=args.f
    print("The features are taken from " + features2include)
    pkl_file=args.p
    print("The pkl file is taken from " + pkl_file)
    output_directory=args.o
    print("Output directory is " + output_directory)
    
    
    # Store in a Pandas dataframe the content of the file
    pcd=pd.read_csv(input_file,delimiter=' ')
    with open(features2include, "r") as file:
        f2i = [line.strip().split(',') for line in file]    
    X_test=pcd[f2i[0]].ffill()
    # Load the pickled model
    loaded_model = joblib.load(pkl_file)
    # # Assuming 'new_data' contains the data you want to predict on
    y_pred = loaded_model.predict(X_test)
    
    pcd_testing_subset = pcd[['X', 'Y', 'Z']].copy()
    pcd_testing_subset['Predictions'] = y_pred
    # Saving the DataFrame to a CSV file
    pcd_testing_subset.to_csv(os.path.join(output_directory, 'predictions.txt'), index=False)
    
    
if __name__=='__main__':
	main()