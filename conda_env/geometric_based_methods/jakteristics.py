# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 13:12:14 2024

@author: Digi_2
"""

import argparse
import os
import yaml

import pandas as pd
from jakteristics import compute_features, las_utils

def main():
    # Import all the parameters form the CMD
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--i',type=str,help='Yaml configuration file')
    # parser.add_argument('--o',type=str,help='Output_directory')    
    # args=parser.parse_args()    
            
    # # Read the configuration from the YAML file for the set-up
    # with open(args.i, 'r') as yaml_file:
    #     config_data = yaml.safe_load(yaml_file)
    # input_file= config_data.get('INPUT_POINT_CLOUD')
    # output_directory= config_data.get('OUTPUT_DIRECTORY')
    # radius=config_data['CONFIGURATION']['radius']
    # features=config_data['CONFIGURATION']['features']
    
    input_file= r"C:\Users\Digi_2\Documents\gf\input_point_cloud.txt"
    output_directory= r"C:\Users\Digi_2\Documents\gf"
    radius= [0.1,0.2,0.4,0.8]
    features_selected= ['eigenentropy', 'anisotropy']
    
    
    # input_file=args.i
    print("Input file located in " + str(input_file))
    # output_directory=args.o
    print("Output file located in " + str(output_directory))
    # cross_val=args.cv
    print("Radius = " + str(radius))
    # features=args.f
    print("Features chosen = " + str(features_selected))
    
        
    xyz=pd.read_csv(input_file,delimiter=' ')
        
    
    for r in radius:
        features = compute_features(xyz, search_radius=r, feature_names= features_selected)
    
    las_utils.write_with_extra_dims(input_file, output_directory, features, features_selected)



if __name__=='__main__':
	main()
    