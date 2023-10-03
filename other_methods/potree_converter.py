# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:15:01 2023

@author: Pablo
"""
import cccorelib
import pycc
import os
import subprocess

import pandas as pd
import numpy as np
import open3d as o3d

import os
import sys
# ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS

script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'
print (additional_modules_directory)
sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance

type_data, number = get_istance()

CC = pycc.GetInstance() 
current_directory=os.path.dirname(os.path.abspath(__file__))
params = pycc.FileIOFilter.LoadParameters()
params.parentWidget = CC.getMainWindow()
input_file=os.path.join(os.path.dirname(current_directory),'temp_folder_for_results\\','INPUT\\','input.laz')
output_file=os.path.join(os.path.dirname(current_directory),'temp_folder_for_results\\','potree')
processing_file=os.path.join(current_directory,'potree-2.1.1\\','PotreeConverter.exe')

#SELECTION CHECK
if not CC.haveSelection():
    raise RuntimeError("No folder or entity selected")
else:
    
    entities = CC.getSelectedEntities()[0]
# RUN THE CMD FOR POINT CLOUD 
    if hasattr(entities, 'points'):


        pc_name = entities.getName()
        output_file_2 = output_file + '\\' + pc_name
      
        if not os.path.exists(output_file_2):
            os.makedirs(output_file_2)
            
        params = pycc.FileIOFilter.SaveParameters()
        result = pycc.FileIOFilter.SaveToFile(entities, input_file, params)

        command = processing_file + ' -i ' + input_file + ' -o ' + output_file_2 + ' --generate-page index '
        os.system(command)
        os.remove(input_file)
        print(command)
        
# RUN THE CMD FOLDER
    else:
        entities = CC.getSelectedEntities()[0]
        number = entities.getChildrenNumber()  
        for i in range (number):
            if hasattr(entities.getChild(i), 'points'):
                pc = entities.getChild(i)
                pc_name = pc.getName()
                output_file_2 = output_file + '\\' + pc_name
               
                if not os.path.exists(output_file_2):
                    os.makedirs(output_file_2)
                    
                params = pycc.FileIOFilter.SaveParameters()
                result = pycc.FileIOFilter.SaveToFile(entities.getChild(i), input_file, params)
                    
            
                pcd = o3d.geometry.PointCloud()            
                pcd.points = o3d.utility.Vector3dVector(pc.points())

                command = processing_file + ' -i ' + input_file + ' -o ' + output_file_2 + ' --generate-page index '
                os.system(command)
                os.remove(input_file)
                print(command)
                
# UPDATE THE DB    
    CC.updateUI()        
print("Potree Converter has finished") 