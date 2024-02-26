# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 19:06:20 2023

@author: Luisja
"""

import cccorelib
import pycc
import os
import sys
import subprocess
import traceback

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import pandas as pd
import numpy as np
import open3d as o3d


#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS

script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'
sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance,get_point_clouds_name, check_input, write_yaml_file
from main_gui import show_features_window, definition_of_labels_type_1,definition_of_entries_type_1, definition_of_combobox_type_1,definition_ok_cancel_buttons_type_1,definition_run_cancel_buttons_type_1, definition_of_buttons_type_1

CC = pycc.GetInstance()
current_directory=os.path.dirname(os.path.abspath(__file__))
params = pycc.FileIOFilter.LoadParameters()
params.parentWidget = CC.getMainWindow()
processing_file=os.path.join(current_directory,'Anisotropic_denoising-1.0.4\\Anisotropic_denoising-1.0.4.exe')
output_file=os.path.join(os.path.dirname(current_directory),'temp\\','output.ply')

#%% INITIAL OPERATIONS
name_list=get_point_clouds_name()

#%% GUI
class GUI_ad(tk.Frame):
    def __init__(self, master=None, **kwargs): # Initial parameters. It is in self because we can update during the interaction with the user
        super().__init__(master, **kwargs)
    
    def main_frame (self, window):    # Main frame of the GUI  

        # Destroy the window
        def destroy (self): 
            window.destroy ()
        
        window.title("Anisotropic denoising")
        # Disable resizing the window
        window.resizable(False, False)
        # Remove minimize and maximize buttons (title bar only shows close button)
        window.attributes('-toolwindow', 1)
        
        # Create a frame for the form
        form_frame = tk.Frame(window, padx=10, pady=10)
        form_frame.pack()
        
        # Labels
        label_texts = [
            "Select a point cloud:"
        ]
        row_positions = [0]        
        definition_of_labels_type_1 ("window",label_texts,row_positions,form_frame,0)
        
        # Combobox
        combo_point_cloud=ttk.Combobox (form_frame,values=name_list)
        combo_point_cloud.grid(column=1, row=0, sticky="e", pady=2)
        combo_point_cloud.set("Not selected")
        
        # Buttons
        _=definition_run_cancel_buttons_type_1("window",
                                     [lambda:run_algorithm_1(self,name_list,combo_point_cloud.get()),lambda:destroy(self)],
                                     1,
                                     form_frame,
                                     1
                                     )
        
        def run_algorithm_1(self,name_list,input_file):
            
            #SELECTION CHECK
            if not CC.haveSelection():
                raise RuntimeError("No folder or entity selected")
            else:
                
                entities = CC.getSelectedEntities()[0]
            # RUN THE CMD FOR POINT CLOUD ONE TIME
                if hasattr(entities, 'points'):
            
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(entities.points())
                    
                    o3d.io.write_point_cloud(input_file,pcd, write_ascii=True)
                    #run the cmd of anisotropic filter 
                    command= processing_file + ' --i ' + input_file + ' --o ' + output_file
                    os.system(command)
                    pcd = o3d.io.read_point_cloud(output_file)
                    # Convert Open3D.o3d.geometry.PointCloud to numpy array
                    xyz_load = np.asarray(pcd.points)
                    point_cloud = pycc.ccPointCloud(xyz_load[:,0], xyz_load[:,1], xyz_load[:,2])
                    point_cloud.setName(entities.getName()+'_denoised')
                    CC.addToDB(point_cloud)
                    os.remove(input_file)
                    os.remove(output_file)        
            # RUN THE CMD THE SAME NUMBER THAN THE NUMBER OF POINT CLOUD WE HAVE
                else:
                    entities = CC.getSelectedEntities()[0]
                    number = entities.getChildrenNumber()  
                    for i in range (number):
                        if hasattr(entities.getChild(i), 'points'):
                            pc = entities.getChild(i)
                            pcd = o3d.geometry.PointCloud()            
                            pcd.points = o3d.utility.Vector3dVector(pc.points())
                            
                            o3d.io.write_point_cloud(input_file,pcd, write_ascii=True)
                            #run the cmd of anisotropic filter 
                            command= processing_file + ' --i ' + input_file + ' --o ' + output_file
                            os.system(command)
                            pcd = o3d.io.read_point_cloud(output_file)
                            # Convert Open3D.o3d.geometry.PointCloud to numpy array
                            xyz_load = np.asarray(pcd.points)
                            point_cloud = pycc.ccPointCloud(xyz_load[:,0], xyz_load[:,1], xyz_load[:,2])
                            point_cloud.setName(pc.getName()+'_denoised')
                            CC.addToDB(point_cloud)
                            os.remove(input_file)
                            os.remove(output_file)
            # UPDATE THE DB    
                CC.updateUI()
                print('The denosing stage has been finished')
                
#%% RUN THE GUI
if __name__ == "__main__":        
    try:
        # START THE MAIN WINDOW        
        window = tk.Tk()
        app = GUI_ad()
        app.main_frame(window)
        window.mainloop()    
    except Exception as e:
        print("An error occurred during the computation of the algorithm:", e)
        # Optionally, print detailed traceback
        traceback.print_exc()
        window.destroy()
        