# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 15:19:01 2024

@author: LuisJa
"""
#%% LIBRARIES
import os
import sys
#CloudCompare Python Plugin
import cccorelib
import pycc

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog

#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS
script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'

sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance,get_point_clouds_name
#%% FUNCTIONS

def show_features_window(self,name_list,training_pc_name="Not selected",excluded_feature="Classification"):
    """
    This fuction allows to render a form for selecting the features of the point cloud
        
    Parameters
    ----------
    self (self): allow to store the data outside this window. Examples: self.features2include
    
    name_list (list) : list of available point clouds
    
    training_pc_name (str): target point cloud. Default: "Not selected"    
    
    excluded_feature (str): name of the feeature that need to be excluded from the selection. Default: "Classification"
   
    Returns
    -------

    """
    # Functions
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
    def select_all_checkbuttons(checkbuttons_vars): 
        for var in checkbuttons_vars:
            var.set(True)
            
    def ok_features_window():
        features2include = [value for value, var in zip(values_list, checkbuttons_vars) if var.get()]
        if not features2include:
            print("Please, check at least one feature")
        else:
            if len(features2include) == 1:
                print("The feature " + str(features2include) + " has been included for the training")
            else:
                print("The features " + str(features2include) + " have been included for the training")
        self.features2include=features2include
        feature_window.destroy()

    def cancel_features_window():
        feature_window.destroy()
        
    # Some lines of code for ensuring thee proper selection of the point cloud

    if training_pc_name=="Not selected":
        raise RuntimeError("Please select a point cloud to evaluate the features")
    CC = pycc.GetInstance()
    entities = CC.getSelectedEntities()[0]    
    type_data, number = get_istance()
    if type_data=='point_cloud':
        pc_training=entities
    else:
        for ii, item in enumerate(name_list):
            if item == training_pc_name:
                pc_training = entities.getChild(ii)
                break
    # Transform the point cloud to a pandas dataframe
    pcd_training = P2p_getdata(pc_training, False, True, True)
    
    #GUI
    feature_window = tk.Toplevel()
    feature_window.title("Features of the point cloud")

    # Checkbutton
    checkbutton_frame = tk.Frame(feature_window)
    checkbutton_frame.pack(side="left", fill="y")

    # Canvas
    canvas = tk.Canvas(checkbutton_frame)
    features_frame = tk.Frame(canvas)
    
    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((0, 0), window=features_frame, anchor="nw")
    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    # Scrollbar
    scrollbar = tk.Scrollbar(checkbutton_frame, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Buttons
    button_frame = tk.Frame(feature_window)
    button_frame.pack(side="right", fill="y")
    select_all_button = ttk.Button(button_frame, text="Select All", command=lambda: select_all_checkbuttons(checkbuttons_vars))
    select_all_button.pack(side="top", pady=5)
    ok_button_features = ttk.Button(button_frame, text="OK", command=ok_features_window, width=10)
    ok_button_features.pack(side="left")
    cancel_button_features = ttk.Button(button_frame, text="Cancel", command=cancel_features_window, width=10)
    cancel_button_features.pack(side="right")
    
    # Render the features within the check button
    values_list = [col for col in pcd_training.columns if col != excluded_feature]
    checkbuttons_vars = [tk.BooleanVar() for _ in values_list]
    for value, var in zip(values_list, checkbuttons_vars):
        ttk.Checkbutton(features_frame, text=value, variable=var, onvalue=True, offvalue=False).pack(anchor="w")
        
        
def definition_of_labels (header,label_texts,row_positions,window,column=0,pady=2,sticky="w"):
    
    """
    This fuction allows to create the labels of a tab
        
    Parameters
    ----------
    header (str): name of the label. It will be as: header_idx. Where idx is the row on which the label appears. I.e t1_1 the header is t1 and the row is 1 for this element
    
    row_positions (list): a list with the position (rows) of each label text
    
    label_text (list) a list with the name of each label
    
    window (tk window): the window on which the information will be rendered
    
    column (int): the column to place the labels. Default: 0
    
    pady (int): the pad along the y-axis. Default: 2
    
    sticky (str): the position of the text.  Default: "w"
   
    Returns
    -------

    """      
   
    labels = {}  # Dictionary to store labels    
    for idx, (text, row) in enumerate(zip(label_texts, row_positions)):
        label_name = f"t1_{idx}"
        labels[label_name] = ttk.Label(window, text=text)
        labels[label_name].grid(column=0, row=row, pady=2, sticky="w")     

              
    