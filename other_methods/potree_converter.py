# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:15:01 2023

@author: Pablo
"""

#%% LIBRARIES
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog


import cccorelib
import pycc
import os
import subprocess

import pandas as pd
import numpy as np
import open3d as o3d

import os
import sys


#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS

script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'
sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance

type_data, number = get_istance()

CC = pycc.GetInstance() 
current_directory=os.path.dirname(os.path.abspath(__file__))
params = pycc.FileIOFilter.LoadParameters()
processing_file=os.path.join(current_directory,'potree-2.1.1\\','PotreeConverter.exe')


#%% GUI

def destroy ():
    window.destroy ()

def save_file_dialog ():    
    file_path = filedialog.askdirectory()
    entry_out.delete(0, tk.END)  # Clear the current value in the entry
    entry_out.insert(0, file_path)  # Insert the new value into the entry
def run_algorithm ():
    
    
    #SELECTION CHECK
    if not CC.haveSelection():
        raise RuntimeError("No folder or entity selected")
    else:
        
        entities = CC.getSelectedEntities()[0]
    # RUN THE CMD FOR POINT CLOUD 
        if hasattr(entities, 'points'):
    
    
            pc_name = entities.getName()
            output_file_2 = os.path.join(entry_out.get(),pc_name)
            pc_name_full=pc_name+'.las'
            input_file=os.path.join(entry_out.get(), pc_name_full)  
            params = pycc.FileIOFilter.SaveParameters()
            result = pycc.FileIOFilter.SaveToFile(entities, input_file, params)
        
            command = processing_file + ' -i ' + input_file + ' -o ' + output_file_2 + ' --generate-page'
            os.system(command)
            current_name = os.path.join(output_file_2,'.html')
            new_name = os.path.join(output_file_2,'index.html')
            # Rename the file
            os.rename(current_name, new_name)
            os.remove (input_file)
    # RUN THE CMD FOLDER
        else:
            entities = CC.getSelectedEntities()[0]
            number = entities.getChildrenNumber()  
            for i in range (number):
                if hasattr(entities.getChild(i), 'points'):
                    pc = entities.getChild(i)
                    pc_name = pc.getName()
                    output_file_2 = os.path.join(entry_out.get(),pc_name)                
                    pc_name_full=pc_name+'.las'
                    input_file=os.path.join(entry_out.get(),pc_name_full)   
                    params = pycc.FileIOFilter.SaveParameters()
                    result = pycc.FileIOFilter.SaveToFile(entities.getChild(i), input_file, params)
                        
                    command = processing_file + ' -i ' + input_file + ' -o ' + output_file_2 + ' --generate-page'
                    os.system(command)
                    current_name = os.path.join(output_file_2,'.html')
                    new_name = os.path.join(output_file_2,'index.html')
                    # Rename the file
                    os.rename(current_name, new_name)
                    os.remove (input_file)
          
    print("Potree Converter has finished") 
    
# Create the main window
window = tk.Tk()

window.title("Arch analyzer")
# Disable resizing the window
window.resizable(False, False)
# Remove minimize and maximize buttons (title bar only shows close button)
window.attributes('-toolwindow', 1)

# Create a frame for the form
form_frame = tk.Frame(window, padx=10, pady=10)
form_frame.pack()

# Labels
label_out = tk.Label(form_frame, text="Choose output directory:")
label_out.grid(row=0, column=0, sticky="w",pady=2)

# Entry
entry_out = ttk.Entry(form_frame, width=30)
entry_out.grid(row=0, column=1, sticky="e", pady=2)
entry_out.insert(0, current_directory)

# Button
out_button= ttk.Button (form_frame, text="...", command=lambda: save_file_dialog(), width=10)
out_button.grid (row=0,column=2,sticky="e",padx=100)        
out_run_button= ttk.Button (form_frame, text="OK", command=lambda:run_algorithm (), width=10)
out_run_button.grid (row=1,column=1,sticky="e",padx=100)
out_cancel_button= ttk.Button (form_frame, text="Cancel", command=lambda:destroy(),width=10)
out_cancel_button.grid (row=1,column=1,sticky="e")
#%% START THE GUI
# Start the main event loop
window.mainloop()
