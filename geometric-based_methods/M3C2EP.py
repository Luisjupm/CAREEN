# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:07:07 2023

@author: Luisja
"""

import tkinter as tk

import cccorelib
import pycc

import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import distance
from scipy.optimize import fsolve
from itertools import combinations
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tkinter import ttk
from tkinter import filedialog

import os 
import sys
# ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS

script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'
print (additional_modules_directory)
sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance, get_point_clouds_name

name_list =get_point_clouds_name ()

def on_select(event):
    selected_name = combo_e1.get()
    selected_name = combo_e2.get()
def toggle_entry_state():
    if checkbox_1_var.get():
        entry_normal.config(state=tk.NORMAL)
    else:
        entry_normal.config(state=tk.DISABLED)
    if checkbox_2_var.get():
        combo_core.config(state=tk.NORMAL)
        label_core.config(state=tk.NORMAL)
    else:
        combo_core.config(state=tk.DISABLED)
        label_core.config(state=tk.DISABLED)
def destroy():
    window.destroy()  # Close the window    
# Create the main window
window = tk.Tk()

window.title("M3C2EP-Change detection for 3D point clouds")
# Disable resizing the window
window.resizable(False, False)
# Remove minimize and maximize buttons (title bar only shows close button)
window.attributes('-toolwindow', 1)

# Create a frame for the form
form_frame = tk.Frame(window, padx=10, pady=10)
form_frame.pack()



# Labels for the algorithms
label_tolerance = tk.Label(form_frame, text="Epoch #1:")
label_tolerance.grid(row=0, column=0, sticky="w",pady=2)

label_degree=tk.Label(form_frame, text="Epoch #2:")
label_degree.grid(row=1, column=0, sticky="w",pady=2)

#Labels for different parts
label_scales = tk.Label(form_frame, text="Scales")
label_scales.grid(row=2, column=0, sticky="w",pady=2)

checkbox1_label = tk.Label(form_frame, text="Calculate normals")
checkbox1_label.grid (row=3, column=0, sticky="w",pady=2,padx=25)

label_normal = tk.Label(form_frame, text="Normals:")
label_normal.grid(row=4, column=0, sticky="w",pady=2,padx=25)

label_diameter_of_cylinder = tk.Label(form_frame, text="Diameter of cylinder:")
label_diameter_of_cylinder.grid(row=5, column=0, sticky="w",pady=2,padx=25)

label_diameter_of_cylinder = tk.Label(form_frame, text="Lenght of cylinder:")
label_diameter_of_cylinder.grid(row=6, column=0, sticky="w",pady=2,padx=25)

label_use_core = tk.Label(form_frame, text="Use core points:")
label_use_core.grid(row=7, column=0, sticky="w",pady=2)

label_core = tk.Label(form_frame, text="Core points:")
label_core.grid(row=8, column=0, sticky="w",pady=2,padx=25)
label_core.config(state=tk.DISABLED)
# Combobox
combo_e1 = ttk.Combobox(form_frame, values=name_list)
combo_e1.set("Select the point cloud of epoch #1")
combo_e1.grid(row=0, column=1, sticky="w",pady=2)

combo_e2 = ttk.Combobox(form_frame, values=name_list)
combo_e2.set("Select the point cloud of epoch #2")
combo_e2.grid(row=1, column=1, sticky="w",pady=2)

combo_core = ttk.Combobox(form_frame, values=name_list)
combo_core.set("Select the point cloud of core points")
combo_core.grid(row=8, column=1, sticky="w",pady=2)
combo_core.config(state=tk.DISABLED)

label_use_core = tk.Label(form_frame, text="Uncertainities:")
label_use_core.grid(row=7, column=0, sticky="w",pady=2)

label_use_core = tk.Label(form_frame, text="Laser scanner uncertainity")
label_use_core.grid(row=7, column=0, sticky="w",pady=25)



# Checkbox
# Variables de control para las opciones
checkbox_1_var = tk.BooleanVar()
checkbox_1 = tk.Checkbutton(form_frame, variable=checkbox_1_var, command=toggle_entry_state)
checkbox_1.grid (row=3, column=1, sticky="e",pady=2)

checkbox_2_var = tk.BooleanVar()
checkbox_2 = tk.Checkbutton(form_frame, variable=checkbox_2_var, command=toggle_entry_state)
checkbox_2.grid (row=7, column=1, sticky="e",pady=2)

# Entries
entry_normal = tk.Entry(form_frame,width=5)
entry_normal.insert(0,0.20)
entry_normal.grid(row=4, column=1, sticky="e",pady=2)


# Entries
entry_normal = tk.Entry(form_frame,width=5)
entry_normal.insert(0,0.20)
entry_normal.grid(row=4, column=1, sticky="e",pady=2)
entry_normal.config(state=tk.DISABLED)

entry_diameter_of_cylinder = tk.Entry(form_frame,width=5)
entry_diameter_of_cylinder.insert(0,1)
entry_diameter_of_cylinder.grid(row=5, column=1, sticky="e",pady=2)

entry_lenght_of_cylinder = tk.Entry(form_frame,width=5)
entry_lenght_of_cylinder.insert(0,3)
entry_lenght_of_cylinder.grid(row=6, column=1, sticky="e",pady=2)










# Start the main event loop
window.mainloop()