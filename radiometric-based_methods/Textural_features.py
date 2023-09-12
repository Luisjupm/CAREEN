# -*- coding: utf-8 -*-
"""
Created on Mon May 15 23:00:29 2023

@author: Luisja
"""
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import numpy as np
import open3d as o3d
from scipy.stats import kurtosis, skew
import cccorelib
import pycc
import pandas as pd
import os
import sys

# ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS

script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'
print (additional_modules_directory)
sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance

class App(tk.Tk):

    
    ## LOAD THE SELECTED POINT CLOUD
    type_data, number = get_istance()
    if type_data=="point_cloud":
        CC = pycc.GetInstance() 
        entities = CC.getSelectedEntities()
        pc = entities[0]
        pcd=P2p_getdata(pc,False,True,True)      

    else:
        raise RuntimeError("You need to select a point cloud")
 
    def __init__(self):
        super().__init__()
        self.title("Texture-related features")
        # Disable resizing the window
        self.resizable(False, False)
        # Remove minimize and maximize buttons (title bar only shows close button)
        self.attributes('-toolwindow', 1)
        features=list(self.pcd.columns.values)


        # Create a frame for the form
        form_frame = tk.Frame(self, padx=10, pady=10)
        form_frame.pack()
        # Variables for controlling the options
        self.algorithm1_var = tk.BooleanVar()
        self.algorithm2_var = tk.BooleanVar()
        self.algorithm3_var = tk.BooleanVar()
        self.algorithm4_var = tk.BooleanVar()

        # Labels for the algorithm
        self.radius_label=tk.Label(form_frame,text="Radius")
        self.radius_label.grid(row=0, column=0, sticky=tk.W, pady=2) 
        
        self.combobox_label=tk.Label(form_frame,text="Feature to be used")
        self.combobox_label.grid(row=1, column=0, sticky=tk.W, pady=4)  
        
        
        self.algorithm1_label = tk.Label(form_frame, text="Mean value")
        self.algorithm2_label = tk.Label(form_frame, text="Variance")
        self.algorithm3_label = tk.Label(form_frame, text="Kurtosis")
        self.algorithm4_label = tk.Label(form_frame, text="Skewness")
        self.algorithm1_label.grid(row=2, column=0, sticky=tk.W, pady=2)
        self.algorithm2_label.grid(row=3, column=0, sticky=tk.W, pady=2)
        self.algorithm3_label.grid(row=4, column=0, sticky=tk.W, pady=2)
        self.algorithm4_label.grid(row=5, column=0, sticky=tk.W, pady=2)
        
        # Entries
        self.entry_param1 = tk.Entry(form_frame,width=5)
        self.entry_param1.insert(0,'0.1')
        self.entry_param1.grid(row=0, column=1, sticky=tk.E, pady=2)  
        
        # Combobox
        self.combobox = ttk.Combobox(form_frame, values=features,width=15)
        self.combobox.grid(row=1, column=1, sticky=tk.E, pady=2) 
        self.combobox.current (0) 
        
        self.algorithm1_checkbox = tk.Checkbutton(form_frame, variable=self.algorithm1_var)
        self.algorithm1_checkbox.grid(row=2, column=1, sticky="e")
        
        self.algorithm2_checkbox = tk.Checkbutton(form_frame, variable=self.algorithm2_var)
        self.algorithm2_checkbox.grid(row=3, column=1, sticky="e")
        
        self.algorithm3_checkbox = tk.Checkbutton(form_frame, variable=self.algorithm3_var)
        self.algorithm3_checkbox.grid(row=4, column=1, sticky="e")
        
        self.algorithm4_checkbox = tk.Checkbutton(form_frame, variable=self.algorithm4_var)
        self.algorithm4_checkbox.grid(row=5, column=1, sticky="e")
        
        # Buttons      
        self.run_button = tk.Button(form_frame, text="OK", command=self.run_algorithms,width=10)
        self.cancel_button = tk.Button(form_frame, text="Cancel", command=self.destroy,width=10)
        self.run_button.grid(row=6, column=1, sticky="e",padx=100)
        self.cancel_button.grid(row=6, column=1, sticky="e")
            

    def run_algorithms(self):
        features2include=self.combobox.get()
        radii=list(map(float, self.entry_param1.get().split(",")))     
        # Initialize empty lists for results
        array = self.pcd.values
        array_f=self.pcd[features2include].values

        # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
        pcdxyz = o3d.geometry.PointCloud()
        pcdxyz.points = o3d.utility.Vector3dVector(array[:,0:3])
        pcdxyz_tree = o3d.geometry.KDTreeFlann(pcdxyz)


        for radius in radii:
            mean_values = []
            var_values = []
            kurtosis_values = []
            skewness_values = []  
            for i in range(len(pcdxyz.points)):
                [k, idx, _] = pcdxyz_tree.search_radius_vector_3d(pcdxyz.points[i], radius)
                filtered_array = array_f[idx[1:]]+array_f[i]
                mean_values.append(np.mean(filtered_array))
                var_values.append(np.var(filtered_array))
                kurtosis_values.append(kurtosis(filtered_array))
                skewness_values.append(skew(filtered_array))
            if self.algorithm1_var.get():            
                idx = self.pc.addScalarField("Mean(Texture)("+str(radius)+")", mean_values)
            if self.algorithm2_var.get():                       
                idx = self.pc.addScalarField("Variance(Texture)("+str(radius)+")", var_values)
            if self.algorithm3_var.get():
                idx = self.pc.addScalarField("Kurtosis(Texture)("+str(radius)+")", kurtosis_values)
            if self.algorithm4_var.get():
                idx = self.pc.addScalarField("Skewness(Texture)("+str(radius)+")", skewness_values)
        ## STORE IN THE DATABASE OF CLOUDCOMPARE
        self.CC.addToDB(self.pc)
        self.CC.updateUI() 
        print('The features has been computed')
        
        self.destroy()  # Close the window
        
        
app = App()
app.mainloop()


