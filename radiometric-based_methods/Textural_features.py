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

class App(tk.Tk):
    def P2p_getdata (pc,nan_value=False,sc=True):
        ## CREATE A DATAFRAME WITH THE POINTS OF THE PC
       pcd = pd.DataFrame(pc.points(), columns=['X', 'Y', 'Z'])
       if (sc==True):       
       ## ADD SCALAR FIELD TO THE DATAFRAME
           for i in range(pc.getNumberOfScalarFields()):
               scalarFieldName = pc.getScalarFieldName(i)  
               scalarField = pc.getScalarField(i).asArray()[:]              
               pcd.insert(len(pcd.columns), scalarFieldName, scalarField) 
       ## DELETE NAN VALUES
       if (nan_value==True):
           pcd.dropna(inplace=True)
       return pcd  
    
    ## LOAD THE SELECTED POINT CLOUD
    CC = pycc.GetInstance() 
    entities = CC.getSelectedEntities()
    print(f"Selected entities: {entities}")
    if not entities:
        raise RuntimeError("No entities selected")
    else:
        pc = entities[0]
        pcd=P2p_getdata(pc,False,True)
    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize

 
    def __init__(self):
        CC = pycc.GetInstance()  
        super().__init__()
        self.title("Computation of texture-related features")
        
        # Variables de control para las opciones
        self.algorithm1_var = tk.BooleanVar()
        self.algorithm2_var = tk.BooleanVar()
        self.algorithm3_var = tk.BooleanVar()
        self.algorithm4_var = tk.BooleanVar()

        # Etiquetas de los algoritmos
        self.radius_label=tk.Label(self,text="Radius")
        self.combobox_label=tk.Label(self,text="Feature to be used")
        self.algorithm1_label = tk.Label(self, text="Mean value")
        self.algorithm2_label = tk.Label(self, text="Variance")
        self.algorithm3_label = tk.Label(self, text="Kurtosis")
        self.algorithm4_label = tk.Label(self, text="Skewness")
            
        self.entry_param1 = tk.Entry(self)
        self.entry_param1.insert(0,'0.1')
        features=list(self.pcd.columns.values)
        self.combobox = ttk.Combobox(self, values=features, width=40)
        
        self.algorithm1_checkbox = tk.Checkbutton(self, variable=self.algorithm1_var)
        self.algorithm2_checkbox = tk.Checkbutton(self, variable=self.algorithm2_var)
        self.algorithm3_checkbox = tk.Checkbutton(self, variable=self.algorithm3_var)
        self.algorithm4_checkbox = tk.Checkbutton(self, variable=self.algorithm4_var)
        # Botón de ejecución
        self.run_button = tk.Button(self, text="OK", command=self.run_algorithms)
        self.cancel_button = tk.Button(self, text="Cancel", command=self.destroy)

        # Posicionamiento de los elementos en la ventana
        self.radius_label.grid(row=0, column=0, sticky=tk.W, padx=10)  
        self.entry_param1.grid(row=0, column=1, sticky=tk.W, padx=10)
        self.combobox_label.grid(row=1, column=0, sticky=tk.W, padx=10)  
        self.combobox.grid(row=1, column=1, sticky=tk.W, padx=10) 
        self.combobox.current (0)       
        self.entry_param1.grid(row=0, column=1, sticky=tk.W, padx=10)        
        self.algorithm1_label.grid(row=2, column=0, sticky=tk.W, padx=10)
        self.algorithm2_label.grid(row=3, column=0, sticky=tk.W, padx=10)
        self.algorithm3_label.grid(row=4, column=0, sticky=tk.W, padx=10)
        self.algorithm4_label.grid(row=5, column=0, sticky=tk.W, padx=10)
        
        self.algorithm1_checkbox.grid(row=2, column=1)
        self.algorithm2_checkbox.grid(row=3, column=1)
        self.algorithm3_checkbox.grid(row=4, column=1)
        self.algorithm4_checkbox.grid(row=5, column=1)
        
        self.run_button.grid(row=6, column=0, pady=10)
        self.cancel_button.grid(row=6, column=1, pady=10)
        
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
            point_coordinates = np.asarray(pcdxyz.points)    
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


