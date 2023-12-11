# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 17:35:39 2023

@author: Luisja
"""

import cccorelib
import pycc
import numpy as np
import pandas as pd
import os

import tkinter as tk
from tkinter import filedialog
from fcmeans import FCM

import joblib


CC = pycc.GetInstance()

class LoadModelGUI:
    
    def __init__(self):
        entities = CC.getSelectedEntities()
        print(f"Selected entities: {entities}")
    
        if not entities:
            raise RuntimeError("No entities selected")
            
        self.pc = entities[0]
        print (self.pc)
        ## CREATE A DATAFRAME WITH THE POINTS OF THE PC
        self.pcd = pd.DataFrame(self.pc.points(), columns=['X', 'Y', 'Z'])
  
        # ATTACH THE SCALAR FIELDS
        count = 0
        for i in range(self.pc.getNumberOfScalarFields()):
            scalarFieldName = self.pc.getScalarFieldName(i)
            scalarField = self.pc.getScalarField(i).asArray()[:]        
            self.pcd.insert(len(self.pcd.columns), scalarFieldName, scalarField)      
            # FOR DELETING NAN VALUES
            self.pcd.dropna(inplace=True)
          
        self.window = tk.Tk()
        self.window.title("Load Model GUI")
        
        # Crear los widgets
        self.load_path_label = tk.Label(self.window, text="Ruta del archivo:")
        self.load_path_textbox = tk.Entry(self.window)
        self.load_path_button = tk.Button(self.window, text="Seleccionar ruta", command=self.select_load_path)
        self.run_button = tk.Button(self.window, text="Ejecutar", command=self.run_model)
        
        # Colocar los widgets en la ventana
        self.load_path_label.grid(row=0, column=0)
        self.load_path_textbox.grid(row=0, column=1)
        self.load_path_button.grid(row=0, column=2)
        self.run_button.grid(row=1, column=1)
        
    def select_load_path(self):
        # Abrir el diálogo para seleccionar la ruta del archivo
        path = filedialog.askopenfilename(filetypes=[("Archivos pkl", "*.pkl")])
        
        # Mostrar la ruta seleccionada en el textbox correspondiente
        self.load_path_textbox.delete(0, tk.END)
        self.load_path_textbox.insert(0, path)
        
    def run_model(self):
        # Cargar el modelo desde el archivo especificado
        load_path = self.load_path_textbox.get()
        model = joblib.load(load_path)
        # Cargar el archivo con las características a incluir
        feature_path = os.path.join(os.path.dirname(load_path), 'features2include.txt')
        with open(feature_path, 'r') as f:
            lines = f.readlines()
            features_line = lines[0].strip() # Lee la primera línea del archivo
            features_list = eval(features_line.split(': ')[-1]) # Obtiene la lista de características
        pcd_reduced=self.pcd[features_list]
        
        if "Kmeans" in load_path:            
            kmeans = joblib.load(load_path)
            result = kmeans.predict(pcd_reduced)
        elif 'Fuzzy-K-means' in load_path:
            pass
#***FATAL ESTA IMPLEMENTACION
        idx=self.pc.addScalarField("Clusters", result)
        print (result)
        CC.updateUI() 
        print('The predicted point cloud has been updated, adding the Classification layer')        
app = LoadModelGUI()
app.window.mainloop()