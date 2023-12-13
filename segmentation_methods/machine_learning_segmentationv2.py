# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 10:47:10 2023

@author: LuisJa

"""


import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
import pandas as pd
from fcmeans import FCM
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN, OPTICS
import pickle
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import InterclusterDistance
import matplotlib.pyplot as plt
from yellowbrick.cluster import SilhouetteVisualizer
import sys
import os

import cccorelib
import pycc


#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS
""" ESTA SECCIÓN ES COMUN, ES PARA CARGAR FUNCIONES DEL MODULO MAIN"""
script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'

sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance,get_point_clouds_name

#%% INPUTS AT THE BEGINING
""" ESTA SECCIÓN ES COMUN, ES PARA PONER ALGUNAS VARIABLES DE INICIO"""
name_list=get_point_clouds_name()
current_directory=os.path.dirname(os.path.abspath(__file__))
output_directory=os.path.join(current_directory,'..','temp_folder_for_results','Machine_Learning','OUTPUT')
#%% GUI
""" AQUI VIENE LA DIFERENCIA, LA GUI TIENE QUE ESTAR INTEGRADA EN UNA CLASE CON UNA ORDEN DE INIT QUE ALMACENA TODAS AQUELLAS VARIABLES QUE SE DECLARAN AL INICIO Y VAN A SERVIR PARA VARIAS VENTANAS
POR EJEMPLO: VARIABLE FEATURES2INCLUDE VA A SERVIR PARA LA VENTANA PRINCIPAL (FILTRARÁ LA NUBE DE PUNTOS DE ACUERDO A LOS FEATURES) Y PARA LA VENTANA DE FEATURES2 INCLUDE
PARA LA SCRIPT TUYE SE ME OCURRE FEATURES2INCLUDE Y VARIABLES TIPO SEP_UP_RANDOM_FOREST ETC. DONDE PONES VALORES DE INICIO POR DEFECTO, ESOS VALORES SON LOS INPUTS QUE YA METTES EN LOS ENTRIES 
COMO POR EJEMPLO NUMERO DE ARBOLES ETC. """
class GUI:
    def __init__(self):
        # Initial features2include
        self.features2include=[]
        # Initial paramertes for the different algorithms
        # K-means
        """ESTAS SON LAS VARIABLES DE INICIO DE LOS ALGORITMOS, AQUI TENEMOS KMEANS, TU TENDRAS OPTIMAL FLOW, RANDOM FOREST, AUTOML POR AHROA. PUEDES MEZCLAR INTEGERS, FLOATS
        Y STRING""" """ahora cada vez que lo quieras llamar vale con hacer self.set_up-parameters_km[0] aqui llamas al valor 5"""
        self.set_up_parameters_km=(5,200,'means')

        """ESTAS VARIABLES ALMACENAN LOS DIRECTORIOS DE GUARDADO, INTERESANTE CREARLAS POR QUE VERAS QUE EN LOS ENTRIES LAS LLAMAMOS Y CADA VEZ QUE CAMBIAMOS EL DIRECTORIO SE CAMBIA EN EL 
        ENTRY"""
        #Directoy to save the files (training)
        self.file_path=os.path.join(current_directory, output_directory)
        #Directory to load the features file (prediction)
        self.load_features=os.path.join(current_directory, output_directory)
        #Directory to load the configuration file (prediction)
        self.load_configuration=os.path.join(current_directory, output_directory)
        """ESTA VARIABLE TE SIRVE, ES PARA ALMACENAR LAS FEATURES TRAS LEER EL ARCHIVO EN EL MODULO DE PREDICTION"""
        #List with the features loaded (prediction)
        self.features_prediction=[]
    def main_frame (self, root):
        """ESTA SERÍA LA VENTANA PRINCIPAL, DONDE VAN LOS TABS, YO TENGO DOS PERO TU TRES"""
        root.title ("Machine Learning Segmentation")
        root.resizable (False, False)
        # Remove minimize and maximize button 
        root.attributes ('-toolwindow',-1)
        
        """CREAR AQUI LOS TRES TABS"""
        # Create two tabs
        tab_control = ttk.Notebook(root)
        tab_control.pack(expand=1, fill="both")
        
        tab1 = ttk.Frame(tab_control)
        tab1.pack()
        
        tab2 = ttk.Frame(tab_control)
        tab2.pack()
        
        tab_control.add(tab1, text='Training')
        tab_control.add(tab2, text='Prediction')
        
        tooltip = tk.Label(root, text="", relief="solid", borderwidth=1)
        tooltip.place_forget()
        
        
        """DEFINE EL CONTENIDO DE CADA TAB AQUI, DE FORMA NORMAL, TRATA DE QUE PONGAN AL INIO t1_label..... para tab 1 y t2_label.... así los tienes bien localizados"""
        # TAB1= TRAINING
        
        # Labels            
        label_pc_training = tk.Label(tab1, text="Choose point cloud for training:")
        label_pc_training.grid(row=0, column=0, sticky=tk.W)
        label_algo = tk.Label(tab1, text="Select a clustereing algorithm:")
        label_algo.grid(row=1, column=0, sticky=tk.W) 
        label_fea = tk.Label(tab1, text="Select the features to include:")
        label_fea.grid(row=2, column=0, sticky=tk.W) 
        label_out= ttk.Label(tab1, text="Choose output directory_")
        label_out.grid(row=3, column=0, sticky=tk.W)
        
        # Combobox
        combot1=ttk.Combobox (tab1,values=name_list)
        combot1.grid(row=0,column=1, sticky="e", pady=2)
        combot1.set("Not selected")
        
        algorithms = ["K-means", "Fuzzy-K-means","DBSCAN","OPTICS"]
        combot2=ttk.Combobox (tab1,values=algorithms, state="readonly")
        combot2.current(0)
        combot2.grid(column=1, row=1, sticky="e", pady=2)
        combot2.set("Not selected")
        
        # Entry
        self.entry_widget = ttk.Entry(tab1, width=30)
        self.entry_widget.grid(row=3, column=1, sticky="e", pady=2)
        self.entry_widget.insert(0, self.file_path)
        
        
        """AQUI YA LLAMAS FUNCIONES CON LOS BOTONES, AHORA TIENE EL ENCABEZADO DE SELF PORQUE SI TE FIJAS MÁS ABAJO SON FUNCTIONES QUE STAN AL MISMO NIVEL QUE LA FUNCION QUE CREA
        LA VENTANA PRINCIPAL"""
        # Button
        t1_setup_button= ttk.Button (tab1, text="Set-up", command=lambda: self.show_set_up_window(combot2.get()), width=10) """ESTA TE SIRVE, ES PARA ABRIR LA VENTANA DE CONFIGURACIÓN
        DE LOS ALGORITMOS, FIJATE QUE COJE COMO INPUT EL COMBOT2.GET() PARA SABER QUE ALGORITHMO ES. IR A FUNCION SHOW_SET_UP_WINDOW"""
        t1_setup_button.grid (row=1,column=2,sticky="e",padx=100)
        t1_features= ttk.Button (tab1, text="...", command=lambda: self.show_features_window(combot1.get()), width=10)"""ESTA TE SIRVE, ES PARA ABRIR LA VENTANA DE CONFIGURACIÓN
        DE LOS ALGORITMOS, FIJATE QUE COJE COMO INPUT EL COMBOT1.GET() PARA SABER QUE NUBE DE PUNTOS ES. IR A FUNCION SHOW_FEATURES_WINDOW"""
        t1_features.grid (row=2,column=2,sticky="e",padx=100)
        t1_button_widget = ttk.Button(tab1, text="...", command=self.save_file_dialog, width=10)"""ESTA TE SIRVE, ES PARA GUARDAR EL RESTULADO DEL ALGORITMO. IR A FUNCION
        SAVE_FILE_DIALOG"""
        t1_button_widget.grid(row=3, column=2, sticky="e", padx=100)
        t1_run_button= ttk.Button (tab1, text="OK", command=lambda:self.run_algorithm_1(combot2.get(),combot1.get()), width=10)"""ESTA TE SIRVE, ES PARA CORRER EL ALGORITMO
        YO USE, COMO TU RUN_ALGORITHM_1 PARA TAB 1 Y RUN_ALGORITHM_2 PARA TAB2"""
        t1_run_button.grid (row=4,column=1,sticky="e",padx=100)
        t1_cancel_button= ttk.Button (tab1, text="Cancel", command=self.destroy,width=10)
        t1_cancel_button.grid (row=4,column=1,sticky="e")
        
        """SE REPITE EL MISMO ESQUEMA QUE TAB 1"""
        # TAB2= PREDICTION
        # Labels            
        label_pc_prediction = tk.Label(tab2, text="Choose point cloud for prediction:")
        label_pc_prediction.grid(row=0, column=0, sticky=tk.W)
        label_features = tk.Label(tab2, text="Load feature file:")
        label_features.grid(row=1, column=0, sticky=tk.W) 
        label_configuration = tk.Label(tab2, text="Load configuration file:")
        label_configuration.grid(row=2, column=0, sticky=tk.W)
        label_out= ttk.Label(tab2, text="Choose output directory")
        label_out.grid(row=3, column=0, sticky=tk.W)
        # Combobox
        combot1_p=ttk.Combobox (tab2,values=name_list)
        combot1_p.grid(row=0,column=1, sticky="e", pady=2)
        combot1_p.set("Not selected")
        
        # Button
        t2_features= ttk.Button (tab2, text="...", command=lambda: self.load_features_dialog(), width=10)
        t2_features.grid (row=1,column=2,sticky="e",padx=100)
        t2_configuration= ttk.Button (tab2, text="...", command=lambda: self.load_configuration_dialog(), width=10)
        t2_configuration.grid(row=2,column=2,sticky="e",padx=100)
        t2_out = ttk.Button(tab2, text="...", command=self.save_file_dialog, width=10)
        t2_out.grid(row=3, column=2, sticky="e", padx=100)
        t2_run_button= ttk.Button (tab2, text="OK", command=lambda:self.run_algorithm_2(combot1_p.get()), width=10)
        t2_run_button.grid (row=4,column=1,sticky="e",padx=100)
        t2_cancel_button= ttk.Button (tab2, text="Cancel", command=self.destroy,width=10)
        t2_cancel_button.grid (row=4,column=1,sticky="e")   
        
    def save_setup_parameters (self,algo,*params): """TE INTERESA ES PARA GUARDAR EN LAS VARIABLES QUE IRAN EN VARIAS VENTANAS, DEFINIDAS EN EL INIT, LOS PARAMETROS ELEGIDOS POR EL USUARIO
    EN EL MOMENTO DE ABRIR ESTA VENTANA DE CONFIGURACIÓN. CAMBIA LOS VALORES DE ALGO Y EL SELF.SET_UP_PARAMETERS_KM POR LOS ALGORITMOS QUE CORRESPONDAN. LA VARIABLE *PARAMS ES MUY UTIL, AL 
    PONER EL ASTERISCO AL INICIO PYTHON INTERPRETA QUE ESA LISTA ES VARIABLEY POR TANTO UNAS VECES TENDRA 4 ELEMENTOS OTRAS 5 Y ASÍ DE FORMA QUE DEFINIENDO SOLO UNA ALMACENAS LAS DIFERENTES
    VARIABLES QUE TENGAS"""
        if algo=="K-means":
            self.set_up_parameters_km=params
        elif algo=="Fuzzy-K-means":
            self.set_up_parameters_fkm=params
        elif algo=="DBSCAN":
            self.set_up_parameters_dbscan=params  
        elif algo=="OPTICS":
            self.set_up_parameters_optics=params   
      
    
    def show_set_up_window (self,algo): """TE INITERESA. ESTA FUNCION DEFINE LA VENTANA DE SEP-UP DE LOS ALGORITMOS, SE LLAMA EN LINEA 129, LA VARIABLE ALGO SE METE GRACIAS AL COMBOT2"""
        def on_ok_button_click(algo):"""DEFINE LO QUE HACE EL BOTON OK,BASICAMENTE RECOJE LOS VALORES DE LOS ENTRIEES Y SE LOS PASA A LA FUNCIÓN DE LINEA 174 COMO *PARAMS, AL TENER ASTERISCO
        NO TIENES UN NUMERO PREDEFINIDO DE VARIABLES A PASAR SINO QUE CAMBIA SEGÚN EL ALGORITMO"""
            if algo=="K-means":
                self.save_setup_parameters(algo, int(entry_param1_km.get()), int(entry_param2_km.get()))  """ojo no olvides poner el tipo de variable que es int(),float(),str(), así la 
                almacenas como corresponde"""
            elif algo=="Fuzzy-K-means":
                self.save_setup_parameters(algo, int(entry_param1_fkm.get()), int(entry_param2_fkm.get()))
            elif algo=="DBSCAN":
                self.save_setup_parameters(algo, float(entry_param1_dbscan.get()), int(entry_param2_dbscan.get()))
            elif algo=="OPTICS":
                self.save_setup_parameters(algo, int(entry_param1_optics.get()), float(entry_param2_optics.get()), str(entry_param3_optics.get()), str(entry_param4_optics.get()), float(entry_param5_optics.get()),int(entry_param6_optics.get()))                
            
           
            set_up_window.destroy()  # Close the window after saving parameters
         
        """AQUI ESTAMOS FUERA DE LA FUNCIÓN, SE DEFINE LA GUI DE LA VENTANA SET-UP"""    
        set_up_window = tk.Toplevel(root)
        set_up_window.title("Set Up the algorithm")
        set_up_window.resizable (False, False)
        # Remove minimize and maximize button 
        set_up_window.attributes ('-toolwindow',-1)
           
        if algo=="K-means": """AL METER LA VARIABLE ALGO (VER LINE 186) YA PUEDES HACER CICLOS IF Y CUSTOMIZAR EL CONT4ENIDO DE CADA VENTANA SEGÚN EL ALGORITYMO ELEGIDO."""
             
            # Labels            
            label_param1_km = tk.Label(set_up_window, text="Number of clusters:")
            label_param1_km.grid(row=0, column=0, sticky=tk.W)
            label_param2_km = tk.Label(set_up_window, text="Number of iterations:")
            label_param2_km.grid(row=1, column=0, sticky=tk.W) 
            label_max_clusters = tk.Label(set_up_window, text="Maximum number of clusters:")
            label_max_clusters.grid(row=3, column=0, sticky=tk.W)
            label_max_clusters.config(state=tk.DISABLED)
            label_min_clusters = tk.Label(set_up_window, text="Minimum number of clusters:")
            label_min_clusters.grid(row=4, column=0, sticky=tk.W)   
            label_min_clusters.config(state=tk.DISABLED)
            # Entries
            entry_param1_km= tk.Entry(set_up_window)
            entry_param1_km.insert(0,self.set_up_parameters_km[0])
            entry_param1_km.grid(row=0, column=1, sticky=tk.W) 
            
            entry_param2_km= tk.Entry(set_up_window)
            entry_param2_km.insert(0,self.set_up_parameters_km[1])
            entry_param2_km.grid(row=1, column=1, sticky=tk.W)
            
            entry_max_clusters= tk.Entry(set_up_window)
            entry_max_clusters.insert(0,10)
            entry_max_clusters.grid(row=3, column=1, sticky=tk.W)
            entry_max_clusters.config(state=tk.DISABLED)
            entry_min_clusters= tk.Entry(set_up_window)
            entry_min_clusters.insert(0,1)
            entry_min_clusters.config(state=tk.DISABLED)
            entry_min_clusters.grid(row=4,column=1, sticky=tk.W)
            # Checkbox
            var1 = tk.IntVar()
            var2 = tk.IntVar()
            checkbox1 = tk.Checkbutton(set_up_window, text="Optimize the number of clusters by using the Elbow method", variable=var1, command=lambda: [check_uncheck_1(),toggle_row()])
            checkbox1.grid(row=2, column=0, sticky=tk.W)
            # Buttons
            button_ok = tk.Button(set_up_window, text="OK", command=lambda: on_ok_button_click(algo))
            button_ok.grid(row=5, column=1)
                   


    def select_all_checkbuttons(self,checkbuttons_vars): """ESTA MANTENLA TAL CUAL ES LA DE SELECCIÓN DE FEATURES""" 
        for var in checkbuttons_vars:
            var.set(True)   
            
    def show_features_window(self,training_pc_name): """ESTA MANTENLA TAL CUAL ES LA DE SELECCIÓN DE FEATURES""" """SE LLAMA EN LA LINEA 132 Y TIENE DE INPUT EL NOMBRE 
    DE LA NUBE QUE LO COJE DE COMBOT1"""
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
        
        pcd_training = P2p_getdata(pc_training, False, True, True)
    
        feature_window = tk.Toplevel()
        feature_window.title("Features of the point cloud")
    
        checkbutton_frame = tk.Frame(feature_window)
        checkbutton_frame.pack(side="left", fill="y")
    
        canvas = tk.Canvas(checkbutton_frame)
        features_frame = tk.Frame(canvas)
    
        scrollbar = tk.Scrollbar(checkbutton_frame, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
    
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
    
        canvas.create_window((0, 0), window=features_frame, anchor="nw")
    
        # Buttons frame (static)
        button_frame = tk.Frame(feature_window)
        button_frame.pack(side="right", fill="y")
    
        # Your checkbuttons and variables
        values_list = [col for col in pcd_training.columns if col != 'Class']
        checkbuttons_vars = [tk.BooleanVar() for _ in values_list]
    
        for value, var in zip(values_list, checkbuttons_vars):
            ttk.Checkbutton(features_frame, text=value, variable=var, onvalue=True, offvalue=False).pack(anchor="w")
    
        select_all_button = ttk.Button(button_frame, text="Select All", command=lambda: self.select_all_checkbuttons(checkbuttons_vars))
        select_all_button.pack(side="top", pady=5)
    
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
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
    
        ok_button_features = ttk.Button(button_frame, text="OK", command=ok_features_window, width=10)
        ok_button_features.pack(side="left")
        cancel_button_features = ttk.Button(button_frame, text="Cancel", command=cancel_features_window, width=10)
        cancel_button_features.pack(side="right")
   
    def destroy (self): """MANTENLA IGUAL"""
        root.destroy ()


    def run_algorithm_1 (self,algo,training_pc_name): 
        

        print("The process has been finished")
# START THE MAIN WINDOW        
root = tk.Tk()
app = GUI()
app.main_frame(root)
root.mainloop()
