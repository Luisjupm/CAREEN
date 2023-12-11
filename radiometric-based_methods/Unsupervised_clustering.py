# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 11:43:22 2023

@author: Luisja
"""

import cccorelib
import pycc
import numpy as np
import pandas as pd
import time
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN, OPTICS
from fcmeans import FCM

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
from yellowbrick.cluster import KElbowVisualizer
import joblib
import sys
import pickle

#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS
script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'
sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance,get_point_clouds_name

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\segmentation_methods\optimal_flow-0.1.11'
sys.path.insert(0, additional_modules_directory)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\segmentation_methods\tpot-0.12.1'
sys.path.insert(0, additional_modules_directory)

#%% CREATE A INSTANCE WITH THE ELEMENT SELECTED
CC = pycc.GetInstance() 
entities= CC.getSelectedEntities()[0]
#%% INPUTS AT THE BEGINING
name_list=get_point_clouds_name()
clustering_types=["K-means","Fuzzy k-means", "OPTICS", "DBSCAN"]
current_directory=os.path.dirname(os.path.abspath(__file__))
output_directory=os.path.join(current_directory,'..','temp_folder_for_results','Machine_Learning','OUTPUT')
features2include=[]
# values_list=[]
# features=[]

# #%% FUNCTIONS OF THE GUI
# fuzzy_k_means_iterations=200



#     # Buttons frame (static)
#     button_frame = tk.Frame(feature_window)
#     button_frame.pack(side="right", fill="y")

#     # Your checkbuttons and variables
#     values_list = [col for col in pcd_training.columns if col != 'Class']
#     checkbuttons_vars = [tk.BooleanVar() for _ in values_list]

#     for value, var in zip(values_list, checkbuttons_vars):
#         ttk.Checkbutton(features_frame, text=value, variable=var, onvalue=True, offvalue=False).pack(anchor="w")

#     select_all_button = ttk.Button(button_frame, text="Select All", command=lambda: select_all_checkbuttons(checkbuttons_vars))
#     select_all_button.pack(side="top", pady=5)

#     def _on_mousewheel(event):
#         canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

#     canvas.bind_all("<MouseWheel>", _on_mousewheel)

#     def ok_features_window():
#         global features2include
#         features2include = [value for value, var in zip(values_list, checkbuttons_vars) if var.get()]
#         if not features2include:
#             print("Please, check at least one feature")
#         else:
#             if len(features2include) == 1:
#                 print("The feature " + str(features2include) + " has been included for the training")
#             else:
#                 print("The features " + str(features2include) + " have been included for the training")

#         feature_window.destroy()

#     def cancel_features_window():
#         feature_window.destroy()

#     ok_button_features = ttk.Button(button_frame, text="OK", command=ok_features_window, width=10)
#     ok_button_features.pack(side="left")
#     cancel_button_features = ttk.Button(button_frame, text="Cancel", command=cancel_features_window, width=10)
#     cancel_button_features.pack(side="right")

#     return features2include
#     print(features2include)
    
# def select_all_checkbuttons(checkbuttons_vars):
#     for var in checkbuttons_vars:
#         var.set(True)
# def create_entry_button_output(tab, row):
#     entry_widget = ttk.Entry(tab, width=30)
#     entry_widget.grid(row=row, column=1, sticky="e", pady=2)
#     entry_widget.insert(0, output_directory)

#     button_widget = ttk.Button(tab, text="...", command=lambda: select_directory([entry_widget]), width=10)
#     button_widget.grid(row=row, column=2, sticky="e", padx=100)
#     return entry_widget    
# def select_directory(entry_widgets):
#     global output_directory
#     directory = filedialog.askdirectory()
#     if directory:
#         output_directory = directory

#         # Update all entry widgets
#         for entry_widget in entry_widgets:
#             entry_widget.delete(0, tk.END)
#             entry_widget.insert(0, output_directory)
            
#         # Change script directory before writting the pickle file
#         os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
#         # Save the last selected folder in a pickle file
#         with open('config.pkl', 'wb') as file:
#             pickle.dump(output_directory, file) 

# def run_algorithm_1():
    
#     # Error if there is not selection
#     if combot1.get()=="Not selected":
#         raise RuntimeError("Please select a point cloud to process the data")
#     if not CC.haveSelection():
#         raise RuntimeError("No folder or entity selected")
#     else:
#         # Load the selected point cloud. If the entity has the attribute points is a folder. Otherwise is a point cloud
#         if hasattr (entities, 'points'):            
#             pc=entities
#         else:            
#             for ii, item in enumerate (name_list):
#                 if item== combot1.get():
#                     pc = entities.getChild(ii)
#                     break 
#         if hasattr(pc, 'points'): # If there is selection and the selected entity is a point
#             pass
#         else:
#             pass
#     # FUNCTIONS TO RUN EACH ALGORITHM   
#     def receive_parameters (parameter_values):
#         print("Received parameter in first window:", parameter_values)    
#     # RUN THE SELECTED ALGORITHM   
#     print (receive_parameters)
#     # SAVE THE FILES IN THE OUTPUTDIRECTORY    
#     # Join the list items with commas to create a comma-separated string
#     comma_separated = ','.join(features2include)    
#     # Write the comma-separated string to a text file
#     with open(os.path.join(output_directory, 'features.txt'), 'w') as file:
#         file.write(comma_separated)  
# def run_algorithm_2():
#     pass
# #%% GUI
# # Create the main window
# 

# # Combobox
# combot1=ttk.Combobox (tab1,values=name_list)
# combot1.grid(column=1, row=0, sticky="e", pady=2)
# combot1.set("Not selected")

# algorithms = ["K-means", "Fuzzy-K-means","DBSCAN","OPTICS"]
# combot2=ttk.Combobox (tab1,values=algorithms, state="readonly")
# combot2.current(0)
# combot2.grid(column=1, row=1, sticky="e", pady=2)
# combot2.set("Not selected")
# # Entry
# entry_a = create_entry_button_output(tab1, 3)
# # Button
# setup_button= ttk.Button (tab1, text="Set-up", command=lambda: show_setup_window(), width=10)
# setup_button.grid (row=1,column=2,sticky="e",padx=100)
# features_button= ttk.Button (tab1, text="...", command=show_features_window, width=10)
# features_button.grid (row=2,column=2,sticky="e",padx=100)

# run_button_p= ttk.Button (tab1, text="OK", command=run_algorithm_1, width=10)
# run_button_p.grid (row=4,column=1,sticky="e",padx=100)
# cancel_button_p= ttk.Button (tab1, text="Cancel", command=destroy,width=10)
# cancel_button_p.grid (row=4,column=1,sticky="e")

# # TAB ALGORITHMS FOR TRAINING

# def show_setup_window():
#     def pass_parameters():
#         parameter_values=entry_param2.get()
#     if combot2.get() == "K-means" or combot2.get() == "Fuzzy-K-means":
#         window = tk.Toplevel(root)
#         window.title("Set Up the algorithm")
            
#         label_param1 = tk.Label(window, text="Number of clusters:")
#         label_param1.grid(row=0, column=0, sticky=tk.W)
       
#         entry_param1 = tk.Entry(window)
#         entry_param1.insert(0,5)
#         entry_param1.grid(row=0, column=1)
       
#         label_param2 = tk.Label(window, text="Maximum number of iterations:")
#         label_param2.grid(row=1, column=0, sticky=tk.W)
       
#         entry_param2 = tk.Entry(window)
#         entry_param2.insert(0,200)
#         entry_param2.grid(row=1, column=1)  
        
#         # Button
#         button = tk.Button(window, text="Pass Parameters", command=pass_parameters)
#         # run_button_p= ttk.Button (tab2, text="OK", command=get_parametes, width=10)
#         # run_button_p.grid (row=4,column=1,sticky="e",padx=100)
#         # cancel_button_p= ttk.Button (tab2, text="Cancel", command=destroy,width=10)
#         # cancel_button_p.grid (row=4,column=1,sticky="e")
        
       
            
# #             frame_clustering=tk.Frame(window)             
#     elif combot2.get() == "DBSCAN":
#         print ("hola")
#     elif combot2.get() == "OPTICS": 
#         print ("hola")
    



# # TAB PREDICTION

# # Labels
# label_p1= ttk.Label(tab2, text="Choose point cloud for prediction")
# label_p1.grid(column=0, row=0, pady=2, sticky="w")
# label_p2= ttk.Label(tab2, text="Load feature file")
# label_p2.grid(column=0, row=1, pady=2, sticky="w")
# label_p3= ttk.Label(tab2, text="Load pkl file")
# label_p3.grid(column=0, row=2, pady=2, sticky="w")
# label_p4= ttk.Label(tab2, text="Choose output directory")
# label_p4.grid(column=0, row=3, pady=2, sticky="w")

# # Button

# run_button_p= ttk.Button (tab2, text="OK", command=run_algorithm_2, width=10)
# run_button_p.grid (row=4,column=1,sticky="e",padx=100)
# cancel_button_p= ttk.Button (tab2, text="Cancel", command=destroy,width=10)
# cancel_button_p.grid (row=4,column=1,sticky="e")



# root.mainloop()

# # def P2p_getdata (pc,nan_value=False,sc=True):
# #         ## CREATE A DATAFRAME WITH THE POINTS OF THE PC
# #        pcd = pd.DataFrame(pc.points(), columns=['X', 'Y', 'Z'])
# #        if (sc==True):       
# #        ## ADD SCALAR FIELD TO THE DATAFRAME
# #            for i in range(pc.getNumberOfScalarFields()):
# #                scalarFieldName = pc.getScalarFieldName(i)  
# #                scalarField = pc.getScalarField(i).asArray()[:]              
# #                pcd.insert(len(pcd.columns), scalarFieldName, scalarField) 
# #        ## DELETE NAN VALUES
# #        if (nan_value==True):
# #            pcd.dropna(inplace=True)
# #        return pcd 
    
# # CC = pycc.GetInstance()
def destroy ():
    root.destroy ()
def run_algorithm_1():
    print ("run_algorithm_1")
class GUI:
    def __init__(self):
        pass
    def main_frame (self, root):
        root = tk.Tk()
        root.title ("Machine Learning Segmentation")
        root.resizable (False, False)
        # Remove minimize and maximize button 
        root.attributes ('-toolwindow',-1)
        
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
        
        # TAB1= TRAINING
        # Combobox
        combot1=ttk.Combobox (tab1,values=name_list)
        combot1.grid(column=1, row=0, sticky="e", pady=2)
        combot1.set("Not selected")
        
        algorithms = ["K-means", "Fuzzy-K-means","DBSCAN","OPTICS"]
        combot2=ttk.Combobox (tab1,values=algorithms, state="readonly")
        combot2.current(0)
        combot2.grid(column=1, row=1, sticky="e", pady=2)
        combot2.set("Not selected")
        # Entry

        
        # features_button= ttk.Button (tab1, text="...", command=show_features_window, width=10)
        # features_button.grid (row=2,column=2,sticky="e",padx=100)
        
        run_button_p= ttk.Button (tab1, text="OK", command=run_algorithm_1(), width=10)
        run_button_p.grid (row=4,column=1,sticky="e",padx=100)
        cancel_button_p= ttk.Button (tab1, text="Cancel", command=destroy(),width=10)
        cancel_button_p.grid (row=4,column=1,sticky="e")      
        
    # Functions for working the gui

    

# START THE MAIN WINDOW        
root = tk.Tk()
app = GUI()
app.main_frame(root)
root.mainloop()




# # TAB TRAINING
# # Labels
# label_t1= ttk.Label(tab1, text="Choose point cloud for training")
# label_t1.grid(column=0, row=0, pady=2, sticky="w")
# label_t2= ttk.Label(tab1, text="Select a clustering algorithm")
# label_t2.grid(column=0, row=1, pady=2, sticky="w")
# label_t3= ttk.Label(tab1, text="Select the features to include")
# label_t3.grid(column=0, row=2, pady=2, sticky="w")
# label_t4= ttk.Label(tab1, text="Choose output directory")
# label_t4.grid(column=0, row=3, pady=2, sticky="w")
#     def destroy ():
#         root.destroy ()
#     def show_features_window():
#         global features2include
    
#         CC = pycc.GetInstance()
#         entities = CC.getSelectedEntities()[0]
    
#         training_pc_name = combot1.get()
    
#         index = -1
#         for ii, item in enumerate(name_list):
#             if item == training_pc_name:
#                 pc_training = entities.getChild(ii)
#                 break
#         if pc_training is None:
#             raise RuntimeError ("Please select a point cloud for the training")
#         else:            
#             pcd_training = P2p_getdata(pc_training, False, True, True)
    
#         feature_window = tk.Toplevel()
#         feature_window.title("Features of the point cloud")
    
#         checkbutton_frame = tk.Frame(feature_window)
#         checkbutton_frame.pack(side="left", fill="y")
    
#         canvas = tk.Canvas(checkbutton_frame)
#         features_frame = tk.Frame(canvas)
    
#         scrollbar = tk.Scrollbar(checkbutton_frame, orient="vertical", command=canvas.yview)
#         scrollbar.pack(side="right", fill="y")
    
#         canvas.configure(yscrollcommand=scrollbar.set)
#         canvas.pack(side="left", fill="both", expand=True)
    
#         canvas.create_window((0, 0), window=features_frame, anchor="nw")











       
# #         entities = CC.getSelectedEntities()
# #         print(f"Selected entities: {entities}")
# #         if not entities:
# #             raise RuntimeError("No entities selected")
# #         else:
# #             pc = entities[0]
# #             self.pcd=P2p_getdata(pc,True,True)
            
        
# #         self.master = master
# #         master.title("Clustering App")
        
# #         # First row
# #         self.label_algo = tk.Label(master, text="Select a clustering algorithm:")
# #         self.label_algo.grid(row=0, column=0, sticky=tk.W)
        
# #         # Define the list of features
# #         self.features2include = []
# #         self.values_list = [col for col in self.pcd.columns if col !='Classification']
# #         self.set_up_parameters = {} 
# #        # First row
        
# #         algorithms = ["K-means", "Fuzzy-K-means","DBSCAN","OPTICS"]
# #         self.combo_algo = ttk.Combobox(master, values=algorithms, state="readonly")
# #         self.combo_algo.current(0)  # set the default value to "Algorithm 1"
# #         self.combo_algo.grid(row=0, column=1, sticky=tk.W)
        
# #         self.button_setup = tk.Button(master, text="Set Up", command=self.show_setup_window)
# #         self.button_setup.grid(row=0, column=2)
        
# #         # Second row
# #         self.label_feat = tk.Label(master, text="Choose the features to include:")
# #         self.label_feat.grid(row=1, column=0, sticky=tk.W)
        
# #         self.button_feat = tk.Button(master, text="Feature", command=self.show_feature_window)
# #         self.button_feat.grid(row=1, column=1)
        
# #         # THIRD ROW    
# #         self.save_path_label = tk.Label(master, text="Ruta de guardado:")
# #         self.save_path_textbox = tk.Entry(master)
# #         self.save_path_button = tk.Button(master, text="...", command=self.select_path)

          
# #         self.save_path_label.grid(row=2, column=0)
# #         self.save_path_textbox.grid(row=2, column=1)  
# #         self.save_path_button.grid(row=2, column=3)  
        
# #         # FITH ROW
# #         self.button_run = tk.Button(master, text="Run", command=self.run_algorithm)
# #         self.button_run.grid(row=3, column=1, pady=10)
        
        
# #     def select_path(self):
# #         # Abrir el diálogo para seleccionar la ruta de guardado
# #         if self.combo_algo.get()=='K-means':
# #             inifile="Kmeans.pkl"
# #         elif self.combo_algo.get()=='Fuzzy-K-means':
# #             inifile="Fuzzy-K-means.pkl"
# #         elif self.combo_algo.get()=='DBSCAN':
# #             inifile="DBSCAN.pkl"           
# #         elif self.combo_algo.get()=='OPTICS':
# #             inifile="OPTICS.pkl"           
# #         path = filedialog.asksaveasfilename(initialfile=inifile)        
# #         # Mostrar la ruta seleccionada en el textbox correspondiente
# #         self.save_path_textbox.delete(0, tk.END)
# #         self.save_path_textbox.insert(0, path)
    
            
    
# #     def show_setup_window(self):
# #         algo = self.combo_algo.get()
# #         def checkbox_changed():
# #             if checkbox_var.get():
# #                 frame_variable.set(True)
# #                 combobox['state'] = 'readonly'
# #                 entry_param3.config(state='normal')
# #                 entry_param4.config(state='normal')
# #             else:
# #                 frame_variable.set(False)
# #                 combobox['state'] = 'disabled'
# #                 entry_param3.config(state='disabled')
# #                 entry_param4.config(state='disabled')      
# #         if algo == "K-means" or algo == "Fuzzy-K-means":

# #             window = tk.Toplevel(self.master)
# #             window.title("Set Up the algorithm")
            
# #             label_param1 = tk.Label(window, text="Number of clusters:")
# #             label_param1.grid(row=0, column=0, sticky=tk.W)
            
# #             entry_param1 = tk.Entry(window)
# #             entry_param1.insert(0,5)
# #             entry_param1.grid(row=0, column=1)
            
# #             label_param2 = tk.Label(window, text="Maximum number of iterations:")
# #             label_param2.grid(row=1, column=0, sticky=tk.W)
            
# #             entry_param2 = tk.Entry(window)
# #             entry_param2.insert(0,200)
# #             entry_param2.grid(row=1, column=1)      
            
# #             # Checkbox for chossing optimization of clusters
# #                 # Frame Variable
# #             frame_variable = tk.BooleanVar()
# #             checkbox_var = tk.BooleanVar()
# #             checkbox = tk.Checkbutton(window, text="Automatic selection of clusters", variable=checkbox_var, command=checkbox_changed)
# #             checkbox.grid(row=2, column=0, sticky=tk.W)
            
# #             frame_clustering=tk.Frame(window)
            
            
# #             # Combobox
# #             combobox_label = tk.Label(frame_clustering, text="Select Option:")
# #             combobox_label.grid(row=0, column=0, sticky=tk.W)
# #             options = ["Elbow Method"]
# #             combobox = ttk.Combobox(frame_clustering, values=options, state='disabled')
# #             combobox.current(0)
# #             combobox.grid(row=0, column=1)
            
# #             # Labels with Entries
# #             label_param3 = tk.Label(frame_clustering, text="Minimum number of clusters")
# #             label_param3.grid(row=1, column=0, sticky=tk.W)
# #             entry_param3 = tk.Entry(frame_clustering, state='disable')
# #             entry_param3.grid(row=1, column=1)
            
# #             label_param4 = tk.Label(frame_clustering, text="Maximum number of clusters:")
# #             label_param4.grid(row=2, column=0, sticky=tk.W)
# #             entry_param4 = tk.Entry(frame_clustering, state='disabled')
# #             entry_param4.grid(row=2, column=1, sticky=tk.W) 
# #             frame_clustering.grid(row=3, column=0)
            
        
# #             button_ok = tk.Button(window, text="OK", command=lambda: self.save_setup_parameters(self.set_up_parameters, entry_param1.get(), entry_param2.get(),frame_variable.get(),combobox.get(),entry_param3.get(),entry_param4.get(), window))
# #             button_ok.grid(row=2, column=1)    
                           
# #         elif algo =="DBSCAN":
            
# #             window = tk.Toplevel(self.master)
# #             window.title("Set Up the algorithm")
            
# #             label_param1 = tk.Label(window, text="Epsilon (maximum distance between points of a cluster):")
# #             label_param1.grid(row=0, column=0, sticky=tk.W)
            
# #             entry_param1 = tk.Entry(window)
# #             entry_param1.insert(0,5)
# #             entry_param1.grid(row=0, column=1)
            
# #             label_param2 = tk.Label(window, text="Mimimum number of points to create a cluster:")
# #             label_param2.grid(row=1, column=0, sticky=tk.W)
            
# #             entry_param2 = tk.Entry(window)
# #             entry_param2.insert(0,200)
# #             entry_param2.grid(row=1, column=1)             
            
# #             button_ok = tk.Button(window, text="OK", command=lambda: self.save_setup_parameters(self.set_up_parameters, entry_param1.get(), entry_param2.get(),False,'None',0,0, window))
# #             button_ok.grid(row=2, column=1) 
            
# #         elif algo =="OPTICS":
            
# #             window = tk.Toplevel(self.master)
# #             window.title("Set Up the algorithm")
            
# #             label_param1 = tk.Label(window, text="Number of samples in a neighborhood to be considered as cluster:")
# #             label_param1.grid(row=0, column=0, sticky=tk.W)
            
# #             entry_param1 = tk.Entry(window)
# #             entry_param1.insert(0,5)
# #             entry_param1.grid(row=0, column=1)
            
# #             label_param2 = tk.Label(window, text="Epsilon (maximum distance between poins of a cluster):")
# #             label_param2.grid(row=1, column=0, sticky=tk.W)
            
# #             entry_param2 = tk.Entry(window)
# #             entry_param2.insert(0,200)
# #             entry_param2.grid(row=1, column=1) 


# #             label_param3 = tk.Label(window, text="Metric for distance computation:")
# #             label_param3.grid(row=2, column=0, sticky=tk.W)
# #             features_1 = ["minkowski", "cityblock","cosine","euclidean", "l1","l2","manhattan", "braycurtis","canberra","chebyshev", "correlation","dice","hamming", "jaccard","kulsinski","mahalanobis", "rogerstanimoto","russellrao","seuclidean", "sokalmichener","sokalsneath","sqeuclidean","yule"]
# #             entry_param3 = ttk.Combobox(window,values=features_1, state="readonly")
# #             entry_param3.current(0) 
# #             entry_param3.grid(row=2, column=1) 

# #             label_param4 = tk.Label(window, text="Extraction method:")
# #             label_param4.grid(row=3, column=0, sticky=tk.W)
# #             features_2 = ["xi","dbscan"]
# #             entry_param4 = ttk.Combobox(window,values=features_2, state="readonly")
# #             entry_param4.current(0) 
# #             entry_param4.grid(row=3, column=1) 


# #             label_param5 = tk.Label(window, text="Minimum steepness:")
# #             label_param5.grid(row=4, column=0, sticky=tk.W)
            
# #             entry_param5 = tk.Entry(window)
# #             entry_param5.insert(0,0.05)
# #             entry_param5.grid(row=4, column=1)

# #             label_param6 = tk.Label(window, text="Minimum cluster size:")
# #             label_param6.grid(row=5, column=0, sticky=tk.W)
            
# #             entry_param6 = tk.Entry(window)
# #             entry_param6.insert(0,10)
# #             entry_param6.grid(row=5, column=1)
            
# #             button_ok = tk.Button(window, text="OK", command=lambda: self.save_setup_parameters(self.set_up_parameters, entry_param1.get(), entry_param2.get(), entry_param3.get(), entry_param4.get(), entry_param5.get(), entry_param6.get(), window))
# #             button_ok.grid(row=6, column=1)                
           
# #     def show_feature_window(self):
# #         feature_window = tk.Toplevel(self.master)
# #         feature_window.title("Features of the point cloud")       
        
        
# #         canvas = tk.Canvas(feature_window)
# #         features_frame = tk.Frame(canvas)
        
# #         self.features2include.clear()
# #         for value in self.values_list:
# #             checked_var = tk.BooleanVar()
# #             ttk.Checkbutton(features_frame, text=value, variable=checked_var, onvalue=True, offvalue=False).pack(anchor="w")
# #             checked_var.trace_add("write", lambda var, indx, mode, checked_var=checked_var, value=value: self.on_checkbox_checked(checked_var, value))
        
# #         # Agrega la barra de desplazamiento
# #         scrollbar = tk.Scrollbar(feature_window, orient="vertical", command=canvas.yview)
# #         canvas.configure(yscrollcommand=scrollbar.set)
        
# #         scrollbar.config(command = self.values_list)
# #         scrollbar.pack(side="right", fill="y")
# #         canvas.pack(side="left", fill="both", expand=True)
        
# #         canvas.create_window((0, 0), window=features_frame, anchor="nw")

        
# #         # Configura el desplazamiento del canvas al hacer scroll
# #         def _on_mousewheel(event):
# #             canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
# #         canvas.bind_all("<MouseWheel>", _on_mousewheel) 
        
# #     def show_button(self):
# #         print ('The features selected are:',self.features2include)
        

         
# #     def on_checkbox_checked(self, checked_var, value):
# #         if checked_var.get() and value not in self.features2include:
# #             self.features2include.append(value)
# #         elif not checked_var.get() and value in self.features2include:
# #             self.features2include.remove(value)
# #     def save_setup_parameters(self,set_up_parameters, param1, param2,param3,param4,param5,param6, window):
# #         algo = self.combo_algo.get()
# #         if algo == 'K-means' or algo == 'Fuzzy-K-means':
# #             set_up_parameters['clusters'] =int(param1)
# #             set_up_parameters['max_iterations'] =int(param2)
# #             set_up_parameters['optimization'] =param3
# #             set_up_parameters['method'] =param4
# #             if param3==True:
# #                 set_up_parameters['min_clusters'] =int(param5)
# #                 set_up_parameters['max_clusters'] =int(param6)   
                
# #         elif algo == 'DBSCAN':
# #             set_up_parameters['epsilon'] =float(param1)
# #             set_up_parameters['min_points'] =int(param2)         
        
          
# #         elif algo == 'OPTICS':
# #             set_up_parameters['min_samples'] =int(param1)
# #             set_up_parameters['max_eps'] =float(param2)
# #             set_up_parameters['metric'] =str(param3)  
# #             set_up_parameters['cluster_method'] =str(param4)  
# #             set_up_parameters['xi'] =float(param5)  
# #             set_up_parameters['min_cluster_size'] =int(param6)  
# #         print ('The algorithm chosen is a ' + str (algo) + ', with the following set-up parameters: ' + str(set_up_parameters))
# #         window.destroy()                    
# #     def run_algorithm(self):
# #         ## PREPARING THE ALGORITHM FOR RUNNING 
# #         X_train=self.pcd[self.features2include].to_numpy() 
        
        
        
        
# #         ##OPTIONS FOR K-MEANS AND FUZZY K-MEANS
# #         if self.combo_algo.get()=='K-means' or self.combo_algo.get()=='Fuzzy-K-means':
# #             if self.set_up_parameters['optimization']==True:
# #                 if self.set_up_parameters['method']=='Elbow Method':
# #                    if self.combo_algo.get()=='K-means': 
# #                         # Create K-means model
# #                         kmeans = KMeans()                    
# #                         # Instantiate the KElbowVisualizer with the K-means model
# #                         k=(self.set_up_parameters['min_clusters'],self.set_up_parameters['max_clusters']+1)
# #                         visualizer = KElbowVisualizer(kmeans, k=k)
# #                         visualizer.fit(X_train)

                        
# #                         # Final model with optimal values
# #                         kmeans = KMeans(n_clusters=visualizer.elbow_value_, max_iter=self.set_up_parameters['max_iterations'])
# #                         kmeans.fit(X_train)
# #                         config_algo=kmeans                 
# #                         clusters=kmeans.labels_
# #                    elif self.combo_algo.get()=='Fuzzy-K-means':
# #                         # Create Fuzzy-K-means model 
# #                         fcm = KMeans ()
# #                         # Instantiate the KElbowVisualizer with the K-means model
# #                         k=(self.set_up_parameters['min_clusters'],self.set_up_parameters['max_clusters']+1)
# #                         visualizer = KElbowVisualizer(fcm, k=k)
# #                         visualizer.fit(X_train)
                        
                        
# #                         # Final model with optimal values
# #                         fcm = FCM(n_clusters=visualizer.elbow_value_, max_iter=self.set_up_parameters['max_iterations'])
# #                         fcm.fit(X_train)
# #                         config_algo=fcm                 
# #                         clusters=fcm.u.argmax(axis=1)
                        
# #                    visualizer.show()
# #                    print(f"Optimal number of clusters (Elbow method): {visualizer.elbow_value_}")
                   
# #             else: #No optimization
            
# #                 if self.combo_algo.get()=='K-means':
                    
# #                     kmeans = KMeans(n_clusters=self.set_up_parameters['clusters'], max_iter=self.set_up_parameters['max_iterations'])
# #                     kmeans.fit(X_train)
# #                     #Optimal values
# #                     config_algo=kmeans                 
# #                     clusters=kmeans.labels_
# #                 elif self.combo_algo.get()=='Fuzzy-K-means':
# #                     fcm = FCM (n_clusters=self.set_up_parameters['clusters'], max_iter=self.set_up_parameters['max_iterations'])
# #                     fcm.fit(X_train)
# #                     #Optimal values
# #                     config_algo=fcm                 
# #                     clusters=fcm.u.argmax(axis=1)
# #         elif self.combo_algo.get()== "DBSCAN":
# #             dbscan = DBSCAN(eps=self.set_up_parameters['epsilon'],min_samples=self.set_up_parameters['min_points'])
# #             dbscan.fit(X_train) 
# #             clusters=dbscan.labels_
# #             config_algo=dbscan
            
# #         elif self.combo_algo.get()== "OPTICS":
# #             optics = OPTICS(min_samples=self.set_up_parameters['min_samples'],max_eps=self.set_up_parameters['max_eps'],metric=self.set_up_parameters['metric'],cluster_method=self.set_up_parameters['cluster_method'],xi=self.set_up_parameters['xi'],min_cluster_size=self.set_up_parameters['min_cluster_size'])
# #             optics.fit(X_train) 
# #             clusters=optics.labels_
# #             config_algo=optics
            
# #         ## CREATE THE RESULTING POINT CLOUD 
# #         pc_results_test = pycc.ccPointCloud(self.pcd['X'], self.pcd['Y'], self.pcd['Z'])
# #         pc_results_test.setName("Results_from_clustering")
# #         idx = pc_results_test.addScalarField("Clusters",clusters)    
         
# #         # # SAVE THE RESULTS OF THE UNSUPERVISED CLUSTERING METHOD
# #         save_path = self.save_path_textbox.get()     
# #         joblib.dump(config_algo,save_path) 
# #         feature_path = os.path.join(os.path.dirname(save_path), 'features2include.txt')
# #         with open(feature_path, 'w') as f:
# #             f.write('Features included: {}\n'.format(str(self.features2include)))                    
# #         ## STORE IN THE DATABASE OF CLOUDCOMPARE
# #         CC.addToDB(pc_results_test)
# #         CC.updateUI() 
        
# #         print('The clustered point cloud has been loaded to the DB')
# # root = tk.Tk()
# # app = App(root)
# # root.mainloop()