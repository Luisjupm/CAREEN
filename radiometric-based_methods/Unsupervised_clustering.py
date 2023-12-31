# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 18:05:13 2023

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
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import ward, dendrogram
import pickle
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import InterclusterDistance
import matplotlib.pyplot as plt
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.features import JointPlotVisualizer
from scipy.cluster.hierarchy import dendrogram 
import sys
import os
import numpy as np
import itertools
import traceback
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from yellowbrick.style import set_palette
import cccorelib
import pycc
#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS

script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'
additional_modules_directory_2=script_directory
sys.path.insert(0, additional_modules_directory)
sys.path.insert(0, additional_modules_directory_2)
from main import P2p_getdata,get_istance,get_point_clouds_name
print (additional_modules_directory_2)

#%% INPUTS AT THE BEGINING
name_list=get_point_clouds_name()
current_directory=os.path.dirname(os.path.abspath(__file__))
output_directory=os.path.join(current_directory,'..','temp_folder_for_results','Machine_Learning','OUTPUT')
#%% GUI
class GUI:
    def __init__(self):
        # Initial features2include
        self.features2include=[]
        # Initial paramertes for the different algorithms
        # K-means
        self.set_up_parameters_km= (5,200)
        # Fuzzy k-means
        self.set_up_parameters_fkm= (5,200)
        # Hierarchical clustering
        self.set_up_parameters_hc= (2,"euclidean",None,"ward",0.1,False,"adddduto")            
        #DBSCAN
        self.set_up_parameters_dbscan= (5,200)
        #OPTICS
        self.set_up_parameters_optics=(5,200,"minkowski","xi",0.05,10)
        # Variable to save if we want a cluster optimization or not
        self.optimization_strategy= (0,0,0) # 0 not optimization strategy, 1 elbow method, 2 silhouette method, 3 Calinski-Harabasz index, 4 Davies-Bouldin index,
        #Directoy to save the files (training)
        self.file_path=os.path.join(current_directory, output_directory)
        #Directory to load the features file (prediction)
        self.load_features=os.path.join(current_directory, output_directory)
        #Directory to load the configuration file (prediction)
        self.load_configuration=os.path.join(current_directory, output_directory)
        #List with the features loaded (prediction)
        self.features_prediction=[]
    def main_frame (self, root):
        root.title ("Unsupervised clustering")
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
        
        # Labels            
        label_pc_training = tk.Label(tab1, text="Choose point cloud for training:")
        label_pc_training.grid(row=0, column=0, sticky=tk.W)
        label_algo = tk.Label(tab1, text="Select a clustereing algorithm:")
        label_algo.grid(row=1, column=0, sticky=tk.W) 
        label_fea = tk.Label(tab1, text="Select the features to include:")
        label_fea.grid(row=2, column=0, sticky=tk.W) 
        label_out= ttk.Label(tab1, text="Choose output directory:")
        label_out.grid(row=3, column=0, sticky=tk.W)
        
        # Combobox
        combot1=ttk.Combobox (tab1,values=name_list)
        combot1.grid(row=0,column=1, sticky="e", pady=2)
        combot1.set("Not selected")
        
        algorithms = ["K-means", "Fuzzy-K-means","Hierarchical-clustering","DBSCAN","OPTICS"]
        combot2=ttk.Combobox (tab1,values=algorithms, state="readonly")
        combot2.current(0)
        combot2.grid(column=1, row=1, sticky="e", pady=2)
        combot2.set("Not selected")
        
        # Entry
        self.entry_widget = ttk.Entry(tab1, width=30)
        self.entry_widget.grid(row=3, column=1, sticky="e", pady=2)
        self.entry_widget.insert(0, self.file_path)
        
        # Button
        t1_setup_button= ttk.Button (tab1, text="Set-up", command=lambda: self.show_set_up_window(combot2.get()), width=10)
        t1_setup_button.grid (row=1,column=2,sticky="e",padx=100)
        t1_features= ttk.Button (tab1, text="...", command=lambda: self.show_features_window(combot1.get()), width=10)
        t1_features.grid (row=2,column=2,sticky="e",padx=100)
        t1_button_widget = ttk.Button(tab1, text="...", command=self.save_file_dialog, width=10)
        t1_button_widget.grid(row=3, column=2, sticky="e", padx=100)
        t1_run_button= ttk.Button (tab1, text="OK", command=lambda:self.run_algorithm_1(combot2.get(),combot1.get()), width=10)
        t1_run_button.grid (row=4,column=1,sticky="e",padx=100)
        t1_cancel_button= ttk.Button (tab1, text="Cancel", command=self.destroy,width=10)
        t1_cancel_button.grid (row=4,column=1,sticky="e")
        
        # TAB2= PREDICTION
        # Labels            
        label_pc_prediction = tk.Label(tab2, text="Choose point cloud for prediction:")
        label_pc_prediction.grid(row=0, column=0, sticky=tk.W)
        label_features = tk.Label(tab2, text="Load feature file:")
        label_features.grid(row=1, column=0, sticky=tk.W) 
        label_configuration = tk.Label(tab2, text="Load configuration file:")
        label_configuration.grid(row=2, column=0, sticky=tk.W)
        label_out= ttk.Label(tab2, text="Choose output directory:")
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
        
    def save_setup_parameters (self,algo,*params):
        if algo=="K-means":
            self.set_up_parameters_km=params
        elif algo=="Fuzzy-K-means":
            self.set_up_parameters_fkm=params
        elif algo=="Hierarchical-clustering":
            self.set_up_parameters_hc=params
        elif algo=="DBSCAN":
            self.set_up_parameters_dbscan=params  
        elif algo=="OPTICS":
            self.set_up_parameters_optics=params   
      
    
    def show_set_up_window (self,algo):
        def on_ok_button_click(algo):
            if algo=="K-means":
                self.save_setup_parameters(algo, int(entry_param1_km.get()), int(entry_param2_km.get()))  
            elif algo=="Fuzzy-K-means":
                self.save_setup_parameters(algo, int(entry_param1_fkm.get()), int(entry_param2_fkm.get()))
            elif algo=="Hierarchical-clustering":
                self.save_setup_parameters(algo, int(entry_param1_hc.get()), str(entry_param2_hc.get()),str(entry_param3_hc.get()),str(entry_param4_hc.get()),float(entry_param5_hc.get()),entry_param6_hc.get(),"auto")                         
            elif algo=="DBSCAN":
                self.save_setup_parameters(algo, float(entry_param1_dbscan.get()), int(entry_param2_dbscan.get()))
            elif algo=="OPTICS":
                self.save_setup_parameters(algo, int(entry_param1_optics.get()), float(entry_param2_optics.get()), str(entry_param3_optics.get()), str(entry_param4_optics.get()), float(entry_param5_optics.get()),int(entry_param6_optics.get()))                
            
            if algo=="K-means" or algo=="Fuzzy-K-means": 
                if entry_opt.get()=="Elbow method" and var1.get()==1: # Elbow method is selected
                    self.optimization_strategy= (1,int(entry_max_clusters.get()),int(entry_min_clusters.get()))
                elif entry_opt.get()=="Silhouette coefficient" and var1.get()==1: # Silhouette coefficient is selected
                    self.optimization_strategy= (2,int(entry_max_clusters.get()),int(entry_min_clusters.get()))
                elif entry_opt.get()=="Calinski-Harabasz-index" and var1.get()==1: # Calinski-Harabasz-index is selected
                    self.optimization_strategy= (3,int(entry_max_clusters.get()),int(entry_min_clusters.get()))
                elif entry_opt.get()=="Davies-Bouldin-index" and var1.get()==1: # Davies-Bouldin-index is selected
                    self.optimization_strategy= (4,int(entry_max_clusters.get()),int(entry_min_clusters.get()))
                else:
                    self.optimization_strategy= (0,int(entry_max_clusters.get()),int(entry_min_clusters.get()))
            
            set_up_window.destroy()  # Close the window after saving parameters
            
        set_up_window = tk.Toplevel(root)
        set_up_window.title("Set Up the algorithm")
        set_up_window.resizable (False, False)
        # Remove minimize and maximize button 
        set_up_window.attributes ('-toolwindow',-1)
        def toggle_row():
            if var1.get() == 1:
                label_max_clusters.config(state=tk.NORMAL)
                entry_max_clusters.config(state=tk.NORMAL)
                label_min_clusters.config(state=tk.NORMAL)
                entry_min_clusters.config(state=tk.NORMAL)
                label_opt.config(state=tk.NORMAL)
                entry_opt.config(state=tk.NORMAL)
            else:
                label_max_clusters.config(state=tk.DISABLED)
                entry_max_clusters.config(state=tk.DISABLED)
                label_min_clusters.config(state=tk.DISABLED)
                entry_min_clusters.config(state=tk.DISABLED)
                label_opt.config(state=tk.DISABLED)
                entry_opt.config(state=tk.DISABLED)
        def check_uncheck_1():
            var1.set(1)

           
        if algo=="K-means":
             
            # Labels            
            label_param1_km = tk.Label(set_up_window, text="Number of clusters:")
            label_param1_km.grid(row=0, column=0, sticky=tk.W)
            label_param2_km = tk.Label(set_up_window, text="Number of iterations:")
            label_param2_km.grid(row=1, column=0, sticky=tk.W) 
            label_max_clusters = tk.Label(set_up_window, text="Maximum number of clusters:")
            label_max_clusters.grid(row=4, column=0, sticky=tk.W)
            label_max_clusters.config(state=tk.DISABLED)
            label_min_clusters = tk.Label(set_up_window, text="Minimum number of clusters:")
            label_min_clusters.grid(row=5, column=0, sticky=tk.W)   
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
            entry_max_clusters.grid(row=4, column=1, sticky=tk.W)
            entry_max_clusters.config(state=tk.DISABLED)
            entry_min_clusters= tk.Entry(set_up_window)
            entry_min_clusters.insert(0,1)
            entry_min_clusters.config(state=tk.DISABLED)
            entry_min_clusters.grid(row=5,column=1, sticky=tk.W)
            # Checkbox
            var1 = tk.IntVar()
            checkbox1 = tk.Checkbutton(set_up_window, text="Optimize the number of clusters", variable=var1, command=lambda: [check_uncheck_1(),toggle_row()])
            checkbox1.grid(row=2, column=0, sticky=tk.W)
            # Entries
            label_opt = tk.Label(set_up_window, text="Optimization strategy:")
            label_opt.grid(row=3, column=0, sticky=tk.W)
            label_opt.config(state=tk.DISABLED)
            features_opt = ["Elbow method", "Silhouette coefficient","Calinski-Harabasz-index","Davies-Bouldin-index"]
            entry_opt = ttk.Combobox(set_up_window,values=features_opt, state="readonly")
            entry_opt.current(0) 
            entry_opt.grid(row=3, column=1) 
            entry_opt.config(state=tk.DISABLED)


            # Buttons
            button_ok = tk.Button(set_up_window, text="OK", command=lambda: on_ok_button_click(algo))
            button_ok.grid(row=6, column=1)
        elif algo=="Fuzzy-K-means":
            # Labels            
            label_param1_fkm = tk.Label(set_up_window, text="Number of clusters:")
            label_param1_fkm.grid(row=0, column=0, sticky=tk.W)
            label_param2_fkm = tk.Label(set_up_window, text="Number of iterations:")
            label_param2_fkm.grid(row=1, column=0, sticky=tk.W) 
            label_max_clusters = tk.Label(set_up_window, text="Maximum number of clusters:")
            label_max_clusters.grid(row=4, column=0, sticky=tk.W)
            label_max_clusters.config(state=tk.DISABLED)
            label_min_clusters = tk.Label(set_up_window, text="Minimum number of clusters:")
            label_min_clusters.grid(row=5, column=0, sticky=tk.W)   
            label_min_clusters.config(state=tk.DISABLED)
            # Entries
            entry_param1_fkm= tk.Entry(set_up_window)
            entry_param1_fkm.insert(0,self.set_up_parameters_km[0])
            entry_param1_fkm.grid(row=0, column=1, sticky=tk.W) 
            
            entry_param2_fkm= tk.Entry(set_up_window)
            entry_param2_fkm.insert(0,self.set_up_parameters_km[1])
            entry_param2_fkm.grid(row=1, column=1, sticky=tk.W)
            
            entry_max_clusters= tk.Entry(set_up_window)
            entry_max_clusters.insert(0,10)
            entry_max_clusters.grid(row=4, column=1, sticky=tk.W)
            entry_max_clusters.config(state=tk.DISABLED)
            entry_min_clusters= tk.Entry(set_up_window)
            entry_min_clusters.insert(0,1)
            entry_min_clusters.config(state=tk.DISABLED)
            entry_min_clusters.grid(row=5,column=1, sticky=tk.W)
            # Checkbox
            var1 = tk.IntVar()
            checkbox1 = tk.Checkbutton(set_up_window, text="Optimize the number of clusters", variable=var1, command=lambda: [check_uncheck_1(),toggle_row()])
            checkbox1.grid(row=2, column=0, sticky=tk.W)
            # Entries
            label_opt = tk.Label(set_up_window, text="Optimization strategy:")
            label_opt.grid(row=3, column=0, sticky=tk.W)
            label_opt.config(state=tk.DISABLED)
            features_opt = ["Elbow method", "Silhouette coefficient","Calinski-Harabasz-index","Davies-Bouldin-index"]
            entry_opt = ttk.Combobox(set_up_window,values=features_opt, state="readonly")
            entry_opt.current(0) 
            entry_opt.grid(row=3, column=1) 
            entry_opt.config(state=tk.DISABLED)

            # Buttons
            button_ok = tk.Button(set_up_window, text="OK", command=lambda: on_ok_button_click(algo))
            button_ok.grid(row=6, column=1)      
        elif algo=="Hierarchical-clustering":
            label_param1_hc = tk.Label(set_up_window, text="Number of clusters:")
            label_param1_hc.grid(row=0, column=0, sticky=tk.W)            
            entry_param1_hc = tk.Entry(set_up_window)
            entry_param1_hc.insert(0,self.set_up_parameters_hc[0])
            entry_param1_hc.grid(row=0, column=1)
            
            label_param2_hc = tk.Label(set_up_window, text="Metric for calculating the distance between istances:")
            label_param2_hc.grid(row=1, column=0, sticky=tk.W)            
            features_param2_hc = ["euclidean", "cityblock","cosine","l1","l2","manhattan"]
            entry_param2_hc = ttk.Combobox(set_up_window,values=features_param2_hc, state="readonly")
            entry_param2_hc.current(0) 
            entry_param2_hc.grid(row=1, column=1) 
            
            label_param3_hc = tk.Label(set_up_window, text="Metric for calculating the distance between istances:")
            label_param3_hc.grid(row=2, column=0, sticky=tk.W)            
            features_param3_hc = ["none","euclidean","l1","l2","manhattan","cosine"]
            entry_param3_hc = ttk.Combobox(set_up_window,values=features_param3_hc, state="readonly")
            entry_param3_hc.current(0) 
            entry_param3_hc.grid(row=2, column=1)        
            
            label_param4_hc = tk.Label(set_up_window, text="Linkage criterion:")
            label_param4_hc.grid(row=3, column=0, sticky=tk.W)            
            features_param4_hc = ["ward","complete","average","single"]
            entry_param4_hc = ttk.Combobox(set_up_window,values=features_param4_hc, state="readonly")
            entry_param4_hc.current(0) 
            entry_param4_hc.grid(row=3, column=1)
            
            label_param5_hc = tk.Label(set_up_window, text="Linkage distance threshold:")
            label_param5_hc.grid(row=4, column=0, sticky=tk.W)            
            entry_param5_hc = tk.Entry(set_up_window)
            entry_param5_hc.insert(0,self.set_up_parameters_hc[4])
            entry_param5_hc.grid(row=4, column=1)
            
            label_param6_hc = tk.Label(set_up_window, text="Compute distance between clusters:")
            label_param6_hc.grid(row=5, column=0, sticky=tk.W)            
            features_param6_hc = ["false","true"]
            entry_param6_hc = ttk.Combobox(set_up_window,values=features_param6_hc, state="readonly")
            entry_param6_hc.current(0) 
            entry_param6_hc.grid(row=5, column=1)    
                
            # Buttons
            button_ok = tk.Button(set_up_window, text="OK", command=lambda: on_ok_button_click(algo))
            button_ok.grid(row=6, column=1)              
        elif algo=="DBSCAN":
           
            label_param1_dbscan = tk.Label(set_up_window, text="Epsilon (maximum distance between points of a cluster):")
            label_param1_dbscan.grid(row=0, column=0, sticky=tk.W)
            
            entry_param1_dbscan = tk.Entry(set_up_window)
            entry_param1_dbscan.insert(0,self.set_up_parameters_dbscan[0])
            entry_param1_dbscan.grid(row=0, column=1)
            
            label_param2_dbscan = tk.Label(set_up_window, text="Mimimum number of points to create a cluster:")
            label_param2_dbscan.grid(row=1, column=0, sticky=tk.W)
            
            entry_param2_dbscan = tk.Entry(set_up_window)
            entry_param2_dbscan.insert(0,self.set_up_parameters_dbscan[1])
            entry_param2_dbscan.grid(row=1, column=1)    
            # Buttons
            button_ok = tk.Button(set_up_window, text="OK", command=lambda: on_ok_button_click(algo))
            button_ok.grid(row=2, column=1)
        elif algo=="OPTICS":
            label_param1_optics = tk.Label(set_up_window, text="Number of samples in a neighborhood to be considered as cluster:")
            label_param1_optics.grid(row=0, column=0, sticky=tk.W)
            
            entry_param1_optics = tk.Entry(set_up_window)
            entry_param1_optics.insert(0,self.set_up_parameters_optics[0])
            entry_param1_optics.grid(row=0, column=1)
            
            label_param2_optics = tk.Label(set_up_window, text="Epsilon (maximum distance between poins of a cluster):")
            label_param2_optics.grid(row=1, column=0, sticky=tk.W)
            
            entry_param2_optics = tk.Entry(set_up_window)
            entry_param2_optics.insert(0,self.set_up_parameters_optics[1])
            entry_param2_optics.grid(row=1, column=1) 


            label_param3_optics = tk.Label(set_up_window, text="Metric for distance computation:")
            label_param3_optics.grid(row=2, column=0, sticky=tk.W)
            features_optics = ["minkowski", "cityblock","cosine","euclidean", "l1","l2","manhattan", "braycurtis","canberra","chebyshev", "correlation","dice","hamming", "jaccard","kulsinski","mahalanobis", "rogerstanimoto","russellrao","seuclidean", "sokalmichener","sokalsneath","sqeuclidean","yule"]
            entry_param3_optics = ttk.Combobox(set_up_window,values=features_optics, state="readonly")
            entry_param3_optics.current(0) 
            entry_param3_optics.grid(row=2, column=1) 

            label_param4_optics = tk.Label(set_up_window, text="Extraction method:")
            label_param4_optics.grid(row=3, column=0, sticky=tk.W)
            features_2_optics = ["xi","dbscan"]
            entry_param4_optics = ttk.Combobox(set_up_window,values=features_2_optics, state="readonly")
            entry_param4_optics.current(0) 
            entry_param4_optics.grid(row=3, column=1) 


            label_param5_optics = tk.Label(set_up_window, text="Minimum steepness:")
            label_param5_optics.grid(row=4, column=0, sticky=tk.W)
            
            entry_param5_optics = tk.Entry(set_up_window)
            entry_param5_optics.insert(0,self.set_up_parameters_optics[4])
            entry_param5_optics.grid(row=4, column=1)

            label_param6_optics = tk.Label(set_up_window, text="Minimum cluster size:")
            label_param6_optics.grid(row=5, column=0, sticky=tk.W)
            
            entry_param6_optics = tk.Entry(set_up_window)
            entry_param6_optics.insert(0,self.set_up_parameters_optics[5])
            entry_param6_optics.grid(row=5, column=1)
            # Buttons
            button_ok = tk.Button(set_up_window, text="OK", command=lambda: on_ok_button_click(algo))
            button_ok.grid(row=6, column=1)                        


    def select_all_checkbuttons(self,checkbuttons_vars):
        for var in checkbuttons_vars:
            var.set(True)   
            
    def show_features_window(self,training_pc_name):
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
   
    def destroy (self):
        root.destroy ()

    def save_file_dialog(self):
        
        self.file_path = filedialog.askdirectory()
        self.entry_widget.insert(0, self.file_path) # Insert the new value
    def load_features_dialog(self):
        self.load_features= filedialog.askopenfilename(filetypes=[("Feature file", "*.txt")])
        if self.load_features:
            with open(self.load_features, 'r') as file:
                content = file.read()
                self.features_prediction = content.split(',')  # Splitting data by commas to create a list
            print ("The features file has been loaded")
    def load_configuration_dialog(self):
        self.load_configuration= filedialog.askopenfilename(filetypes=[("Configuration file", "*.pkl")])
        if self.load_configuration:
            with open(self.load_configuration, 'rb') as file:
                self.configuration_prediction = pickle.load(file) 
            print ("The configuration file has been loaded")   
    def run_algorithm_1 (self,algo,training_pc_name): 
        
        # Error to prevent the abscene of point cloud
        CC = pycc.GetInstance() 
        type_data, number = get_istance()
        if type_data=='point_cloud' or type_data=='folder':
            pass
        else:
            raise RuntimeError("Please select a folder that contains points clouds or a point cloud")        
        if number==0:
            raise RuntimeError("There are not entities in the folder")
        else:
            entities = CC.getSelectedEntities()[0]
            number = entities.getChildrenNumber()
            
        if type_data=='point_cloud':
            pc_training=entities
        else:
            for ii, item in enumerate(name_list):
                if item == training_pc_name:
                    pc_training = entities.getChild(ii)
                    break
        pcd=P2p_getdata(pc_training,False,True,True)
        pcd_f=pcd[self.features2include].values 
        

        # Error control to prevent not algorithm for the training
        if algo=="Not selected":
            raise RuntimeError ("Please select an algorithm for the training")
        elif algo=="K-means":
            if self.optimization_strategy[0] != 0: # We want an optimization of clusters
                # Choosing a range of K values for KMeans
                if self.optimization_strategy[2]<=2: #It is neccesary to perform the methods with at least 2 cluster. At exception of the elbow
                    minimum_clusters=2
                elif self.optimization_strategy[2]<=1 and self.optimization_strategy[0]==1: # It is possible to perform the Elbow with one cluster
                    minimum_clusters=1
                else: 
                    minimum_clusters=self.optimization_strategy[2]
                if self.optimization_strategy[1]+1<=minimum_clusters:
                    maximum_clusters=minimum_clusters+1
                else:
                    maximum_clusters=self.optimization_strategy[1]+1
                k_values = range(minimum_clusters, maximum_clusters)
                model = KMeans(n_init='auto')
            if self.optimization_strategy[0]==1: #Perform the elbow method             
                elbow = KElbowVisualizer(model, k=k_values, timings=False)
                elbow.fit(pcd_f) 
                # Get the optimal number of clusters
                optimal_k = elbow.elbow_value_
                # Fit KMeans with the optimal number of clusters
                kmeans = KMeans(n_clusters=optimal_k,n_init='auto', max_iter=self.set_up_parameters_km[1])
                kmeans.fit(pcd_f) 
                # Visualize and save the elbow plot
                elbow.show(outpath=os.path.join(self.file_path, 'optimal_cluster_elbow.png'))
                # Close the figure
                plt.close()
            elif self.optimization_strategy[0]==2: #Perform the sillouethe method    
                silhouette = KElbowVisualizer(model, k=k_values, timings=False, metric='silhouette')
                silhouette.fit(pcd_f)
                # Get the optimal number of clusters
                optimal_k = silhouette.elbow_value_
                # Fit KMeans with the optimal number of clusters
                kmeans = KMeans(n_clusters=optimal_k,n_init='auto', max_iter=self.set_up_parameters_km[1])
                kmeans.fit(pcd_f) 
                # Visualize and save the elbow plot
                silhouette.show(outpath=os.path.join(self.file_path, 'optimal_cluster_sihouette.png'))
                # Close the figure
                plt.close()
            elif self.optimization_strategy[0]==3: #Perform the Calinski-Harabasz index
                ch = KElbowVisualizer(model, k=k_values,metric='calinski_harabasz', timings= False)
                ch.fit(pcd_f)        # Fit the data to the visualizer
                # Get the optimal number of clusters
                optimal_k = ch.elbow_value_
                # Fit KMeans with the optimal number of clusters
                kmeans = KMeans(n_clusters=optimal_k,n_init='auto', max_iter=self.set_up_parameters_km[1])
                kmeans.fit(pcd_f) 
                # Visualize and save the elbow plot
                ch.show(outpath=os.path.join(self.file_path, 'optimal_cluster_calinski_harabasz.png'))
                # Close the figure
                plt.close()
            elif self.optimization_strategy[0]==4: #Perform the , 4 Davies-Bouldin index
                def get_kmeans_score(data, center):
                    kmeans = KMeans(n_clusters=center,n_init='auto')
                    model = kmeans.fit_predict(data)                
                    score = davies_bouldin_score(data, model)
                    return score
                scores = []
                for center in k_values:
                    scores.append(get_kmeans_score(pcd_f, center))
                # Pair each k_value with its corresponding score
                k_scores = list(zip(k_values, scores))
                
                # Find the k_value with the minimum score
                optimal_k, min_score = min(k_scores, key=lambda x: x[1])
                # Apply Yellowbrick style to the plot
                set_palette('flatui')
                
                plt.figure(figsize=(10, 6))
                plt.plot(k_values, scores, linestyle='--', marker='o', color='b')
                plt.xlabel('K')
                plt.ylabel('Davies Bouldin Score')
                plt.title('Davies Bouldin Score for KMeans clustering')
                # Add a vertical line at the minimum score
                plt.axvline(x=optimal_k, color='black', linestyle='--')
                # Save the plot to a file
                plt.savefig(os.path.join(self.file_path, 'optimal_cluster_davies_boulding.png'), format='png', dpi=300)
                # Close the figure
                plt.close()
                # Fit KMeans with the optimal number of clusters
                kmeans = KMeans(n_clusters=optimal_k,n_init='auto', max_iter=self.set_up_parameters_km[1])
                kmeans.fit(pcd_f) 
                # Visualize and save the elbow plot
        
            else:
                kmeans = KMeans(n_clusters=self.set_up_parameters_km[0], max_iter=self.set_up_parameters_km[1],n_init='auto')
                kmeans.fit(pcd_f)               
            labels =kmeans.labels_
            config_algo=kmeans
        elif algo=="Fuzzy-K-means":
            if self.optimization_strategy[0] != 0: # We want an optimization of clusters
                # Choosing a range of K values for KMeans
                if self.optimization_strategy[2]<=2: #It is neccesary to perform the methods with at least 2 cluster. At exception of the elbow
                    minimum_clusters=2
                elif self.optimization_strategy[2]<=1 and self.optimization_strategy[0]==1: # It is possible to perform the Elbow with one cluster
                    minimum_clusters=1
                else: 
                    minimum_clusters=self.optimization_strategy[2]
                if self.optimization_strategy[1]+1<=minimum_clusters:
                    maximum_clusters=minimum_clusters+1
                else:
                    maximum_clusters=self.optimization_strategy[1]+1
                k_values = range(minimum_clusters, maximum_clusters)
                model = KMeans(n_init='auto')
            if self.optimization_strategy[0]==1: #Perform the elbow method             
                elbow = KElbowVisualizer(model, k=k_values, timings=False)
                elbow.fit(pcd_f) 
                # Get the optimal number of clusters
                optimal_k = elbow.elbow_value_
                # Fit Fuzzy KMeans with the optimal number of clusters
                fcm = FCM(n_clusters=optimal_k, max_iter=self.set_up_parameters_fkm[1])
                fcm.fit(pcd_f)  # Pass the DataFrame values as input to the algorithm
                # Visualize and save the elbow plot
                elbow.show(outpath=os.path.join(self.file_path, 'optimal_cluster_elbow.png'))
                # Close the figure
                plt.close()
            elif self.optimization_strategy[0]==2: #Perform the sillouethe method    
                silhouette = KElbowVisualizer(model, k=k_values, timings=False, metric='silhouette')
                silhouette.fit(pcd_f)
                # Get the optimal number of clusters
                optimal_k = silhouette.elbow_value_
                # Fit Fuzzy KMeans with the optimal number of clusters
                fcm = FCM(n_clusters=optimal_k, max_iter=self.set_up_parameters_fkm[1])
                fcm.fit(pcd_f)  # Pass the DataFrame values as input to the algorithm
                # Visualize and save the elbow plot
                silhouette.show(outpath=os.path.join(self.file_path, 'optimal_cluster_sihouette.png'))
                # Close the figure
                plt.close()
            elif self.optimization_strategy[0]==3: #Perform the Calinski-Harabasz index
                ch = KElbowVisualizer(model, k=k_values,metric='calinski_harabasz', timings= False)
                ch.fit(pcd_f)        # Fit the data to the visualizer
                # Get the optimal number of clusters
                optimal_k = ch.elbow_value_
                # Fit Fuzzy KMeans with the optimal number of clusters
                fcm = FCM(n_clusters=optimal_k, max_iter=self.set_up_parameters_fkm[1])
                fcm.fit(pcd_f)  # Pass the DataFrame values as input to the algorithm
                # Visualize and save the elbow plot
                ch.show(outpath=os.path.join(self.file_path, 'optimal_cluster_calinski_harabasz.png'))
                # Close the figure
                plt.close()
            elif self.optimization_strategy[0]==4: #Perform the , 4 Davies-Bouldin index
                def get_kmeans_score(data, center):
                    kmeans = KMeans(n_clusters=center,n_init='auto', max_iter=self.set_up_parameters_km[1])
                    model = kmeans.fit_predict(data)                
                    score = davies_bouldin_score(data, model)
                    return score
                scores = []
                for center in k_values:
                    scores.append(get_kmeans_score(pcd_f, center))
                # Pair each k_value with its corresponding score
                k_scores = list(zip(k_values, scores))
                
                # Find the k_value with the minimum score
                optimal_k, min_score = min(k_scores, key=lambda x: x[1])
                # Fit Fuzzy KMeans with the optimal number of clusters
                fcm = FCM(n_clusters=optimal_k, max_iter=self.set_up_parameters_fkm[1])
                fcm.fit(pcd_f)  # Pass the DataFrame values as input to the algorithm
                # Apply Yellowbrick style to the plot
                set_palette('flatui')
                
                plt.figure(figsize=(10, 6))
                plt.plot(k_values, scores, linestyle='--', marker='o', color='b')
                plt.xlabel('K')
                plt.ylabel('Davies Bouldin Score')
                plt.title('Davies Bouldin Score for KMeans clustering')
                # Add a vertical line at the minimum score
                plt.axvline(x=optimal_k, color='black', linestyle='--')
                # Save the plot to a file
                plt.savefig(os.path.join(self.file_path, 'optimal_cluster_davies_boulding.png'), format='png', dpi=300)
                # Close the figure
                plt.close()
  
            else:
                fcm = FCM(n_clusters=self.set_up_parameters_fkm[0], max_iter=self.set_up_parameters_fkm[1])
                fcm.fit(pcd_f)  # Pass the DataFrame values as input to the algorithm
            # Retrieve cluster labels
            labels = fcm.u.argmax(axis=1)
            config_algo=fcm
        elif algo=="Hierarchical-clustering":
            # Some restriction due to the library version
            if self.set_up_parameters_hc[0]==0:
                temp_list = list(self.set_up_parameters_hc)
                temp_list[0] = None                
                self.set_up_parameters_hc = tuple(temp_list)
            else:
                temp_list = list(self.set_up_parameters_hc)
                temp_list[4] = None
                self.set_up_parameters_hc = tuple(temp_list)
            if self.set_up_parameters_hc[3]=="ward":
                temp_list = list(self.set_up_parameters_hc)
                temp_list[1] = "euclidean"
                self.set_up_parameters_hc = tuple(temp_list)
            if self.set_up_parameters_hc[2]=="none" or self.set_up_parameters_hc[3]=="ward":
                temp_list = list(self.set_up_parameters_hc)
                temp_list[2] = "euclidean"
                self.set_up_parameters_hc = tuple(temp_list)
            if self.set_up_parameters_hc[4]==0:
                temp_list = list(self.set_up_parameters_hc)
                temp_list[0] = None
                temp_list[6] = True
                self.set_up_parameters_hc = tuple(temp_list)
 
            if self.set_up_parameters_hc[5]=='true':
                temp_list = list(self.set_up_parameters_hc)
                temp_list[5] = True
                self.set_up_parameters_hc = tuple(temp_list)
            else:
                temp_list = list(self.set_up_parameters_hc)
                temp_list[5] = False
                self.set_up_parameters_hc = tuple(temp_list)
            # Perform clustering
            hc = AgglomerativeClustering(n_clusters=self.set_up_parameters_hc[0], metric=self.set_up_parameters_hc[2],linkage=self.set_up_parameters_hc[3],distance_threshold=self.set_up_parameters_hc[4],compute_distances=self.set_up_parameters_hc[5],compute_full_tree=self.set_up_parameters_hc[6])
            hc.fit(pcd_f)
            labels = hc.labels_
            config_algo=hc
        elif algo=="DBSCAN":
            dbscan = DBSCAN(eps=self.set_up_parameters_dbscan[0],min_samples=self.set_up_parameters_dbscan[1])
            dbscan.fit(pcd_f) 
            labels=dbscan.labels_
            config_algo=dbscan
        elif algo=="OPTICS":            
            optics = OPTICS(min_samples=self.set_up_parameters_optics[0],max_eps=self.set_up_parameters_optics[1],metric=self.set_up_parameters_optics[2],cluster_method=self.set_up_parameters_optics[3],xi=self.set_up_parameters_optics[4],min_cluster_size=self.set_up_parameters_optics[5])
            optics.fit(pcd_f) 
            labels=optics.labels_
            config_algo=optics
                         
        ## CREATE THE RESULTING POINT CLOUD 
        pc_results = pycc.ccPointCloud(pcd['X'], pcd['Y'], pcd['Z'])
        pc_results.setName("Results_from_clustering")
        idx = pc_results.addScalarField("Clusters",labels) 
        # STORE IN THE DATABASE OF CLOUDCOMPARE
        CC.addToDB(pc_results)
        CC.updateUI() 
        # SAVE THE CONFIGURATION FILE, THE FEATURES2INCLUDE FOR PREDICTION, THE SCATTER PLOTS OF EACH VARIABLE AND OTHER PLOTS SUCH AS DENDOGRAMS OR ELBOWGRAPHS AMONG OTHERS 
        # Join the list items with commas to create a comma-separated string
        comma_separated = ','.join(self.features2include)    
        # Write the comma-separated string to a text file
        with open(os.path.join(self.file_path, 'features.txt'), 'w') as file:
            file.write(comma_separated)  
        with open(os.path.join(self.file_path,'config.pkl'), 'wb') as file:
            pickle.dump(config_algo, file) 
        # Generate the scattere plots of each variable combination
        df = pd.DataFrame(pcd_f, columns=self.features2include)
        df['Cluster'] = labels
        # Create a directory for plots
        folder_name = os.path.join(self.file_path,'scatter_plots')
        os.makedirs(folder_name, exist_ok=True)
        # Generate all combinations of features
        features = df.columns[:-1]  # Exclude the cluster label
        combinations = list(itertools.combinations(features, 2))
        # Create and save scatter plots
        for combo in combinations:
            plt.figure()
            plt.scatter(df[combo[0]], df[combo[1]], c=df['Cluster'], cmap='viridis')
            plt.xlabel(combo[0])
            plt.ylabel(combo[1])
            plt.title(f"{combo[0]}-{combo[1]}")
            plt.colorbar(label='Cluster')
            plt.savefig(f"{folder_name}/{combo[0]}-{combo[1]}.png")
            plt.close()
        if algo=="Hierarchical-clustering":
            # Plot the dendrogram
            # Create a new figure
            def plot_dendrogram(model, **kwargs):
                # Create linkage matrix and then plot the dendrogram
            
                # create the counts of samples under each node
                counts = np.zeros(model.children_.shape[0])
                n_samples = len(model.labels_)
                for i, merge in enumerate(model.children_):
                    current_count = 0
                    for child_idx in merge:
                        if child_idx < n_samples:
                            current_count += 1  # leaf node
                        else:
                            current_count += counts[child_idx - n_samples]
                    counts[i] = current_count
            
                linkage_matrix = np.column_stack(
                    [model.children_, model.distances_, counts]
                ).astype(float)
            
                # Plot the corresponding dendrogram
                dendrogram(linkage_matrix, **kwargs)
            
            if self.set_up_parameters_hc[0]==None: # print the dendogram if is possible. Number of cluster is not defined and the algorithm computes the full tree
                plt.title("Hierarchical Clustering Dendrogram")
                # plot the top three levels of the dendrogram
                plot_dendrogram(hc, truncate_mode="level", p=3)
                plt.xlabel("Number of points in node (or index of point if no parenthesis).")
                plt.tight_layout()
                plt.savefig(os.path.join(self.file_path,'dendogram.png'))
            else: 
                message = "If you want to compute the dendogram of the dataset, you need to set the number of clusters to 0 or the linkeage distance to 0."                
                # Open the file in write mode
                with open(os.path.join(self.file_path,'dendogram_issue.txt'), "w") as file:
                    # Write the message to the file
                    file.write(message)
        print("The process has been finished")
        root.destroy()
    def run_algorithm_2 (self,prediction_pc_name):
        # Error to prevent the abscene of point cloud
        CC = pycc.GetInstance() 
        type_data, number = get_istance()
        if type_data=='point_cloud' or type_data=='folder':
            pass
        else:
            raise RuntimeError("Please select a folder that contains points clouds or a point cloud")        
        if number==0:
            raise RuntimeError("There are not entities in the folder")
        else:
            entities = CC.getSelectedEntities()[0]
            number = entities.getChildrenNumber()
            
        if type_data=='point_cloud':
            pc_prediction=entities
        else:   
            entities = CC.getSelectedEntities()[0]
            number = entities.getChildrenNumber()
            
        if type_data=='point_cloud':
            pc_prediction=entities
        else:
            for ii, item in enumerate(name_list):
                if item == prediction_pc_name:
                    pc_prediction = entities.getChild(ii)
                    break
        pcd_p=P2p_getdata(pc_prediction,False,False,True)
        pcd_p_f=pcd_p[self.features_prediction].values
        # Use the loaded model for prediction if possible
        try:
            predicted_clusters=self.configuration_prediction.predict(pcd_p_f)
        except AttributeError as e:
            raise RuntimeError(f"Error: {type(self.configuration_prediction).__name__} has no 'predict' method.")        
        ## CREATE THE RESULTING POINT CLOUD 
        pc_results_prediction = pycc.ccPointCloud(pcd_p['X'], pcd_p['Y'], pcd_p['Z'])
        pc_results_prediction.setName("Results_from_clustering")
        idx = pc_results_prediction.addScalarField("Clusters",predicted_clusters) 
        # STORE IN THE DATABASE OF CLOUDCOMPARE
        CC.addToDB(pc_results_prediction)
        CC.updateUI() 
        print("The process has been finished")
        root.destroy()
#%% RUN THE GUI        
try:
    # START THE MAIN WINDOW        
    root = tk.Tk()
    app = GUI()
    app.main_frame(root)
    root.mainloop()    
except Exception as e:
    print("An error occurred during the computation of the algorithm:", e)
    # Optionally, print detailed traceback
    traceback.print_exc()
    root.destroy()


