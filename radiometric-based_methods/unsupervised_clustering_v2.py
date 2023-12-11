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
import pickle

import sys
import os

import cccorelib
import pycc
#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS

script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'

sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance,get_point_clouds_name

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
        #DBSCAN
        self.set_up_parameters_dbscan= (5,200)
        #OPTICS
        self.set_up_parameters_optics=(5,200,"minkowski","xi",0.05,10)
        self.file_path=os.path.join(current_directory, output_directory)
        
    def main_frame (self, root):
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
        
        # Labels            
        label_pc_training = tk.Label(tab1, text="Choose point cloud for training:")
        label_pc_training.grid(row=0, column=0, sticky=tk.W)
        label_algo = tk.Label(tab1, text="Select a clustereing algorithm:")
        label_algo.grid(row=1, column=0, sticky=tk.W) 
        label_fea = tk.Label(tab1, text="Select the features to include:")
        label_fea.grid(row=2, column=0, sticky=tk.W) 
        label_out= ttk.Label(tab1, text="Choose output directory")
        label_out.grid(row=3, column=0, sticky=tk.W)
        
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
        self.entry_widget = ttk.Entry(tab1, width=30)
        self.entry_widget.grid(row=3, column=1, sticky="e", pady=2)
        self.entry_widget.insert(0, self.file_path)
        
        # Button
        t1_setup_button= ttk.Button (tab1, text="Set-up", command=lambda: self.show_set_up_window(combot2.get()), width=10)
        t1_setup_button.grid (row=1,column=2,sticky="e",padx=100)
        t1_features= ttk.Button (tab1, text="...", command=lambda: self.show_features_window(combot1.get()), width=10)
        t1_features.grid (row=2,column=2,sticky="e",padx=100)
        button_widget = ttk.Button(tab1, text="...", command=self.save_file_dialog, width=10)
        button_widget.grid(row=3, column=2, sticky="e", padx=100)
        t1_run_button= ttk.Button (tab1, text="OK", command=lambda:self.run_algorithm_1(combot2.get(),combot1.get()), width=10)
        t1_run_button.grid (row=4,column=1,sticky="e",padx=100)
        t1_cancel_button= ttk.Button (tab1, text="Cancel", command=self.destroy,width=10)
        t1_cancel_button.grid (row=4,column=1,sticky="e")
        
    def save_setup_parameters (self,algo,*params):
        if algo=="K-means":
            self.set_up_parameters_km=params
        elif algo=="Fuzzy-K-means":
            self.set_up_parameters_fkm=params
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
            elif algo=="DBSCAN":
                self.save_setup_parameters(algo, float(entry_param1_dbscan.get()), int(entry_param2_dbscan.get()))
            elif algo=="OPTICS":
                self.save_setup_parameters(algo, int(entry_param1_optics.get()), float(entry_param2_optics.get()), str(entry_param3_optics.get()), str(entry_param4_optics.get()), float(entry_param5_optics.get()),int(entry_param6_optics.get()))                
            
            set_up_window.destroy()  # Close the window after saving parameters
            
        set_up_window = tk.Toplevel(root)
        set_up_window.title("Set Up the algorithm")
        set_up_window.resizable (False, False)
        # Remove minimize and maximize button 
        set_up_window.attributes ('-toolwindow',-1)
        
        if algo=="K-means":
            # Labels            
            label_param1_km = tk.Label(set_up_window, text="Number of clusters:")
            label_param1_km.grid(row=0, column=0, sticky=tk.W)
            label_param2_km = tk.Label(set_up_window, text="Number of iterations:")
            label_param2_km.grid(row=1, column=0, sticky=tk.W) 
            
            # Entries
            entry_param1_km= tk.Entry(set_up_window)
            entry_param1_km.insert(0,self.set_up_parameters_km[0])
            entry_param1_km.grid(row=0, column=1) 
            
            entry_param2_km= tk.Entry(set_up_window)
            entry_param2_km.insert(0,self.set_up_parameters_km[1])
            entry_param2_km.grid(row=1, column=1) 
            # Buttons
            button_ok = tk.Button(set_up_window, text="OK", command=lambda: on_ok_button_click(algo))
            button_ok.grid(row=2, column=1)
        elif algo=="Fuzzy-K-means":
            # Labels            
            label_param1_fkm = tk.Label(set_up_window, text="Number of clusters:")
            label_param1_fkm.grid(row=0, column=0, sticky=tk.W)
            label_param2_fkm = tk.Label(set_up_window, text="Number of iterations:")
            label_param2_fkm.grid(row=1, column=0, sticky=tk.W) 
            
            # Entries
            entry_param1_fkm= tk.Entry(set_up_window)
            entry_param1_fkm.insert(0,self.set_up_parameters_fkm[0])
            entry_param1_fkm.grid(row=0, column=1) 
            
            entry_param2_fkm= tk.Entry(set_up_window)
            entry_param2_fkm.insert(0,self.set_up_parameters_fkm[1])
            entry_param2_fkm.grid(row=1, column=1) 
            # Buttons
            button_ok = tk.Button(set_up_window, text="OK", command=lambda: on_ok_button_click(algo))
            button_ok.grid(row=2, column=1)        
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
   
        index = -1
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
            
        # Get the point cloud as a dataframe with the features2include
        for ii, item in enumerate(name_list):
            if item == training_pc_name:
                pc_training = entities.getChild(ii)
                break
        pcd=P2p_getdata(pc_training,False,False,True)
        pcd_f=pcd[self.features2include].values 

        # Error control to prevent not algorithm for the training
        if algo=="Not selected":
            raise RuntimeError ("Please select and algorithm for the training")
        elif algo=="K-means":
            kmeans = KMeans(n_clusters=self.set_up_parameters_km[0], max_iter=self.set_up_parameters_km[1],n_init='auto')
            kmeans.fit(pcd_f)               
            labels =kmeans.labels_
            config_algo=kmeans
        elif algo=="Fuzzy-K-means":
            fcm = FCM(n_clusters=self.set_up_parameters_fkm[0], max_iter=self.set_up_parameters_fkm[1])
            fcm.fit(pcd_f)  # Pass the DataFrame values as input to the algorithm
            # Retrieve cluster labels
            labels = fcm.u.argmax(axis=1)
            config_algo=fcm
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
        # SAVE THE CONFIGURATION FILE AS WELL AS THE FEATURES2INCLUDE FOR PREDICTION  
        # Join the list items with commas to create a comma-separated string
        comma_separated = ','.join(self.features2include)    
        # Write the comma-separated string to a text file
        with open(os.path.join(self.file_path, 'features.txt'), 'w') as file:
            file.write(comma_separated)  
        with open(os.path.join(self.file_path,'config.pkl'), 'wb') as file:
            pickle.dump(output_directory, file) 
        print("The process has been finished")  
# START THE MAIN WINDOW        
root = tk.Tk()
app = GUI()
app.main_frame(root)
root.mainloop()
