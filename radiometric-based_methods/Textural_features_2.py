# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 13:42:37 2024

@author: LuisJa
"""
import tkinter as tk
import sys
import traceback
import sys
import os
import numpy as np
import open3d as o3d

#CloudCompare Python Plugin
import cccorelib
import pycc

from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde

#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS
script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'

sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata, get_point_clouds_name,check_input
from main_gui import definition_of_labels_type_1,definition_of_entries_type_1,show_features_window,definition_of_buttons_type_1,definition_of_combobox_type_1,definition_run_cancel_buttons_type_1,definition_of_checkbutton_type_1
#%% INITIAL OPERATIONS
name_list=get_point_clouds_name()
#%% GUI
class GUI:
    def __init__(self): # Initial parameters. It is in self because we can update during the interaction with the user
        self.features2include = []
    def main_frame (self, root):    # Main frame of the GUI  
        
        # FUNCTIONS 
        
        # Destroy the window
        def destroy (self): 
            root.destroy ()        
        
        # GENERAL CONFIGURATION OF THE GUI
        
        # Configuration of the window        
        root.title ("Textural related features")
        root.resizable (False, False)     
        root.attributes ('-toolwindow',-1) # Remove minimize and maximize button   
        
        # Labels
        label_texts = [
            "Choose point cloud:",
            "Search radius:",
            "Feature to be used:",
            "Mean value:",
            "Standard deviation:",
            "Range:",
            "Energy:",
            "Entropy:",
            "Kurtosis:",
            "Skewness:",
            "Skewness with binary values",
        ]
        row_positions = [0,1,2,3,4,5,6,7,8,9,10]        
        definition_of_labels_type_1 ("root",label_texts,row_positions,root,0) 
        
        # Entries
        entry_insert = [10]
        row_positions = [1]        
        root_entries = definition_of_entries_type_1 ("root",entry_insert,row_positions,root,1) 
        
        # Checkbuttons
        row_positions = [3,4,5,6,7,8,9,10]
        initial_states= [True,True,False,False,False,False,False,False]
        root_checkbuttons, root_checkbutton_vars=definition_of_checkbutton_type_1("root", row_positions, root, initial_states,1)
        
        # Combobox
        combobox_insert = [name_list]
        row_positions = [0]
        selected_element = []
        root_comboboxes =definition_of_combobox_type_1 ("root",combobox_insert,row_positions, selected_element,root,1) 
        
        # Buttons
        row_buttons=[2]  
        button_names=["..."]
        
        _=definition_of_buttons_type_1("root",button_names,row_buttons,[lambda: show_features_window(self,name_list,root_comboboxes[0].get(),"Classification",True)],root,1)

        _=definition_run_cancel_buttons_type_1("root",
                                               [lambda:run_algorithm_1(self,root_comboboxes[0].get(),float(root_entries[1].get()),root_checkbutton_vars[3].get(),root_checkbutton_vars[4].get(),root_checkbutton_vars[5].get(),root_checkbutton_vars[6].get(),root_checkbutton_vars[7].get(),root_checkbutton_vars[8].get(),root_checkbutton_vars[9].get(),root_checkbutton_vars[10].get()),
                                                lambda:destroy(self)],
                                               11,
                                               root,
                                               1)         

        # RUN THE PROCESS
        def run_algorithm_1 (self,pc,radius,selected_mean,selected_std,selected_range,selected_energy,selected_entropy,selected_kurtosis,selected_skewness,selected_skewness_binary):
            
            # Start the progress_Bar
            progress = pycc.ccProgressDialog()
            progress.start()
            
            progress.setMethodTitle ("Reading the point cloud (please wait)")
            
            # Check if the selection is a point cloud
            pc=check_input(name_list,pc)
            
            # Get the desired numpy array (X,Y,Z and selected scalar)
            pcd=P2p_getdata(pc)
            array_f=pcd[self.features2include].values
            
            # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
            pcdxyz = o3d.geometry.PointCloud()
            pcdxyz.points = o3d.utility.Vector3dVector(pcd.values[:,0:3])
            pcdxyz_tree = o3d.geometry.KDTreeFlann(pcdxyz)
            
            # Pre-compute lengths and prepare data structures
            num_points = len(pcdxyz.points)
            if selected_mean:
                mean_values= np.zeros(num_points)
            if selected_std:
                std_values= np.zeros(num_points)
            if selected_range:
                range_values= np.zeros(num_points)            
            if selected_energy:
                energy_values= np.zeros(num_points)                  
            if selected_entropy:
                entropy_values= np.zeros(num_points)             
            if selected_kurtosis:
                kurtosis_values= np.zeros(num_points)                 
            if selected_skewness:
                skewness_values= np.zeros(num_points) 
                
            progress.setMethodTitle ("Calculating the texture-related features (please wait)")
            
            # Defining the 1% for updating the progress bar
            one_percent = max(1, num_points // 100)
            
            
            # Iterate over each point in the point cloud
            for i in range(num_points):
                if i % one_percent == 0:
                    progress.update ((i / num_points) * 100)
                
                [k, idx, _] = pcdxyz_tree.search_radius_vector_3d(pcdxyz.points[i], radius)
                filtered_array = array_f[idx[1:]]+array_f[i]
                
                if selected_mean:
                    # Calculate the mean value and standard deviation of the data
                    mean_values [i]=np.mean(filtered_array)
                if selected_std:
                    std_values [i]=np.std(filtered_array)
                
                if selected_range:
                    # Calculate the range of the data
                    range_values [i]= np.max(filtered_array) - np.min(filtered_array)
                
                if selected_energy:
                    # Calculate the energy
                    energy_values [i]=np.sum(np.max(filtered_array)**2)
                
                if selected_entropy:
                    # Calculate the entropy
                    kde = gaussian_kde(filtered_array.T) #  Calculate de Gaussian KDE
                    pdf = kde(filtered_array.T)
                    entropy_values [i]=-np.sum(pdf * np.log2(pdf)) # Compute entropy
                if selected_kurtosis:
                    # Calculate the kurtosis and skewness
                    n = len(filtered_array)  
                    if n<4:
                        kurt=None
                    else:
                        std_dev = np.std(filtered_array, ddof=1)  # ddof=1 for sample standard deviation
                        summed = np.sum(((filtered_array - np.mean(filtered_array)) / std_dev) ** 4)                           
                        kurt = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3)) * summed
                        kurt -= 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
                    kurtosis_values[i]=kurt
                if selected_skewness or selected_skewness_binary:
                    std_dev= np.std(filtered_array, ddof=1)  # ddof=1 for sample standard deviation
                    summed=np.sum(((filtered_array - np.mean(filtered_array)) / std_dev) ** 3)          
                    skewness_values[i]=(n / ((n - 1) * (n - 2))) * summed
                    
            if selected_skewness_binary:          
                skewness_values_binary = np.where(skewness_values > 0, 1, 0) # skeness binary (1 if is possitive, 0 if is negative)  
                
            # Introducing the new values as scalar fields    
            pc_results_prediction = pycc.ccPointCloud(pcd['X'], pcd['Y'], pcd['Z'])
            pc_results_prediction.setName("Results_from_textural_features")

            if selected_mean:
                pc_results_prediction.addScalarField(self.features2include[0]+"_mean (Texture) ("+str(radius)+")", mean_values)
            if selected_std:
                pc_results_prediction.addScalarField(self.features2include[0]+"_standard deviation (Texture) ("+str(radius)+")", std_values)
            if selected_range:
                pc_results_prediction.addScalarField(self.features2include[0]+"_range (Texture) ("+str(radius)+")", range_values)                
            if selected_energy:
                pc_results_prediction.addScalarField(self.features2include[0]+"_energy (Texture) ("+str(radius)+")", energy_values)                   
            if selected_entropy:
                pc_results_prediction.addScalarField(self.features2include[0]+"_entropy (Texture) ("+str(radius)+")", entropy_values)               
            if selected_kurtosis:
                pc_results_prediction.addScalarField(self.features2include[0]+"_kurtosis (Texture) ("+str(radius)+")", kurtosis_values)                 
            if selected_skewness:
                pc_results_prediction.addScalarField(self.features2include[0]+"_skewness (Texture) ("+str(radius)+")", skewness_values) 
            if selected_skewness_binary:
                pc_results_prediction.addScalarField(self.features2include[0]+"_skewness binary (Texture) ("+str(radius)+")", skewness_values_binary)                 
            
            # STORE IN THE DATABASE OF CLOUDCOMPARE
            CC = pycc.GetInstance()
            CC.addToDB(pc_results_prediction)
            CC.updateUI() 
            root.destroy()
            
            print("The process has been finished")       
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