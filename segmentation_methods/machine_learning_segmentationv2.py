# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:58:50 2024

@author: LuisJa
"""
#%% LIBRARIES
import os
import subprocess
import sys
import yaml

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
import traceback
import pandas as pd

#CloudCompare Python Plugin
import cccorelib
import pycc

#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS
script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'

sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance,get_point_clouds_name, check_input, write_yaml_file
from main_gui import show_features_window, definition_of_labels

#%% ADDING PATHS FROM THE CONFIGS FILES
current_directory= os.path.dirname(os.path.abspath(__file__))

config_file=os.path.join(current_directory,r'..\configs\executables.yml')

# Read the configuration from the YAML file for the set-up
with open(config_file, 'r') as yaml_file:
    config_data = yaml.safe_load(yaml_file)
path_optimal_flow= os.path.join(current_directory,config_data['OPTIMAL_FLOW'])
path_random_forest= os.path.join(current_directory,config_data['RANDOM_FOREST'])
path_prediction= os.path.join(current_directory,config_data['PREDICTION'])
path_aml= os.path.join(current_directory,config_data['TPOT'])

#%% INITIAL OPERATIONS
name_list=get_point_clouds_name()

#%% GUI
class GUI:
    def __init__(self): # Initial parameters. It is in self because we can update during the interaction with the user
       
        # Features2include
        self.features2include=[] 
        self.values_list=[]
        self.features=[]
        
        
        # Optimal flow
        self.set_up_parameters_of= {
            "selectors": ['kbest_f','rfe_lr','rfe_tree','rfe_rf','rfecv_tree','rfecv_rf','rfe_svm','rfecv_svm'],
            "percentage": 25,
            "cv": 5,
            "point_cloud":"input_point_cloud.txt"
        }
        
        # Random forest
        self.set_up_parameters_rf= {
            "estimators": 200,
            "criterion": "gini",
            "trees": 100,
            "internal_node": 2,
            "leaf_node": 1,
            "weights": 0,
            "features": "sqrt",
            "bootstrap": True,
            "scoring": 'balanced_accuracy',
            "njobs": -1
            }
            
        # Auto-ml
        self.set_up_parameters_aml= {
            "generations": 5,
            "size": 20,
            "mutation": 0.9,
            "crossover": 0.1,
            "cv": 2,
            "max_time_total": 60,
            "max_time_ev": 10,
            "no_improvement":2,
            "scoring": "balanced_accuracy"
            }     

        # Directory to save the files (output)
        self.output_directory=os.getcwd() # The working directory
        
    def main_frame (self, root):    # Main frame of the GUI  
        
        # FUNCTIONS 
        
        # Function to create tooltip
        def create_tooltip(widget, text):
            widget.bind("<Enter>", lambda event: show_tooltip(text))
            widget.bind("<Leave>", hide_tooltip)

        def show_tooltip(text):
            tooltip.config(text=text)
            tooltip.place(relx=0.5, rely=0.5, anchor="center", bordermode="outside")
            
        def hide_tooltip(event):
            tooltip.place_forget()

        tooltip = tk.Label(root, text="", relief="solid", borderwidth=1)
        tooltip.place_forget() 
        
        # Function to get the selectors of the set_up_paramenters
        def on_ok_button_click():
            selected_items = listbox.curselection()
            selected_values = [listbox.get(index) for index in selected_items]
            messagebox.showinfo("Selectors", f"Selected options: {', '.join(selected_values)}")
        
        # Function to save and get the output_directory
        def save_file_dialog(tab):
            directory = filedialog.askdirectory()
            if directory:
                self.output_directory = directory                
                if tab ==1:  # Update the entry widget of the tab                   
                    t1_entry_widget.delete(0, tk.END)
                    t1_entry_widget.insert(0, self.output_directory)    
                elif tab==2:
                    t2_entry_widget.delete(0, tk.END)
                    t2_entry_widget.insert(0, self.output_directory)  
                elif tab==3:
                    t3_entry_widget.delete(0, tk.END)
                    t3_entry_widget.insert(0, self.output_directory)      
                
        # Destroy the window
        def destroy (self): 
            root.destroy ()
        
        # Safe the set_up_paramenters in accordance with the GUI
        def save_setup_parameters (self,algo,*params): 
            if algo=="Random Forest":
                self.set_up_parameters_rf["estimators"]=params[0]
                self.set_up_parameters_rf["criterion"]=params[1]
                self.set_up_parameters_rf["trees"]=params[2]
                self.set_up_parameters_rf["internal_node"]=params[3]
                self.set_up_parameters_rf["leaf_node"]=params[4]
                self.set_up_parameters_rf["weights"]=params[5]
                self.set_up_parameters_rf["features"]=params[6]
                self.set_up_parameters_rf["bootstrap"]=params[7]
                self.set_up_parameters_rf["scoring"]=params[8]

                    
            elif algo=="Auto Machine Learning":
                self.set_up_parameters_aml["generations"]=params[0]
                self.set_up_parameters_aml["size"]=params[1]
                self.set_up_parameters_aml["mutation"]=params[2]
                self.set_up_parameters_aml["crossover"]=params[3]
                self.set_up_parameters_aml["cv"]=params[4]
                self.set_up_parameters_aml["max_time_total"]=params[5]
                self.set_up_parameters_aml["max_time_ev"]=params[6]
                self.set_up_parameters_aml["no_improvement"]=params[7]
                self.set_up_parameters_aml["scoring"]=params[8]
                
        # Load the configuration files for prediction
        def load_configuration_dialog():
           global load_configuration   
           file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
           if file_path:
               load_configuration = file_path     
                    
        def load_features_dialog():
            global load_features 
            file_path = filedialog.askopenfilename(filetypes=[("Feature file", "*.txt")])
            if file_path:
                load_features = file_path      
                
        # Window for the configuration of the machine learning algorithms
        def show_set_up_window (self,algo): 
            
                    
            def on_ok_button_click(algo):
                if algo=="Random Forest":
                    save_setup_parameters(self,algo, int(entry_param1_rf.get()), str(combo_param2_rf.get()), int(entry_param3_rf.get()), int(entry_param4_rf.get()), int(entry_param5_rf.get()), int(entry_param6_rf.get()), str(combo_param7_rf.get()), bool(combo_param8_rf.get()), str(combo_param9_rf.get()))
                elif algo=="Auto Machine Learning":                    
                    save_setup_parameters(self,algo, int(entry_param1_aml.get()), int(entry_param2_aml.get()), float(entry_param3_aml.get()), float(entry_param4_aml.get()), int(entry_param5_aml.get()), int(entry_param6_aml.get()), int(entry_param7_aml.get()), int(entry_param8_aml.get()), str(combo_param9_aml.get()))
               
                set_up_window.destroy()  # Close the window after saving parameters
                
            # Setup window
            set_up_window = tk.Toplevel(root)
            set_up_window.title("Set Up the algorithm")
            set_up_window.resizable (False, False)
            # Remove minimize and maximize button 
            set_up_window.attributes ('-toolwindow',-1)
               
            if algo=="Random Forest":
                
                #Labels        
                label_texts = [
                    "Number of estimators:",
                    "Criterion:",
                    "Maximum depth of trees:",
                    "Minimum number of samples required to Split the internal node:",
                    "Minimum number of samples required to be a leaf node:",
                    "Minimum weight fraction of the total sum of weights:",
                    "Maximum number of features:",
                    "Bootstrap:",
                    "Scoring:"
                ]
                row_positions = [0,1,2,3,4,5,6,7,8]        
                definition_of_labels ("root",label_texts,row_positions,set_up_window,0,2,"w")

                
                # Entries                
        
                entry_param1_rf= tk.Entry(set_up_window, width=10)
                entry_param1_rf.insert(0,self.set_up_parameters_rf["estimators"])
                entry_param1_rf.grid(row=0, column=1, sticky="e")
                
                criterion = ["gini","entropy","log_loss"]
                combo_param2_rf=ttk.Combobox (set_up_window, values=criterion, state="readonly")
                combo_param2_rf.set(self.set_up_parameters_rf["criterion"])
                combo_param2_rf.grid(column=1, row=1, sticky="e", pady=2)
                
                entry_param3_rf= ttk.Entry(set_up_window, width=10)
                entry_param3_rf.insert(0,self.set_up_parameters_rf["trees"])
                entry_param3_rf.grid(row=2, column=1, sticky="e")
                
                entry_param4_rf= ttk.Entry(set_up_window, width=10)
                entry_param4_rf.insert(0,self.set_up_parameters_rf["internal_node"])
                entry_param4_rf.grid(row=3, column=1, sticky="e")
                
                entry_param5_rf= ttk.Entry(set_up_window, width=10)
                entry_param5_rf.insert(0,self.set_up_parameters_rf["leaf_node"])
                entry_param5_rf.grid(row=4, column=1, sticky="e")
                
                entry_param6_rf= ttk.Entry(set_up_window, width=10)
                entry_param6_rf.insert(0,self.set_up_parameters_rf["weights"])
                entry_param6_rf.grid(row=5, column=1, sticky="e")
                
                max_n_features = ['sqrt','log2']
                combo_param7_rf=ttk.Combobox (set_up_window, values=max_n_features, state="readonly")
                combo_param7_rf.set(self.set_up_parameters_rf["features"])
                combo_param7_rf.grid(column=1, row=6, sticky="e", pady=2)
                
                bootstrap= [True, False]
                combo_param8_rf=ttk.Combobox (set_up_window, values=bootstrap, state="readonly")
                combo_param8_rf.set(self.set_up_parameters_rf["bootstrap"])
                combo_param8_rf.grid(column=1, row=7, sticky="e", pady=2)
                
                scoring=["balanced_accuracy","accuracy","f1","f1_weighted","precision","precision_weighted"]
                combo_param9_rf=ttk.Combobox (set_up_window, values=scoring, state="readonly")
                combo_param9_rf.set(self.set_up_parameters_rf["scoring"])
                combo_param9_rf.grid(column=1, row=8, sticky="e", pady=2)
                
                ok_button = ttk.Button(set_up_window, text="OK", command=lambda: on_ok_button_click(algo), width=10)
                ok_button.grid(row=9, column=1, sticky="w", padx=100)

                       
            elif algo=="Auto Machine Learning":
                
                # Labels
                label_texts = [
                    "Generations:",
                    "Population size:",
                    "Mutation rate:",
                    "Crossover rate:",
                    "Number of flods for k-fold cross-validation:",
                    "Maximum time in mins for the evaluation:",
                    "Maximum time in mins for each evaluation:",
                    "Number of generations without improvement:",
                    "Scoring:"
                ]
                row_positions = [0,1,2,3,4,5,6,7,8]        
                definition_of_labels ("root",label_texts,row_positions,set_up_window,0,2,"w")                     
                
                # Entries
                entry_param1_aml= ttk.Entry(set_up_window, width=10)
                entry_param1_aml.insert(0,self.set_up_parameters_aml["generations"])
                entry_param1_aml.grid(row=0, column=1, sticky="e")
                
                entry_param2_aml= ttk.Entry(set_up_window, width=10)
                entry_param2_aml.insert(0,self.set_up_parameters_aml["size"])
                entry_param2_aml.grid(row=1, column=1, sticky="e")
                
                entry_param3_aml= ttk.Entry(set_up_window, width=10)
                entry_param3_aml.insert(0,self.set_up_parameters_aml["mutation"])
                entry_param3_aml.grid(row=2, column=1, sticky="e")
                
                entry_param4_aml= ttk.Entry(set_up_window, width=10)
                entry_param4_aml.insert(0,self.set_up_parameters_aml["crossover"])
                entry_param4_aml.grid(row=3, column=1, sticky="e")
                
                entry_param5_aml= ttk.Entry(set_up_window, width=10)
                entry_param5_aml.insert(0,self.set_up_parameters_aml["cv"])
                entry_param5_aml.grid(row=4, column=1, sticky="e")
                
                entry_param6_aml= ttk.Entry(set_up_window, width=10)
                entry_param6_aml.insert(0,self.set_up_parameters_aml["max_time_total"])
                entry_param6_aml.grid(row=5, column=1, sticky="e")
                
                entry_param7_aml= ttk.Entry(set_up_window, width=10)
                entry_param7_aml.insert(0,self.set_up_parameters_aml["max_time_ev"])
                entry_param7_aml.grid(row=6, column=1, sticky="e")
                
                entry_param8_aml= ttk.Entry(set_up_window, width=10)
                entry_param8_aml.insert(0,self.set_up_parameters_aml["no_improvement"])
                entry_param8_aml.grid(row=7, column=1, sticky="e")
                
                scoring=["balanced_accuracy","accuracy","f1","f1_weighted","precision","precision_weighted"]
                combo_param9_aml=ttk.Combobox (set_up_window, values=scoring, state="readonly")
                combo_param9_aml.set(self.set_up_parameters_aml["scoring"])
                combo_param9_aml.grid(column=1, row=8, sticky="e", pady=2)
                
                ok_button = ttk.Button(set_up_window, text="OK", command=lambda: on_ok_button_click(algo), width=10)
                ok_button.grid(row=9, column=1, sticky="w", padx=100) 
                
            
        # GENERAL CONFIGURATION OF THE GUI
        
        # Configuration of the window        
        root.title ("Machine Learning segmentation")
        root.resizable (False, False)     
        root.attributes ('-toolwindow',-1) # Remove minimize and maximize button 
        
        # Configuration of the tabs
        tab_control = ttk.Notebook(root)
        tab_control.pack(expand=1, fill="both")   
        
        tab1 = ttk.Frame(tab_control) # Create 3 tabs
        tab1.pack()
        tab_control.add(tab1, text='Feature selection')
        tab2 = ttk.Frame(tab_control) # Create 3 tabs
        tab2.pack() 
        tab_control.add(tab2, text='Classification')
        tab3 = ttk.Frame(tab_control) # Create 3 tabs
        tab3.pack()
        tab_control.add(tab3, text='Prediction')
        tab_control.pack(expand=1, fill="both")
       
        
        # TAB1 = FEATURE SELECTION      
        
        # Some lines to start the tab      
        listbox = tk.Listbox(tab1, selectmode=tk.MULTIPLE, height=len(self.set_up_parameters_of["selectors"]))
        for value in self.set_up_parameters_of["selectors"]:
            listbox.insert(tk.END, value)
            
        # Select first six elements
        for i in range(6):
            listbox.selection_set(i)
        listbox.grid(row=1, column=1, sticky="e", padx=10, pady=10)
        
        # Get the selected parameters from the Listbox
        selected_params = [self.set_up_parameters_of["selectors"][i] for i in listbox.curselection()]
        
        # Labels
        label_texts = [
            "Choose point cloud:",
            "Selectors:",
            "Features to include:",
            "Number of features to consider:",
            "Folds for cross-validation:",
            "Choose output directory:",
        ]
        row_positions = [0,1,2,3,4,5]        
        definition_of_labels ("t1",label_texts,row_positions,tab1,0,2,"w") 
            
        # Combobox
        t1_combo_point_cloud=ttk.Combobox (tab1,values=name_list)
        t1_combo_point_cloud.grid(column=1, row=0, sticky="e", pady=2)
        t1_combo_point_cloud.set("Select the point cloud used for feature selection:")

        # Entry
        t1_entry_percentage = ttk.Entry(tab1, width=10)
        t1_entry_percentage.insert(0,self.set_up_parameters_of["percentage"])
        t1_entry_percentage.grid(row=3, column=1, sticky="e", pady=2)
        
        t1_entry_cv = ttk.Entry(tab1, width=10)
        t1_entry_cv.insert(0,self.set_up_parameters_of["cv"])
        t1_entry_cv.grid(row=4, column=1, sticky="e", pady=2)
        
        t1_entry_widget = ttk.Entry(tab1, width=30)
        t1_entry_widget.grid(row=5, column=1, sticky="e", pady=2)
        t1_entry_widget.insert(0, self.output_directory)

        # Button
        t1_ok_button_selectors = tk.Button(tab1, text="OK", command=on_ok_button_click,width=10)
        t1_ok_button_selectors.grid(row=1, column=2, pady=10)
        t1_output = ttk.Button(tab1, text="...", command=lambda:save_file_dialog(1), width=10)
        t1_output.grid(row=5, column=2, sticky="e", padx=100)
        t1_features= ttk.Button (tab1, text="...", command=lambda: show_features_window(self,name_list,t1_combo_point_cloud.get()), width=10)
        t1_features.grid (row=2,column=2,sticky="e",padx=100)
        
        t1_run_button= ttk.Button (tab1, text="Run", command=lambda:run_algorithm_1(self,name_list,t1_combo_point_cloud.get(),listbox.curselection(),int(t1_entry_percentage.get()),int(t1_entry_cv.get())), width=10)
        t1_run_button.grid (row=6,column=1,sticky="e",padx=100)
        t1_cancel_button= ttk.Button (tab1, text="Cancel", command=lambda:destroy,width=10)
        t1_cancel_button.grid (row=6,column=1,sticky="e")
        
        # TAB2 = CLASSIFICATION

        # Labels
        label_texts = [
            "Choose point cloud for training:",
            "Choose point cloud for testing:",
            "Select machine learning algorithm:",
            "Select the features to include:",
            "Choose output directory:"
        ]
        row_positions = [0,1,2,3,4]        
        definition_of_labels ("t2",label_texts,row_positions,tab2,0,2,"w")     

        # Combobox
        t2_combo_point_cloud_training=ttk.Combobox (tab2,values=name_list)
        t2_combo_point_cloud_training.grid(column=1, row=0, sticky="e", pady=2)
        t2_combo_point_cloud_training.set("Select the point cloud used for training:")
        
        t2_combo_point_cloud_testing=ttk.Combobox (tab2,values=name_list)
        t2_combo_point_cloud_testing.grid(column=1, row=1, sticky="e", pady=2)
        t2_combo_point_cloud_testing.set("Select the point cloud used for testing:")
        
        algorithms=["Random Forest", "Logistic Regression", "Auto Machine Learning"]
        t2_combo_algo=ttk.Combobox (tab2,values=algorithms, state="readonly")
        t2_combo_algo.grid(column=1, row=2, sticky="e", pady=2)
        t2_combo_algo.set("Not selected")
        
        # Entry
        t2_entry_widget = ttk.Entry(tab2, width=30)
        t2_entry_widget.grid(row=4, column=1, sticky="e", pady=2)
        t2_entry_widget.insert(0, self.output_directory)
        
        # Button
        t2_setup_button= ttk.Button (tab2, text="Set-up", command=lambda: show_set_up_window(self,t2_combo_algo.get()), width=10)
        t2_setup_button.grid (row=2,column=2,sticky="e",padx=100)
        t2_features_button= ttk.Button (tab2, text="...", command=lambda: show_features_window(self,name_list,t2_combo_point_cloud_training.get()), width=10)
        t2_features_button.grid (row=3,column=2,sticky="e",padx=100)
        t2_output = ttk.Button(tab2, text="...", command=lambda:save_file_dialog(2), width=10)
        t2_output.grid(row=4, column=2, sticky="e", padx=100)
    
        t2_run_button= ttk.Button (tab2, text="Run", command=lambda:run_algorithm_2(self,t2_combo_algo.get(),t2_combo_point_cloud_training.get(),t2_combo_point_cloud_testing.get()), width=10)
        t2_run_button.grid (row=5,column=1,sticky="e",padx=100)
        t2_cancel_button= ttk.Button (tab2, text="Cancel", command=lambda:destroy,width=10)
        t2_cancel_button.grid (row=5,column=1,sticky="e")
        
        # TAB3 = PREDICTION

        # Labels
        label_texts = [
            "Choose point cloud for prediction:",
            "Load feature file:",
            "Load pkl file:",
            "Select the features to include:",
            "Choose output directory:"
        ]
        row_positions = [0,1,2,3]        
        definition_of_labels ("t3",label_texts,row_positions,tab3,0,2,"w")  
        
        # Combobox
        t3_combo_1=ttk.Combobox (tab3,values=name_list)
        t3_combo_1.grid(column=1, row=0, sticky="e", pady=2)
        t3_combo_1.set("Select the point cloud used for prediction:")

        # Entry
        t3_entry_widget = ttk.Entry(tab3, width=30)
        t3_entry_widget.grid(row=3, column=1, sticky="e", pady=2)
        t3_entry_widget.insert(0, self.output_directory)

        # Button
        t3_features= ttk.Button (tab3, text="...", command=lambda: load_features_dialog(), width=10)
        t3_features.grid (row=1,column=2,sticky="e",padx=100)
        t3_configuration= ttk.Button (tab3, text="...", command=lambda: load_configuration_dialog(), width=10)
        t3_configuration.grid(row=2,column=2,sticky="e",padx=100)
        t3_output = ttk.Button(tab3, text="...", command=lambda:save_file_dialog(3), width=10)
        t3_output.grid(row=3, column=2, sticky="e", padx=100)

        t3_run_button= ttk.Button (tab3, text="Run", command=lambda:run_algorithm_3(self,t3_combo_1.get(),load_features,load_configuration), width=10)
        t3_run_button.grid (row=4,column=1,sticky="e",padx=100)
        t3_cancel_button= ttk.Button (tab3, text="Cancel", command=lambda:destroy,width=10)
        t3_cancel_button.grid (row=4,column=1,sticky="e")
        
        # To run the optimal flow   
        def run_algorithm_1 (self,name_list,pc_training_name,selected_indices,f,cv): 
            # Update de data
            self.set_up_parameters_of["percentage"]=f
            self.set_up_parameters_of["cv"]=cv
            
            # Check if the selection is a point cloud
            pc_training=check_input(name_list,pc_training_name)
            
            # Transform the point cloud into a dataframe and select only the interseting columns
            feature_selection_pcd=P2p_getdata(pc_training,False,True,True)
            
            # Save the point cloud with the features selected
            input_path_point_cloud=os.path.join(self.output_directory,"input_point_cloud.txt")
            feature_selection_pcd[self.features2include].to_csv(input_path_point_cloud,sep=' ',header=True,index=False)
            
            # Save the selectors of the optimal flow algorithm             
            selected_items = [self.set_up_parameters_of["selectors"][i] for i in selected_indices]
            with open(os.path.join(self.output_directory,'selected_params.txt'), "w") as output_file:
                output_file.write('\n'.join(selected_items))
                
            # YAML file
            yaml_of = {
                'ALGORITHM': "Optimal-flow",
                'INPUT_POINT_CLOUD': input_path_point_cloud,                
                'SELECTORS_FILE': os.path.join(self.output_directory,'selected_params.txt'),
                'OUTPUT_DIRECTORY': self.output_directory,
                'CONFIGURATION': {
                    'cv': self.set_up_parameters_of["cv"],
                    'f': self.set_up_parameters_of["percentage"]
                }
            }            
            
            
            write_yaml_file (self.output_directory,yaml_of)
            
            # RUN THE COMMAND LINE      
            command = path_optimal_flow + ' --i ' + os.path.join(self.output_directory,'algorithm_configuration.yaml') + ' --o ' + self.output_directory
            print (command)
            # os.system(command)
            print("The process has been finished") 
        # To run the machine learning segmentation
        def run_algorithm_2 (self,algo,pc_training_name,pc_testing_name):
            
            # Check if the selection is a point cloud
            pc_training=check_input(name_list,pc_training_name)
            pc_testing=check_input(name_list,pc_testing_name)
            
            # Convert to a pandasdataframe
            pcd_training=P2p_getdata(pc_training,False,True,True)
            pcd_testing=P2p_getdata(pc_testing,False,True,True)
            
            # Error control to prevent not algorithm for the training
            if algo=="Not selected":
                raise RuntimeError ("Please select and algorithm for the training")
            else: 
                
                # Create the features file
                comma_separated = ','.join(self.features2include)    
                with open(os.path.join(self.output_directory, 'features.txt'), 'w') as file:
                    file.write(comma_separated)
                # Save the point clouds and the features
                pcd_training.to_csv(os.path.join(self.output_directory, 'input_point_cloud_training.txt'),sep=' ',header=True,index=False)
                pcd_testing.to_csv(os.path.join(self.output_directory, 'input_point_cloud_testing.txt'),sep=' ',header=True,index=False)   

            if algo=="Random Forest":
                  
                # YAML file
                yaml = {
                    'INPUT_POINT_CLOUD_TRAINING': os.path.join(self.output_directory, 'input_point_cloud_training.txt'),  
                    'INPUT_POINT_CLOUD_TESTING': os.path.join(self.output_directory, 'input_point_cloud_testing.txt'), 
                    'INPUT_FEATURES': os.path.join(self.output_directory, 'features.txt'),
                    'OUTPUT_DIRECTORY': self.output_directory,
                    'ALGORITHM': "Random Forest",
                    'CONFIGURATION': 
                        {
                        'ne': self.set_up_parameters_rf["estimators"],
                        'c': self.set_up_parameters_rf["criterion"],
                        'md': self.set_up_parameters_rf["trees"],
                        'ms': self.set_up_parameters_rf["internal_node"],
                        'mns': self.set_up_parameters_rf["leaf_node"],
                        'mwf': self.set_up_parameters_rf["weights"],
                        'mf': self.set_up_parameters_rf["features"],
                        'bt': self.set_up_parameters_rf["bootstrap"],
                        's': self.set_up_parameters_rf["scoring"],
                        'nj': self.set_up_parameters_rf["njobs"]
                        }
                }                          
                command = path_random_forest + ' --i ' + os.path.join(self.output_directory,'algorithm_configuration.yaml') + ' --o ' + self.output_directory
                
            elif algo=="Auto Machine Learning":                
                # Join the list items with commas to create a comma-separated string
                comma_separated = ','.join(self.features2include)    
                # Write the comma-separated string to a text file
                with open(os.path.join(self.output_directory, 'features.txt'), 'w') as file:
                    file.write(comma_separated)  
                # YAML file
                yaml = {
                    'INPUT_POINT_CLOUD_TRAINING': os.path.join(self.output_directory, 'input_point_cloud_training.txt'),  
                    'INPUT_POINT_CLOUD_TESTING': os.path.join(self.output_directory, 'input_point_cloud_testing.txt'),  
                    'INPUT_FEATURES': os.path.join(self.output_directory, 'features.txt'),
                    'OUTPUT_DIRECTORY': self.output_directory,
                    'ALGORITHM': "TPOT",
                    'CONFIGURATION': 
                        {
                        'ge': self.set_up_parameters_aml["generations"],
                        'ps': self.set_up_parameters_aml["size"],
                        'mr': self.set_up_parameters_aml["mutation"],
                        'cr': self.set_up_parameters_aml["crossover"],
                        'cv': self.set_up_parameters_aml["cv"],
                        'mtm': self.set_up_parameters_aml["max_time_total"],
                        'metm': self.set_up_parameters_aml["max_time_ev"],
                        'ng': self.set_up_parameters_aml["no_improvement"],
                        's': self.set_up_parameters_aml["scoring"]
                        }
                } 
                write_yaml_file (self.output_directory,yaml)
                command = path_aml + ' --i ' + os.path.join(self.output_directory,'algorithm_configuration.yaml') + ' --o ' + self.output_directory
            
            # RUN THE COMMAND LINE
            os.system(command)            

            # CREATE THE RESULTING POINT CLOUD 
            # Load the predictions
            pcd_prediction = pd.read_csv(os.path.join(self.output_directory,'predictions.txt'), sep=',')  # Use sep='\t' for tab-separated files       
            # # Select only the 'Predictions' column
            pc_results_prediction = pycc.ccPointCloud(pcd_prediction['X'], pcd_prediction['Y'], pcd_prediction['Z'])
            pc_results_prediction.setName("Results_from_segmentation")
            idx = pc_results_prediction.addScalarField("Labels",pcd_prediction['Predictions']) 
            # STORE IN THE DATABASE OF CLOUDCOMPARE
            CC = pycc.GetInstance()
            CC.addToDB(pc_results_prediction)
            CC.updateUI() 
            root.destroy()
            # Revome files
            os.remove(os.path.join(self.output_directory,'predictions.txt'))
            os.remove(os.path.join(self.output_directory, 'input_point_cloud_training.txt'))
            os.remove(os.path.join(self.output_directory, 'input_point_cloud_testing.txt'))
            print("The process has been finished")            
        # To run the prediction of machine learning
        def run_algorithm_3 (self,pc_prediction_name,path_features,path_pickle):
            
            # Check if the selection is a point cloud
            pc_prediction=check_input(name_list,pc_prediction_name)
            
            # Convert to a pandasdataframe
            pcd_prediction=P2p_getdata(pc_prediction,False,True,True)
            
            # Save the point cloud
            pcd_prediction.to_csv(os.path.join(self.output_directory, 'input_point_cloud_prediction.txt'),sep=' ',header=True,index=False)
            command= path_prediction

            # YAML file
            yaml = {
                'INPUT_POINT_CLOUD': os.path.join(self.output_directory, 'input_point_cloud_prediction.txt'),  
                'OUTPUT_DIRECTORY': self.output_directory,
                'ALGORITHM': "Prediction",
                'CONFIGURATION': 
                    {
                    'f': path_features,
                    'p': path_pickle,
                    }
            } 
            write_yaml_file (self.output_directory,yaml)    
            # RUN THE COMMAND LINE
            command = path_prediction + ' --i ' + os.path.join(self.output_directory,'algorithm_configuration.yaml') + ' --o ' + self.output_directory
            os.system(command) 
            
            # CREATE THE RESULTING POINT CLOUD 
            # Load the predictions
            pcd_prediction = pd.read_csv(os.path.join(self.output_directory,'predictions.txt'), sep=',')  # Use sep='\t' for tab-separated files       

            # Select only the 'Predictions' column
            pc_results_prediction = pycc.ccPointCloud(pcd_prediction['X'], pcd_prediction['Y'], pcd_prediction['Z'])
            pc_results_prediction.setName("Results_from_segmentation")
            idx = pc_results_prediction.addScalarField("Labels",pcd_prediction['Predictions']) 
            
            # STORE IN THE DATABASE OF CLOUDCOMPARE
            CC = pycc.GetInstance()
            CC.addToDB(pc_results_prediction)
            CC.updateUI() 
            root.destroy()
            
            # Revome files
            os.remove(os.path.join(self.output_directory,'predictions.txt'))
            os.remove(os.path.join(self.output_directory, 'input_point_cloud_prediction.txt'))
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