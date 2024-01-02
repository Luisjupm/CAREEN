# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 10:47:10 2023

@author: Digi_2

"""


#%% LIBRARIES
import os
import subprocess
import sys

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog

import pandas as pd
import numpy as np
import pickle
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import InterclusterDistance
import matplotlib.pyplot as plt
from yellowbrick.cluster import SilhouetteVisualizer

# FEATURE SELECTION
import time

#CloudCompare Python Plugin
import cccorelib
import pycc


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

#%% INPUTS AT THE BEGINING
name_list=get_point_clouds_name()
current_directory=os.path.dirname(os.path.abspath(__file__))
output_directory=os.path.join(current_directory,'..','temp_folder_for_results','Machine_Learning','OUTPUT')

#Feature selection
processing_file_of=os.path.join(current_directory,'optimal_flow-0.1.11\\optimal_flow.exe')

#Classification
processing_file_rf=os.path.join(current_directory,'random_forest-1.3.2\\random_forest.exe')
processing_file_tpot=os.path.join(current_directory,'tpot-0.12.1\\TPOT.exe')

#Prediction
processing_file_p=os.path.join(current_directory,'prediction_sklearn-1.3.2\\prediction_sklearn.exe')

#%% GUI
class GUI:
    def __init__(self):
        # Initial features2include
        self.features2include=[]
        self.values_list=[]
        self.features=[]
        # Initial paramertes for the different algorithms
        self.set_up_parameters_of=(['kbest_f','rfe_lr','rfe_tree','rfe_rf','rfecv_rf'],25,5)
        self.set_up_parameters_rf=(200,'gini',100,2,1,0,'sqrt',True,'balance_accuracy')
        self.set_up_parameters_aml=(5,20,0.9,0.1,2,60,10,2,'balance_accuracy')
        #Directory to save the files (training)
        self.file_path=os.path.join(current_directory, output_directory)
        #Directory to load the features file (prediction)
        self.load_features=os.path.join(current_directory, output_directory)
        #Directory to load the configuration file (prediction)
        self.load_configuration=os.path.join(current_directory, output_directory)
        #List with the features loaded (prediction)
        self.features_prediction=[]
        
    def main_frame (self, root):
        
        # GUI MACHINE LEARNING SEGMENTATION
        root.title ("Machine Learning Segmentation")
        root.resizable (False, False)
        # Remove minimize and maximize button 
        root.attributes ('-toolwindow',-1)

        tab_control = ttk.Notebook(root)
        tab_control.pack(expand=1, fill="both")
        
        # Create 3 tabs
        tab1 = ttk.Frame(tab_control)
        tab1.pack()

        tab2 = ttk.Frame(tab_control)
        tab2.pack()

        tab3 = ttk.Frame(tab_control)
        tab3.pack()

        tab_control.add(tab1, text='Feature selection')
        tab_control.add(tab2, text='Classification')
        tab_control.add(tab3, text='Prediction')
        tab_control.pack(expand=1, fill="both")
        
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
        
        
        # TAB1 = FEATURE SELECTION
        
        # Listbox
        def on_ok_button_click():
            selected_items = listbox.curselection()
            selected_values = [listbox.get(index) for index in selected_items]
            messagebox.showinfo("Selectors", f"Selected options: {', '.join(selected_values)}")
        
        def save_selection_to_file():
            selected_indices = listbox.curselection()
            selected_items = [selectors[i] for i in selected_indices]

            with open(os.path.join(output_directory, "selected_params.txt"), "w") as output_file:
                output_file.write('\n'.join(selected_items))

        def load_selection_from_file():
            file_path = os.path.join(output_directory, "selected_params.txt")
            if os.path.exists(file_path):
                with open(file_path, "r") as input_file:
                    selected_items = input_file.read().splitlines()
                    selected_indices = [selectors.index(item) for item in selected_items]

                # Clear previous selections
                listbox.selection_clear(0, tk.END)

                # Select the items from the file
                for i in selected_indices:
                    listbox.selection_set(i)
        load_selection_from_file()
        
        selectors = ['kbest_f','rfe_lr','rfe_tree','rfe_rf','rfecv_tree','rfecv_rf','rfe_svm','rfecv_svm']
        s_var = [tk.IntVar() for _ in selectors]
        s_checklist = []
        listbox = tk.Listbox(tab1, selectmode=tk.MULTIPLE, height=len(selectors))
        for value in selectors:
            listbox.insert(tk.END, value)
        # Select first six elements
        for i in range(6):
            listbox.selection_set(i)
        listbox.grid(row=1, column=1, sticky="e", padx=10, pady=10)
        
        # Load the selection from the file
        load_selection_from_file()
        # Get the selected parameters from the Listbox
        selected_params = [selectors[i] for i in listbox.curselection()]
        

        # Labels
        t1_label_1= ttk.Label(tab1, text="Choose point cloud")
        t1_label_1.grid(column=0, row=0, pady=2, sticky="w")
        t1_label_2= ttk.Label(tab1, text="Selectors", cursor="question_arrow")
        t1_label_2.grid(column=0, row=1, pady=2, sticky="w")
        t1_label_3= ttk.Label(tab1, text="Features to include")
        t1_label_3.grid(column=0, row=2, pady=2, sticky="w")
        
        t1_label_4= ttk.Label(tab1, text="Number of features to consider")
        t1_label_4.grid(column=0, row=3, pady=2, sticky="w")
        t1_label_5= ttk.Label(tab1, text="Folds for cross-validation")
        t1_label_5.grid(column=0, row=4, pady=2, sticky="w")
        t1_label_6= ttk.Label(tab1, text="Choose output directory")
        t1_label_6.grid(column=0, row=5, pady=2, sticky="w")

        # Tooltips
        t1_help_2 = "NOTE: SVM based selectors are highly sensitive to the number of features(high-dimension)\n " \
                 "and training records number, i.e.rfe_svm and rfecv_svm. When features number > 50 w/ records\n " \
                 "number over 50K,otherwise will result in long processing time."
        create_tooltip(t1_label_2, t1_help_2)

        t1_help_4 = "The choice of the number of folds (cv) depends on the size of your dataset and the trade-off between computation\n " \
                 "time and the reliability of the performance estimate. Common choices include 5-fold and 10-fold cross-validation."
        create_tooltip(t1_label_4, t1_help_4)

        # Combobox
        t1_combo_1=ttk.Combobox (tab1,values=name_list)
        t1_combo_1.grid(column=1, row=0, sticky="e", pady=2)
        t1_combo_1.set("Select the point cloud used for feature selection")

        # Entry
        t1_entry_features = ttk.Entry(tab1, width=10)
        t1_entry_features.insert(0,self.set_up_parameters_of[0])
        t1_entry_features.grid(row=3, column=1, sticky="e", pady=2)
        
        t1_entry_cv = ttk.Entry(tab1, width=10)
        t1_entry_cv.insert(0,self.set_up_parameters_of[1])
        t1_entry_cv.grid(row=4, column=1, sticky="e", pady=2)
        
        self.entry_widget = ttk.Entry(tab1, width=30)
        self.entry_widget.grid(row=5, column=1, sticky="e", pady=2)
        self.entry_widget.insert(0, self.file_path)

        # Button
        t1_ok_button_selectors = tk.Button(tab1, text="OK", command=on_ok_button_click,width=10)
        t1_ok_button_selectors.grid(row=1, column=2, pady=10)
        t1_output = ttk.Button(tab1, text="...", command=self.save_file_dialog, width=10)
        t1_output.grid(row=5, column=2, sticky="e", padx=100)
        t1_features= ttk.Button (tab1, text="...", command=lambda: self.show_features_window(t1_combo_1.get()), width=10)
        t1_features.grid (row=2,column=2,sticky="e",padx=100)
        
        t1_run_button= ttk.Button (tab1, text="OK", command=lambda:self.run_algorithm_1(), width=10)
        t1_run_button.grid (row=6,column=1,sticky="e",padx=100)
        t1_cancel_button= ttk.Button (tab1, text="Cancel", command=self.destroy,width=10)
        t1_cancel_button.grid (row=6,column=1,sticky="e")
        
        
        # TAB2 = CLASSIFICATION

        # Labels
        t2_label_1= ttk.Label(tab2, text="Choose point cloud for training")
        t2_label_1.grid(column=0, row=0, pady=2, sticky="w")
        t2_label_2= ttk.Label(tab2, text="Choose point cloud for testing")
        t2_label_2.grid(column=0, row=1, pady=2,sticky="w")
        t2_label_3= ttk.Label(tab2, text="Select machine learning algorithm")
        t2_label_3.grid(column=0, row=2, pady=2, sticky="w")
        t2_label_4= ttk.Label(tab2, text="Select the features to include")
        t2_label_4.grid(column=0, row=3, pady=2, sticky="w")
        t2_label_5= ttk.Label(tab2, text="Choose features from the feature selection")
        t2_label_5.grid(column=0, row=4, pady=2, sticky="w")
        t2_label_6= ttk.Label(tab2, text="Choose output directory")
        t2_label_6.grid(column=0, row=5, pady=2, sticky="w")

        # Tooltips
        t2_help_1 = "After selecting the machine learning algorithm, please click Set-up for change parameters"
        create_tooltip(t2_label_3, t2_help_1)

        # Combobox
        t2_combo_1=ttk.Combobox (tab2,values=name_list)
        t2_combo_1.grid(column=1, row=0, sticky="e", pady=2)
        t2_combo_1.set("Select the point cloud used for training")
        
        t2_combo_2=ttk.Combobox (tab2,values=name_list)
        t2_combo_2.grid(column=1, row=1, sticky="e", pady=2)
        t2_combo_2.set("Select the point cloud used for testing")
        
        algorithms=["Random Forest", "Logistic Regression", "Auto Machine Learning"]
        t2_combo_3=ttk.Combobox (tab2,values=algorithms, state="readonly")
        t2_combo_3.current(0)
        t2_combo_3.grid(column=1, row=2, sticky="e", pady=2)
        t2_combo_3.set("Not selected")
        
        # Entry
        self.entry_widget = ttk.Entry(tab2, width=30)
        self.entry_widget.grid(row=5, column=1, sticky="e", pady=2)
        self.entry_widget.insert(0, self.file_path)
        
        # Button
        t2_setup_button= ttk.Button (tab2, text="Set-up", command=lambda: self.show_set_up_window(t2_combo_3.get()), width=10)
        t2_setup_button.grid (row=2,column=2,sticky="e",padx=100)
        t2_features_button= ttk.Button (tab2, text="...", command=lambda: self.load_features_dialog(), width=10)
        t2_features_button.grid (row=3,column=2,sticky="e",padx=100)
        t2_configuration= ttk.Button (tab2, text="...", command=lambda: self.load_configuration_dialog(), width=10)
        t2_configuration.grid(row=4,column=2,sticky="e",padx=100)
        t2_output = ttk.Button(tab2, text="...", command=self.save_file_dialog, width=10)
        t2_output.grid(row=5, column=2, sticky="e", padx=100)
    
        t2_run_button= ttk.Button (tab2, text="Run", command=self.run_algorithm_2(t2_combo_1.get(),t2_combo_2.get(),t2_combo_3.get()), width=10)
        t2_run_button.grid (row=6,column=1,sticky="e",padx=100)
        t2_cancel_button= ttk.Button (tab2, text="Cancel", command=self.destroy,width=10)
        t2_cancel_button.grid (row=6,column=1,sticky="e")
        
        
        # TAB3= PREDICTION

        # Labels
        t3_label_1= ttk.Label(tab3, text="Choose point cloud for prediction")
        t3_label_1.grid(column=0, row=0, pady=2, sticky="w")
        t3_label_2= ttk.Label(tab3, text="Load feature file")
        t3_label_2.grid(column=0, row=1, pady=2, sticky="w")
        t3_label_3= ttk.Label(tab3, text="Load pkl file")
        t3_label_3.grid(column=0, row=2, pady=2, sticky="w")
        t3_label_4= ttk.Label(tab3, text="Choose output directory")
        t3_label_4.grid(column=0, row=3, pady=2, sticky="w")

        # Combobox
        t3_combo_1=ttk.Combobox (tab3,values=name_list)
        t3_combo_1.grid(column=1, row=0, sticky="e", pady=2)
        t3_combo_1.set("Select the point cloud used for prediction")

        # Entry
        self.entry_widget = ttk.Entry(tab3, width=30)
        self.entry_widget.grid(row=3, column=1, sticky="e", pady=2)
        self.entry_widget.insert(0, self.file_path)

        # Button
        t3_features= ttk.Button (tab3, text="...", command=lambda: self.load_features_dialog(), width=10)
        t3_features.grid (row=1,column=2,sticky="e",padx=100)
        t3_configuration= ttk.Button (tab3, text="...", command=lambda: self.load_configuration_dialog(), width=10)
        t3_configuration.grid(row=2,column=2,sticky="e",padx=100)
        t3_output = ttk.Button(tab3, text="...", command=self.save_file_dialog, width=10)
        t3_output.grid(row=3, column=2, sticky="e", padx=100)

        t3_run_button= ttk.Button (tab3, text="OK", command=self.run_algorithm_3(t3_combo_1.get()), width=10)
        t3_run_button.grid (row=4,column=1,sticky="e",padx=100)
        t3_cancel_button= ttk.Button (tab3, text="Cancel", command=self.destroy,width=10)
        t3_cancel_button.grid (row=4,column=1,sticky="e")
        
        

        
    def save_setup_parameters (self,algo,*params): 
        if algo=="Random Forest":
            self.set_up_parameters_rf=params
        elif algo=="Auto Machine Learning":
            self.set_up_parameters_aml=params 
      
    
    def show_set_up_window (self,algo): 
        
        def on_ok_button_click(algo):
            if algo=="Random Forest":
                self.save_setup_parameters(algo, int(entry_param1_rf.get()), str(combo_param2_rf.get()), int(entry_param3_rf.get()), int(entry_param4_rf.get()), int(entry_param5_rf.get()), int(entry_param6_rf.get()), str(combo_param7_rf.get()), bool(combo_param8_rf.get()), str(combo_param9_rf.get()))
            elif algo=="Auto Machine Learning":
                self.save_setup_parameters(algo, int(entry_param1_aml.get()), int(entry_param2_aml.get()), float(entry_param3_aml.get()), float(entry_param4_aml.get()), int(entry_param5_aml.get()), int(entry_param6_aml.get()), int(entry_param7_aml.get()), int(entry_param8_aml.get()), str(combo_param9_aml.get()))
           
            set_up_window.destroy()  # Close the window after saving parameters
            
        # Setup window
        set_up_window = tk.Toplevel(root)
        set_up_window.title("Set Up the algorithm")
        set_up_window.resizable (False, False)
        # Remove minimize and maximize button 
        set_up_window.attributes ('-toolwindow',-1)
           
        if algo=="Random Forest":
            
            #Labels
                
            label_1= ttk.Label(set_up_window, text="Number of estimators")
            label_1.grid(column=0, row=0, pady=2, sticky="w")
            label_2= ttk.Label(set_up_window, text="Criterion")
            label_2.grid(column=0, row=1, pady=2, sticky="w")
            label_3= ttk.Label(set_up_window, text="Maximum depth of trees")
            label_3.grid(column=0, row=2, pady=2, sticky="w")
            label_4= ttk.Label(set_up_window, text="Minimum number of samples required to Split the internal node")
            label_4.grid(column=0, row=3, pady=2, sticky="w")
            label_5= ttk.Label(set_up_window, text="Minimum number of samples required to be a leaf node")
            label_5.grid(column=0, row=4, pady=2, sticky="w")
            label_6= ttk.Label(set_up_window, text="Minimum weight fraction of the total sum of weights")
            label_6.grid(column=0, row=5, pady=2, sticky="w")
            label_7= ttk.Label(set_up_window, text="Maximum number of features")
            label_7.grid(column=0, row=6, pady=2, sticky="w")
            label_8= ttk.Label(set_up_window, text="Bootstrap")
            label_8.grid(column=0, row=7, pady=2, sticky="w")
            label_9= ttk.Label(set_up_window, text="Scoring")
            label_9.grid(column=0, row=8, pady=2, sticky="w")
            
            # Params
            
            entry_param1_rf= tk.Entry(set_up_window, width=10)
            entry_param1_rf.insert(0,self.set_up_parameters_rf[0])
            entry_param1_rf.grid(row=0, column=1, sticky="e")
            
            criterion = ["gini","entropy","log_loss"]
            combo_param2_rf=ttk.Combobox (set_up_window, values=criterion, state="readonly")
            combo_param2_rf.set(self.set_up_parameters_rf[1])
            combo_param2_rf.grid(column=1, row=1, sticky="e", pady=2)
            
            entry_param3_rf= ttk.Entry(set_up_window, width=10)
            entry_param3_rf.insert(0,self.set_up_parameters_rf[2])
            entry_param3_rf.grid(row=2, column=1, sticky="e")
            
            entry_param4_rf= ttk.Entry(set_up_window, width=10)
            entry_param4_rf.insert(0,self.set_up_parameters_rf[3])
            entry_param4_rf.grid(row=3, column=1, sticky="e")
            
            entry_param5_rf= ttk.Entry(set_up_window, width=10)
            entry_param5_rf.insert(0,self.set_up_parameters_rf[4])
            entry_param5_rf.grid(row=4, column=1, sticky="e")
            
            entry_param6_rf= ttk.Entry(set_up_window, width=10)
            entry_param6_rf.insert(0,self.set_up_parameters_rf[5])
            entry_param6_rf.grid(row=5, column=1, sticky="e")
            
            max_n_features = ['sqrt','log2']
            combo_param7_rf=ttk.Combobox (set_up_window, values=max_n_features, state="readonly")
            combo_param7_rf.set(self.set_up_parameters_rf[6])
            combo_param7_rf.grid(column=1, row=6, sticky="e", pady=2)
            
            bootstrap= [True, False]
            combo_param8_rf=ttk.Combobox (set_up_window, values=bootstrap, state="readonly")
            combo_param8_rf.set(self.set_up_parameters_rf[7])
            combo_param8_rf.grid(column=1, row=7, sticky="e", pady=2)
            
            scoring=["balanced_accuracy","accuracy","f1","f1_weighted","precision","precision_weighted"]
            combo_param9_rf=ttk.Combobox (set_up_window, values=scoring, state="readonly")
            combo_param9_rf.set(self.set_up_parameters_rf[8])
            combo_param9_rf.grid(column=1, row=8, sticky="e", pady=2)
            
            ok_button = ttk.Button(set_up_window, text="OK", command=lambda: on_ok_button_click(algo), width=10)
            ok_button.grid(row=9, column=1, sticky="w", padx=100)

                   
        if algo=="Auto Machine Learning":
            
            #Labels
                
            label_1= ttk.Label(set_up_window, text="Generations")
            label_1.grid(column=0, row=0, pady=2, sticky="w")
            label_2= ttk.Label(set_up_window, text="Population size")
            label_2.grid(column=0, row=1, pady=2, sticky="w")
            label_3= ttk.Label(set_up_window, text="Mutation rate")
            label_3.grid(column=0, row=2, pady=2, sticky="w")
            label_4= ttk.Label(set_up_window, text="Crossover rate")
            label_4.grid(column=0, row=3, pady=2, sticky="w")
            label_5= ttk.Label(set_up_window, text="Number of flods for k-fold cross-validation")
            label_5.grid(column=0, row=4, pady=2, sticky="w")
            label_6= ttk.Label(set_up_window, text="Maximum time in mins for the evaluation")
            label_6.grid(column=0, row=5, pady=2, sticky="w")
            label_7= ttk.Label(set_up_window, text="Maximum time in mins for each evaluation")
            label_7.grid(column=0, row=6, pady=2, sticky="w")
            label_8= ttk.Label(set_up_window, text="Number of generations without improvement")
            label_8.grid(column=0, row=7, pady=2, sticky="w")
            label_9= ttk.Label(set_up_window, text="Scoring")
            label_9.grid(column=0, row=8, pady=2, sticky="w")
            
            # Params
            
            entry_param1_aml= ttk.Entry(set_up_window, width=10)
            entry_param1_aml.insert(0,self.set_up_parameters_aml[0])
            entry_param1_aml.grid(row=0, column=1, sticky="e")
            
            entry_param2_aml= ttk.Entry(set_up_window, width=10)
            entry_param2_aml.insert(0,self.set_up_parameters_aml[1])
            entry_param2_aml.grid(row=1, column=1, sticky="e")
            
            entry_param3_aml= ttk.Entry(set_up_window, width=10)
            entry_param3_aml.insert(0,self.set_up_parameters_aml[2])
            entry_param3_aml.grid(row=2, column=1, sticky="e")
            
            entry_param4_aml= ttk.Entry(set_up_window, width=10)
            entry_param4_aml.insert(0,self.set_up_parameters_aml[3])
            entry_param4_aml.grid(row=3, column=1, sticky="e")
            
            entry_param5_aml= ttk.Entry(set_up_window, width=10)
            entry_param5_aml.insert(0,self.set_up_parameters_aml[4])
            entry_param5_aml.grid(row=4, column=1, sticky="e")
            
            entry_param6_aml= ttk.Entry(set_up_window, width=10)
            entry_param6_aml.insert(0,self.set_up_parameters_aml[5])
            entry_param6_aml.grid(row=5, column=1, sticky="e")
            
            entry_param7_aml= ttk.Entry(set_up_window, width=10)
            entry_param7_aml.insert(0,self.set_up_parameters_aml[6])
            entry_param7_aml.grid(row=6, column=1, sticky="e")
            
            entry_param8_aml= ttk.Entry(set_up_window, width=10)
            entry_param8_aml.insert(0,self.set_up_parameters_aml[7])
            entry_param8_aml.grid(row=7, column=1, sticky="e")
            
            scoring=["balanced_accuracy","accuracy","f1","f1_weighted","precision","precision_weighted"]
            combo_param9_aml=ttk.Combobox (set_up_window, values=scoring, state="readonly")
            combo_param9_aml.set(self.set_up_parameters_aml[8])
            combo_param9_aml.grid(column=1, row=8, sticky="e", pady=2)
            
            ok_button = ttk.Button(set_up_window, text="OK", command=lambda: on_ok_button_click(algo), width=10)
            ok_button.grid(row=9, column=1, sticky="w", padx=100)
            
    def save_file_dialog(self, entry_widgets):
        global output_directory
        directory = filedialog.askdirectory()
        if directory:
            output_directory = directory

            # Update all entry widgets
            for entry_widget in entry_widgets:
                entry_widget.delete(0, tk.END)
                entry_widget.insert(0, output_directory)
                
            # Change script directory before writting the pickle file
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
            
            # Save the last selected folder in a pickle file
            with open('config.pkl', 'wb') as file:
                pickle.dump(output_directory, file)

    def create_entry_button_output(self, tab, row):
        entry_widget = ttk.Entry(tab, width=30)
        entry_widget.grid(row=row, column=1, sticky="e", pady=2)
        entry_widget.insert(0, output_directory)

        button_widget = ttk.Button(tab, text="...", command=lambda: self.save_file_dialog([entry_widget]), width=10)
        button_widget.grid(row=row, column=2, sticky="e", padx=100)
        return entry_widget
    
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

    def select_all_checkbuttons(self,checkbuttons_vars): 
        for var in checkbuttons_vars:
            var.set(True)   
            
    def show_features_window(self,training_pc_name): 
        """ESTA MANTENLA TAL CUAL ES LA DE SELECCIÓN DE FEATURES""" """SE LLAMA EN LA LINEA 132 Y TIENE DE INPUT EL NOMBRE 
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
   
    def destroy (self): 
        root.destroy ()


    def run_algorithm_1 (self,algo,classification_pc_name):
        
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
            for i, item in enumerate(name_list):
                if item == classification_pc_name:
                    pc_training = entities.getChild(i)
                    break
        feature_selection_pcd=P2p_getdata(pc_training,False,False,True)
        pcd_f=feature_selection_pcd[self.features2include].values
        
        # COMMAND
        s = self.set_up_parameters_of[0].get()
        f = self.set_up_parameters_of[1].get()
        cv = self.set_up_parameters_of[2].get()
        
        command = processing_file_of + ' --i ' + os.path.join(output_directory, 'input_features.txt') + ' --o ' + output_directory + ' --s ' + s + ' --f ' + f + ' --cv ' + cv
        return command 
        
        # Save the point cloud
        feature_selection_pcd.to_csv(os.path.join(output_directory, 'input_features.txt'),sep=' ',header=True,index=False)
        
        # # RUN THE COMMAND LINE
        print(command)
        os.system(command)
        
        def read_features_and_print(output_directory):
            features_file = os.path.join(output_directory, "features.txt")
            # Wait until the file exists
            while not os.path.exists(features_file):
                time.sleep(1)  # Wait 1 sec 
            # Read and print the selected features 
            with open(features_file, "r") as file:
                features = file.read()
        
            print("Best features selected:", features)
            print("The process has been finished")
            
        read_features_and_print(output_directory)
        
        
        
        
    def run_algorithm_2 (self,algo,training_pc_name,testing_pc_name):
        print("Hola")
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
            pc_testing=entities
        else:
            for ii, item in enumerate(name_list):
                if item == training_pc_name:
                    pc_training = entities.getChild(ii)
                    break
            for it, item in enumerate(name_list):
                if item== testing_pc_name:
                    pc_testing=entities.getChild(it)
                    break
        pcd_training=P2p_getdata(pc_training,False,False,True)
        pcd_testing=P2p_getdata(pc_testing,False,False,True)
        pcd_f=pcd_training[self.features2include].values 

        # Error control to prevent not algorithm for the training
        if algo=="Not selected":
            raise RuntimeError ("Please select and algorithm for the training")
        elif algo=="Random Forest":
            
            ne=self.set_up_parameters_rf[0].get()
            c=self.set_up_parameters_rf[1].get()
            md=self.set_up_parameters_rf[2].get()
            ms=self.set_up_parameters_rf[3].get()
            mns=self.set_up_parameters_rf[4].get()
            mwf=self.set_up_parameters_rf[5].get()
            mf=self.set_up_parameters_rf[6].get()
            bt=self.set_up_parameters_rf[7].get()
            s=self.set_up_parameters_rf[8].get()
            nj_str=-1
            nj=str(nj_str)
            
            command = processing_file_rf + ' --te ' + os.path.join(output_directory, 'input_class_test.txt') + ' --tr ' + os.path.join(output_directory, 'input_class_train.txt') + ' --o ' + output_directory + ' --f ' + os.path.join(output_directory, 'features.txt') + ' --ne ' + ne + ' --c ' + c + ' --md ' + md + ' --ms ' + ms + ' --mns ' + mns + ' --mwf ' + mwf + ' --mf ' + mf + ' --bt ' + bt + ' --s ' + s + ' --nj ' + nj
            return (command)
            
        elif algo=="Auto Machine Learning":
            
            ge = self.set_up_parameters_aml[0].get()
            ps = self.set_up_parameters_aml[1].get()
            mr = self.set_up_parameters_aml[2].get()
            cr = self.set_up_parameters_aml[3].get()
            cv = self.set_up_parameters_aml[4].get()
            mtm = self.set_up_parameters_aml[5].get()
            metm = self.set_up_parameters_aml[6].get()
            ng = self.set_up_parameters_aml[7].get()
            s = self.set_up_parameters_aml[8].get()
            
            command = processing_file_tpot + ' --te ' + os.path.join(output_directory, 'input_class_test.txt') + ' --tr ' + os.path.join(output_directory, 'input_class_train.txt') + ' --o ' + output_directory + ' --f ' + os.path.join(output_directory, 'features.txt') + ' --ge ' + ge + ' --ps ' + ps + ' --mr ' + mr + ' --cr ' + cr + ' --cv ' + cv + ' --mtm ' + mtm + ' --metm ' + metm + ' --ng ' + ng + ' --s ' + s
            return (command)
        
        
        # Save the point clouds and the features
        # Join the list items with commas to create a comma-separated string
        comma_separated = ','.join(self.features2include)    
        # Write the comma-separated string to a text file
        with open(os.path.join(output_directory, 'features.txt'), 'w') as file:
            file.write(comma_separated)  
        pcd_training.to_csv(os.path.join(output_directory, 'input_class_train.txt'),sep=' ',header=True,index=False)
        pcd_testing.to_csv(os.path.join(output_directory, 'input_class_test.txt'),sep=' ',header=True,index=False)   

        
        # # RUN THE COMMAND LINE
        print(command)
        os.system(command)    
        
        def read_features_and_print(output_directory):
            predictions_file = os.path.join(output_directory, 'predictions.txt')
            
            # Wait until the file exists
            while not os.path.exists(predictions_file):
                time.sleep(1)  # Wait 1 sec before to repeat the verifying
        
            # Read and process the features
            data = pd.read_csv(predictions_file, delimiter=',')  # Assuming columns separated by commas
        
            # Create the resulting point cloud
            pc_prediction = pycc.ccPointCloud(data['X'], data['Y'], data['Z'])
            pc_prediction.setName("Results_from_prediction")
            pc_prediction.addScalarField("Predictions", data['Predictions'])
        
            # Add the result to CLoudCompare
            CC.addToDB(pc_prediction)
            CC.updateUI()
        
        # Call function after executting command
        read_features_and_print(output_directory)
        
        print("The process has been finished")
        
        
        
    
    def run_algorithm_3 (self,classification_pc_name):
        
        CC= pycc.GetInstance()
        entities= CC.getSelectedEntities()[0]
        index=-1
        for i, item in enumerate (name_list):
            if item== classification_pc_name:
                pc_classification=entities.getChild(i)
                break  
        pcd_classification=P2p_getdata(pc_classification,False,True,True)
        f='features'
        # Save the point clouds
        pcd_classification.to_csv(os.path.join(output_directory, 'predictions.txt'),sep=' ',header=True,index=False)
        
        #COMMAND
        command = processing_file_p + ' --i ' + os.path.join(output_directory, 'predictions.txt')+ ' --o ' + output_directory +  ' --f ' + load_features + ' --p ' + load_configuration

        # RUN THE COMMAND LINE
        print(command)
        os.system(command) 
        
        data = pd.read_csv(os.path.join(output_directory, 'predictions.txt'), delimiter=',')  # Assumes comma-separated columns
        # Create the resulting point cloud
        pc_prediction = pycc.ccPointCloud(data['X'], data['Y'], data['Z'])
        pc_prediction.setName("Results_from_prediction")
        pc_prediction.addScalarField("Predictions", data['Predictions']) 
        CC.addToDB(pc_prediction)
        CC.updateUI()
        print("The process has been finished") 

# START THE MAIN WINDOW        
root = tk.Tk()
app = GUI()
app.main_frame(root)
root.mainloop()
