# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 13:15:50 2023

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

# FEATURE SELECTION
import time
# from optimalflow.autoFS import dynaFS_clf

# CLASSIFICATION
#Random Forest Libraries
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import f1_score
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import itertools
# from matplotlib import pyplot as plt
# import pickle
# #Auto Machine Learning Libraries
# import IProgress
# from tpot import TPOTClassifier

#CloudCompare Python Plugin
import cccorelib
import pycc


#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS
script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'
sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance,get_point_clouds_name

#%% DEFINING INPUTS OF CMD
current_directory=os.path.dirname(os.path.abspath(__file__))
temp_folder=os.path.join(current_directory,'..','temp')
                          
#Feature selection
input_file=os.path.join(current_directory,'..','temp_folder_for_results','Machine_Learning','INPUT_feature_selection','input_features.txt')
processing_file_of=os.path.join(current_directory,'optimal_flow-0.1.11\\optimal_flow-0.1.11.exe')

#Classification
output_directory=os.path.join(current_directory,'..','temp_folder_for_results','Machine_Learning','OUTPUT')

processing_file_rf=os.path.join(current_directory,'random_forest-1.3.2\\random_forest.exe')
processing_file_tpot=os.path.join(current_directory,'t_pot-0.12.1\\TPOT.exe')

#Prediction
processing_file_p=os.path.join(current_directory,'prediction_sklearn-1.3.2\\prediction_sklearn.exe')

#%% FUNCTIONS FEATURE SELECTION
def destroy ():
    root.destroy ()

def open_file_pkl():
    global path_pkl    
    file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
    if file_path:
        path_pkl = file_path
        
def open_file_feature():
    global path_feature  
    file_path = filedialog.askopenfilename(filetypes=[("Feature file", "*.txt")])
    if file_path:
        path_feature = file_path

def choose_output_directory():
    output_directory = filedialog.askdirectory()
    if output_directory:
        print("Output directory selected:", output_directory)
        
def create_tooltip(widget, text):
    widget.bind("<Enter>", lambda event: show_tooltip(text))
    widget.bind("<Leave>", hide_tooltip)

def show_tooltip(text):
    tooltip.config(text=text)
    tooltip.place(relx=0.5, rely=0.5, anchor="center", bordermode="outside")
    
def hide_tooltip(event):
    tooltip.place_forget()

def create_entry(parent, text, row, default_value):
    label = tk.Label(parent, text=text)
    label.grid(column=0, row=row, pady=2, sticky="w")
    entry_var = tk.StringVar()
    entry_var.set(default_value)
    entry = tk.Entry(parent, width=5, textvariable=entry_var)
    entry.grid(column=1, row=row, sticky="e", pady=2)
    return entry_var

def create_combobox(frame, values, row):
    combo_var = tk.StringVar()
    combo = ttk.Combobox(frame, textvariable=combo_var, values=values)
    combo.grid(column=1, row=row, sticky="e", pady=2)
    combo.set(values[0])
    return combo_var
    
def select_directory():
    global output_directory
    directory = filedialog.askdirectory()
    if directory:
        output_directory = directory
        entry_a.delete(0, tk.END)
        entry_a.insert(0, output_directory)
    elif directory:
        output_directory = directory
        entry_b.delete(0, tk.END)
        entry_b.insert(0, output_directory)
    elif directory:
        output_directory = directory
        entry_c.delete(0, tk.END)
        entry_c.insert(0, output_directory)

        
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



def execute_selected_function():
    selected_function = combo3.get()
    if selected_function == "Random Forest":
        random_forest()
    elif selected_function == "Logistic Regression":
        logistic_regression()
    elif selected_function == "Auto Machine Learning":
        auto_machine_learning()
        
def random_forest(): 
    rf_window = tk.Toplevel()
    rf_window.title("Random Forest")
    rf_window.resizable (False, False) 
    rf_window.attributes ('-toolwindow',-1)
    canvas = tk.Canvas(rf_window)
    rf_frame = tk.Frame(canvas)
    rf_frame.pack(side="bottom", fill="x")
    canvas.create_window((0, 0), window=rf_frame, anchor="nw")
    canvas.pack(side="left", fill="both", expand=True)
    rf_frame.grid(column=0, row=0, padx=10, pady=10)
    
    #Labels
    label_texts = [
    "Number of estimators",
    "Criterion",
    "Maximum depth of trees",
    "Minimum number of samples required to Split the internal node",
    "Minimum number of samples required to be a leaf node",
    "Minimum weight fraction of the total sum of weights",
    "Maximum number of features",
    "Bootstrap",
    "Scoring"
    ]
    for row, text in enumerate(label_texts):
        label = ttk.Label(rf_frame, text=text)
        label.grid(column=0, row=row, pady=2, sticky="w")
    
    #Entry
    entry_var_1 = create_entry(rf_frame, "Number of estimators", 0, "200")
    entry_var_3 = create_entry(rf_frame, "Maximum depth of trees", 2, "100")
    entry_var_4 = create_entry(rf_frame, "Minimum number of samples required to Split the internal node", 3, "2")
    entry_var_5 = create_entry(rf_frame, "Minimum number of samples required to be a leaf node", 4, "1")
    entry_var_6 = create_entry(rf_frame, "Minimum weight fraction of the total sum of weights", 5, "0")
    
    #Combobox 
    combo2=create_combobox(rf_frame, criterion, 1)
    combo7=create_combobox(rf_frame, max_n_features, 6)
    combo8=create_combobox(rf_frame, bootstrap, 7)
    combo9=create_combobox(rf_frame, scoring, 8)
    
    def close_window ():
        rf_window.destroy ()
    ok_button= tk.Button (rf_frame, text="OK", command= close_window, width=10)
    ok_button.grid (row=9,column=1,sticky="w",padx=100)
    
    return {
            'entry_var_1': entry_var_1,
            'combo2': combo2,
            'entry_var_3': entry_var_3,
            'entry_var_4': entry_var_4,
            'entry_var_5': entry_var_5,
            'entry_var_6': entry_var_6,
            'combo7': combo7,
            'combo8': combo8,
            'combo9': combo9
            }
    
def logistic_regression(): 
    lr_window = tk.Toplevel()
    lr_window.title("Logistic Regression") 
    canvas = tk.Canvas(lr_window)
    lr_frame = tk.Frame(canvas)
    lr_frame.pack(side="bottom", fill="x")
    canvas.create_window((0, 0), window=lr_frame, anchor="nw")
    canvas.pack(side="left", fill="both", expand=True)
    
    #Labels
    label_texts = ["This algorithm still doesn't work"]
    for index, text in enumerate(label_texts):
        label = ttk.Label(lr_frame, text=text)
        label.grid(column=0, row=index, pady=2, sticky="w")
    
def auto_machine_learning(): 
    aml_window = tk.Toplevel()
    aml_window.title("Auto Machine Learning")
    aml_window.resizable (False, False) 
    aml_window.attributes ('-toolwindow',-1)
    canvas = tk.Canvas(aml_window)
    aml_frame = tk.Frame(canvas)
    aml_frame.pack(side="bottom", fill="x")
    canvas.create_window((0, 0), window=aml_frame, anchor="nw")
    canvas.pack(side="left", fill="both", expand=True)
    aml_frame.grid(column=0, row=0, padx=10, pady=10)
    
    #Labels
    label_texts = [
        "Generations",
        "Population size",
        "Mutation rate",
        "Crossover rate",
        "Number of flods for k-fold cross-validation",
        "Maximum time in mins for the evaluation",
        "Maximum time in mins for each evaluation",
        "Number of generations without improvement",
        "Scoring"
    ]
    for index, text in enumerate(label_texts):
        label = ttk.Label(aml_frame, text=text)
        label.grid(column=0, row=index, pady=2, sticky="w")
    
    #Entry
    entry_var_1 = create_entry(aml_frame, "Generations", 0, "5")
    entry_var_2 = create_entry(aml_frame, "Population size", 1, "20")
    entry_var_3 = create_entry(aml_frame, "Mutation rate", 2, "0.9")
    entry_var_4 = create_entry(aml_frame, "Crossover rate", 3, "0.1")
    entry_var_5 = create_entry(aml_frame, "Number of folds for k-fold cross-validation", 4, "2")
    entry_var_6 = create_entry(aml_frame, "Maximum time in mins for the evaluation", 5, "60")
    entry_var_7 = create_entry(aml_frame, "Maximum time in mins for each evaluation", 6, "10")
    entry_var_8 = create_entry(aml_frame, "Number of generations without improvement", 7, "2")
    
    #Combobox
    combo9=create_combobox(aml_frame, scoring, 8)
    
    def close_window ():
        aml_window.destroy ()
        
    ok_button= ttk.Button (aml_frame, text="OK", command= close_window, width=10)
    ok_button.grid(row=9,column=1,sticky="w",padx=100)
    
    return {
            'entry_var_1': entry_var_1,
            'entry_var_2': entry_var_2,
            'entry_var_3': entry_var_3,
            'entry_var_4': entry_var_4,
            'entry_var_5': entry_var_5,
            'entry_var_6': entry_var_6,
            'entry_var_7': entry_var_7,
            'entry_var_8': entry_var_8,
            'combo9': combo9
            }
    

def show_features_window ():
    CC= pycc.GetInstance()
    entities= CC.getSelectedEntities()[0]
    
    training_pc_name=combo1.get()
    
    index=-1
    for ii, item in enumerate (name_list):
        if item== training_pc_name:
            pc_training=entities.getChild(ii)
            break
    pcd_training=P2p_getdata(pc_training,False,True,True)
    
    values_list=[col for col in pcd_training.columns if col != 'Class']
    labels2include= ['Classification']
    
    feature_window = tk.Toplevel()
    feature_window.title("Features of the point cloud")
   
    canvas = tk.Canvas(feature_window)
    features_frame = tk.Frame(canvas)
    features_frame.pack(side="bottom", fill="x")
   
    features2include.clear()
    for value in values_list:
        checked_var = tk.BooleanVar()
        ttk.Checkbutton(features_frame, text=value, variable=checked_var, onvalue=True, offvalue=False).pack(anchor="w")
        checked_var.trace_add("write", lambda var, indx, mode, checked_var=checked_var, value=value: on_checkbox_checked(checked_var, value))
   
    # Add the scroll bar
    scrollbar = tk.Scrollbar(feature_window, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
   
    scrollbar.config(command = values_list)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
   
    canvas.create_window((0, 0), window=features_frame, anchor="nw")
            
    # Set the scroll bar while scrolling with mouse
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    canvas.bind_all("<MouseWheel>", _on_mousewheel)
   
    # Close the features window after selected
    def ok_features_window ():
        global features2include
        if not features2include:
            print ("Please, check at least one feature")
        else:
            if len(features2include) == 1:
                print("The feature " + str(features2include) + " has been included for the training")
            else:
                print("The features " + str(features2include) + " have been included for the training")
        
        feature_window.destroy ()
        
    def cancel_features_window ():
        feature_window.destroy ()
        
    ok_button_features= tk.Button (features_frame, text="OK", command= ok_features_window, width=10)
    ok_button_features.pack (side= "left")
    cancel_button_features= tk.Button (features_frame, text="Cancel", command= cancel_features_window, width=10)
    cancel_button_features.pack (side= "right")
    
    return features2include
    print(features2include)
    
def on_checkbox_checked(checked_var, value):
    if checked_var.get() and value not in features2include:
        features2include.append(value)
    elif not checked_var.get() and value in features2include:
        features2include.remove(value) 
        
def run_algorithm_1 ():
    
    CC= pycc.GetInstance()
    entities= CC.getSelectedEntities()[0]
    
    classification_pc_name=combo_pc.get()
    index=-1
    for i, item in enumerate (name_list):
        if item== classification_pc_name:
            pc_classification=entities.getChild(i)
            break  
    feature_selection_pcd=P2p_getdata(pc_classification,False,True,True)
    
    # Save the point clouds
    
    feature_selection_pcd.to_csv(os.path.join(output_directory, 'features.txt'),sep=' ',header=True,index=False)
    
    #OPTIMAL FLOW
    # Load the selection from the file
    load_selection_from_file()
    # Get the selected parameters from the Listbox
    selected_params = [selectors[i] for i in listbox.curselection()]
    
    s = ' '.join(selected_params)
    f = entry_var_c.get()
    cv = entry_var_d.get()
    
    command = processing_file_of + ' --i ' + input_file + ' --o ' + output_directory + ' --s ' + s + ' --f ' + f + ' --cv ' + cv
    # os.system(command)
    # print(command)
    
    print("The process has been finished")
    
def run_algorithm_2 ():
    
    def random_forest_command():
        random_forest_instance = random_forest()
        
        ne=random_forest_instance['entry_var_1'].get()
        c=random_forest_instance['combo2'].get()
        md=random_forest_instance['entry_var_3'].get()
        ms=random_forest_instance['entry_var_4'].get()
        mns=random_forest_instance['entry_var_5'].get()
        mwf=random_forest_instance['entry_var_6'].get()
        mf=random_forest_instance['combo7'].get()
        bt=random_forest_instance['combo8'].get()
        s=random_forest_instance['combo9'].get()
        nj_str=-1
        nj=str(nj_str)
        
        command = processing_file_rf + ' --te ' + os.path.join(output_directory, 'input_class_test.txt') + ' --tr ' + os.path.join(output_directory, 'input_class_train.txt') + ' --o ' + output_directory + ' --f ' + os.path.join(output_directory, 'features.txt') + ' --ne ' + ne + ' --c ' + c + ' --md ' + md + ' --ms ' + ms + ' --mns ' + mns + ' --mwf ' + mwf + ' --mf ' + mf + ' --bt ' + bt + ' --s ' + s + ' --nj ' + nj
        return (command)
    
    def logistic_regression_command():
        print("This algorithm still doesn't work, please, select another algorithm")
    
    
    def auto_machine_learning_command():
        
        auto_ml_instance = auto_machine_learning()
        
        ge = auto_ml_instance['entry_var_1'].get()
        ps = auto_ml_instance['entry_var_2'].get()
        mr = auto_ml_instance['entry_var_3'].get()
        cr = auto_ml_instance['entry_var_4'].get()
        cv = auto_ml_instance['entry_var_5'].get()
        mtm = auto_ml_instance['entry_var_6'].get()
        metm = auto_ml_instance['entry_var_7'].get()
        ng = auto_ml_instance['entry_var_8'].get()
        s = auto_ml_instance['combo9'].get()
        
        
        command = processing_file_tpot + ' --te ' + os.path.join(output_directory, 'input_class_test.txt') + ' --tr ' + os.path.join(output_directory, 'input_class_train.txt') + ' --o ' + output_directory + ' --f ' + os.path.join(output_directory, 'features.txt') + ' --ge ' + ge + ' --ps ' + ps + ' --mr ' + mr + ' --cr ' + cr + ' --cv ' + cv + ' --mtm ' + mtm + ' --metm ' + metm + ' --ng ' + ng + ' --s ' + s
        return (command)
        
        
        
    def execute_selected_function():
        selected_function = combo3.get()
        if selected_function == "Random Forest":
            command=random_forest_command()
        elif selected_function == "Logistic Regression":
            command=logistic_regression_command()
        elif selected_function == "Auto Machine Learning":
            command=auto_machine_learning_command()
        return command
    
    
    
    CC= pycc.GetInstance()
    entities= CC.getSelectedEntities()[0]
    
    training_pc_name=combo1.get()
    testing_pc_name=combo2.get()
    index=-1
    for ii, item in enumerate (name_list):
        if item== training_pc_name:
            pc_training=entities.getChild(ii)
            break
    for it, item in enumerate (name_list):
        if item== testing_pc_name:
            pc_testing=entities.getChild(it)
            break
    pcd_training=P2p_getdata(pc_training,False,True,True)
    pcd_testing=P2p_getdata(pc_testing,False,True,True)
    
    # Save the point clouds and the features
    # Join the list items with commas to create a comma-separated string
    comma_separated = ','.join(features2include)    
    # Write the comma-separated string to a text file
    with open(os.path.join(output_directory, 'features.txt'), 'w') as file:
        file.write(comma_separated)  
    pcd_training.to_csv(os.path.join(output_directory, 'input_class_train.txt'),sep=' ',header=True,index=False)
    pcd_testing.to_csv(os.path.join(output_directory, 'input_class_test.txt'),sep=' ',header=True,index=False)   

    
    # # RUN THE COMMAND LINE
    command=execute_selected_function()
    os.system(command)    
    
    data = pd.read_csv(os.path.join(output_directory, 'predictions.txt'), delimiter=',')  # Assumes comma-separated columns

    
    # Create the resulting point cloud
    pc_prediction = pycc.ccPointCloud(data['X'], data['Y'], data['Z'])
    pc_prediction.setName("Results_from_prediction")
    pc_prediction.addScalarField("Predictions", data['Predictions']) 
    CC.addToDB(pc_prediction)
    CC.updateUI()
    print("The process has been finished")
    
   
def run_algorithm_3 ():
    
  
    classification_pc_name=combo4.get()
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
    command = processing_file_p + ' --i ' + os.path.join(output_directory, 'predictions.txt')+ ' --o ' + output_directory +  ' --f ' + path_feature + ' --p ' + path_pkl

    # RUN THE COMMAND LINE
    os.system(command) 
    
    data = pd.read_csv(os.path.join(output_directory, 'predictions.txt'), delimiter=',')  # Assumes comma-separated columns
    # Create the resulting point cloud
    pc_prediction = pycc.ccPointCloud(data['X'], data['Y'], data['Z'])
    pc_prediction.setName("Results_from_prediction")
    pc_prediction.addScalarField("Predictions", data['Predictions']) 
    CC.addToDB(pc_prediction)
    CC.updateUI()
    print("The process has been finished")    
   
#%% INPUTS AT THE BEGINING
name_list=get_point_clouds_name()
type_data,number=get_istance()
selectors = ['kbest_f','rfe_lr','rfe_tree','rfe_rf','rfecv_tree','rfecv_rf','rfe_svm','rfecv_svm']
criterion = ["gini","entropy","log_loss"]
max_n_features = ['sqrt','log2']
bootstrap= [True, False]
scoring=["balanced_accuracy","accuracy","f1","f1_weighted","precision","precision_weighted"]
option_buttons = {}
mla_types=["Random Forest", "Logistic Regression", "Auto Machine Learning"]
features2include=[]
values_list=[]
features=[]

#%% ERROR CONTROL
if type_data=='point_cloud':
    raise RuntimeError ("Please select the folder that contains the point clouds")
if number ==0:
    raise RuntimeError ("The folder does not contain the minimum number of point clouds (1)") 

#%% GUI
# Create the main window
root = tk.Tk()
root.title ("Machine Learning Segmentation")
root.resizable (False, False)
# Remove minimize and maximize button 
root.attributes ('-toolwindow',-1)

tab_control = ttk.Notebook(root)
tab_control.pack(expand=1, fill="both")

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

tooltip = tk.Label(root, text="", relief="solid", borderwidth=1)
tooltip.place_forget()

# TAB FEATURE SELECTION

# Labels
Label_a= ttk.Label(tab1, text="Choose point cloud")
Label_a.grid(column=0, row=0, pady=2, sticky="w")
Label_b= ttk.Label(tab1, text="Selectors", cursor="question_arrow")
Label_b.grid(column=0, row=1, pady=2, sticky="w")
Label_c= ttk.Label(tab1, text="Number of features to consider")
Label_c.grid(column=0, row=2, pady=2, sticky="w")
Label_d= ttk.Label(tab1, text="Folds for cross-validation")
Label_d.grid(column=0, row=3, pady=2, sticky="w")
Label_e= ttk.Label(tab1, text="Choose output directory")
Label_e.grid(column=0, row=4, pady=2, sticky="w")


#Tooltips
help_b = "NOTE: SVM based selectors are highly sensitive to the number of features(high-dimension)\n " \
         "and training records number, i.e.rfe_svm and rfecv_svm. When features number > 50 w/ records\n " \
         "number over 50K,otherwise will result in long processing time."
create_tooltip(Label_b, help_b)
help_d = "The choice of the number of folds (cv) depends on the size of your dataset and the trade-off between computation\n " \
         "time and the reliability of the performance estimate. Common choices include 5-fold and 10-fold cross-validation."
create_tooltip(Label_d, help_d)

# Combobox
combo_pc=ttk.Combobox (tab1,values=name_list)
combo_pc.grid(column=1, row=0, sticky="e", pady=2)
combo_pc.set("Select the point cloud used for feature selection")

# Entry
entry_var_c = create_entry(tab1, "Number of features to consider", 2, "25")
entry_var_d = create_entry(tab1, "Folds for cross-validation", 3, "5")
entry_a = ttk.Entry(tab1, width=30)
entry_a.grid(row=4, column=1, sticky="e", pady=2)
entry_a.insert(0, output_directory)

# Listbox
s_var = [tk.IntVar() for _ in selectors]
s_checklist = []
listbox = tk.Listbox(tab1, selectmode=tk.MULTIPLE, height=len(selectors))
for value in selectors:
    listbox.insert(tk.END, value)
# Select first six elements
for i in range(6):
    listbox.selection_set(i)
listbox.grid(row=1, column=1, sticky="e", padx=10, pady=10)


# Button
ok_button = tk.Button(tab1, text="OK", command=on_ok_button_click,width=10)
ok_button.grid(row=1, column=2, pady=10)
run_button_a= ttk.Button (tab1, text="OK", command=run_algorithm_1, width=10)
run_button_a.grid (row=5,column=1,sticky="e",padx=100)
cancel_button_a= ttk.Button (tab1, text="Cancel", command=destroy,width=10)
cancel_button_a.grid (row=5,column=1,sticky="e")
output_button_a = ttk.Button(tab1, text="...", command= select_directory, width=10) 
output_button_a.grid(row=4,column=2,sticky="e",padx=100)

# TAB CLASSIFICATION

# Labels
Label_1= ttk.Label(tab2, text="Choose point cloud for training")
Label_1.grid(column=0, row=0, pady=2, sticky="w")
Label_2= ttk.Label(tab2, text="Choose point cloud for testing")
Label_2.grid(column=0, row=1, pady=2,sticky="w")
Label_3= ttk.Label(tab2, text="Select machine learning algorithm")
Label_3.grid(column=0, row=2, pady=2, sticky="w")
Label_4= ttk.Label(tab2, text="Select the features to include")
Label_4.grid(column=0, row=3, pady=2, sticky="w")
Label_4= ttk.Label(tab2, text="Choose output directory")
Label_4.grid(column=0, row=4, pady=2, sticky="w")

#Tooltips
help_3 = "After selecting the machine learning algorithm, please click Set-up for change parameters"
create_tooltip(Label_3, help_3)

# Combobox
combo1=ttk.Combobox (tab2,values=name_list)
combo1.grid(column=1, row=0, sticky="e", pady=2)
combo1.set("Select the point cloud used for training")
combo2=ttk.Combobox (tab2,values=name_list)
combo2.grid(column=1, row=1, sticky="e", pady=2)
combo2.set("Select the point cloud used for testing")
combo3=ttk.Combobox (tab2,values=mla_types)
combo3.grid(column=1, row=2, sticky="e", pady=2)
combo3.set("Select the machine learning algorithm")
    
# Entry
entry_b = ttk.Entry(tab2, width=30)
entry_b.grid(row=4, column=1, sticky="e", pady=2)
entry_b.insert(0, output_directory)

# Button
setup_button= ttk.Button (tab2, text="Set-up", command=lambda: execute_selected_function(), width=10)
setup_button.grid (row=2,column=2,sticky="e",padx=100)
features_button= ttk.Button (tab2, text="...", command=show_features_window, width=10)
features_button.grid (row=3,column=2,sticky="e",padx=100)

run_button_1= ttk.Button (tab2, text="Run", command=run_algorithm_2, width=10)
run_button_1.grid (row=5,column=1,sticky="e",padx=100)
cancel_button_1= ttk.Button (tab2, text="Cancel", command=destroy,width=10)
cancel_button_1.grid (row=5,column=1,sticky="e")

output_button_b = ttk.Button(tab2, text="...", command= select_directory, width=10) 
output_button_b.grid(row=4,column=2,sticky="e",padx=100)

# TAB PREDICTION

# Labels
Label_p1= ttk.Label(tab3, text="Choose point cloud for prediction")
Label_p1.grid(column=0, row=0, pady=2, sticky="w")
Label_p2= ttk.Label(tab3, text="Load feature file")
Label_p2.grid(column=0, row=1, pady=2, sticky="w")
Label_p3= ttk.Label(tab3, text="Load pkl file")
Label_p3.grid(column=0, row=2, pady=2, sticky="w")
Label_p4= ttk.Label(tab3, text="Choose output directory")
Label_p4.grid(column=0, row=3, pady=2, sticky="w")

# Combobox
combo4=ttk.Combobox (tab3,values=name_list)
combo4.grid(column=1, row=0, sticky="e", pady=2)
combo4.set("Select the point cloud used for prediction")

# Entry
entry_c = ttk.Entry(tab3, width=30)
entry_c.grid(row=3, column=1, sticky="e", pady=2)
entry_c.insert(0, output_directory)

# Button
path_pkl = None  # Initializing path_pkl variable
load_button_2= ttk.Button (tab3, text="...", command=open_file_pkl, width=10)
load_button_2.grid (row=2,column=2,sticky="w",padx=100)

path_feature = None  # Initializing path_pkl variable
load_button_3= ttk.Button (tab3, text="...", command=open_file_feature, width=10)
load_button_3.grid (row=1,column=2,sticky="w",padx=100)

run_button_p= ttk.Button (tab3, text="OK", command=run_algorithm_3, width=10)
run_button_p.grid (row=4,column=1,sticky="e",padx=100)
cancel_button_p= ttk.Button (tab3, text="Cancel", command=destroy,width=10)
cancel_button_p.grid (row=4,column=1,sticky="e")
output_button_c = ttk.Button(tab3, text="...", command=select_directory, width=10) 
output_button_c.grid(row=3,column=2,sticky="e",padx=100)


root.mainloop()
