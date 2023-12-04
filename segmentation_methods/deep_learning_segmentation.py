# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 12:54:24 2023

@author: Digi_2
"""

#%% LIBRARIES
import os
import subprocess
import sys

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import pandas as pd
import numpy as np
import pickle

import cccorelib
import pycc

#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS
script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'
print (additional_modules_directory)
sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance,get_point_clouds_name

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\conda_env\segmentation_methods\point_transformer'
print (additional_modules_directory)
sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance,get_point_clouds_name

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\temp_folder_for_results\OUTPUT_training\feature_file.txt'
print (additional_modules_directory)
sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance,get_point_clouds_name

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\temp_folder_for_results\OUTPUT_classification\cloud_classified.txt'
print (additional_modules_directory)
sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance,get_point_clouds_name


#%% DEFINING INPUTS OF CMD
current_directory=os.path.dirname(os.path.abspath(__file__))
directory_bat=os.path.join(current_directory,'..','conda_env','segmentation_methods','point_transformer')
activate_script = os.path.join('env', 'torch_env_38', 'Scripts', 'activate.bat')

# Training and testing
train_file=os.path.join(current_directory,'..','temp_folder_for_results','Deep_Learning','INPUT_training','input_pt_train.txt')
test_file=os.path.join(current_directory,'..','temp_folder_for_results','Deep_Learning','INPUT_training','input_pt_test.txt')
output_directory=os.path.join(current_directory,'..','temp_folder_for_results','Deep_Learning','OUTPUT_training')
input_features= os.path.join(output_directory,'feature_file.txt')
output_log_file=os.path.join(output_directory, 'output.log')
output_file_features= os.path.join(output_directory, 'feature_file.txt')
output_training =  os.path.join(current_directory,'..','conda_env','segmentation_methods','point_transformer','training.py ')

# Classification
classified_model_file=os.path.join(current_directory,'..','temp_folder_for_results','Deep_Learning','OUTPUT_training','model_pt_class.pt')
input_file=os.path.join(current_directory,'..','temp_folder_for_results','Deep_Learning','INPUT_classification','input_pt_class.txt')
input_features_class= os.path.join(current_directory,'..','temp_folder_for_results','Deep_Learning','OUTPUT_training','feature_file.txt')
output_directory_class=os.path.join(current_directory,'..','temp_folder_for_results','Deep_Learning','OUTPUT_classification')
output_class_file=os.path.join(output_directory, 'cloud_classified.txt')
output_log_file_class=os.path.join(output_directory, 'output.log')
output_inference = os.path.join(current_directory,'..','conda_env','segmentation_methods','point_transformer','inference.py')


#%% FUNCTIONS TRAINING

def destroy ():
    root.destroy ()

def choose_output_directory():
    output_directory = filedialog.askdirectory()
    if output_directory:
        print("Output directory selected:", output_directory)

def select_directory(entry_widgets):
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

def create_entry_button_output(tab, row):
    entry_widget = ttk.Entry(tab, width=30)
    entry_widget.grid(row=row, column=1, sticky="e", pady=2)
    entry_widget.insert(0, output_directory)

    button_widget = ttk.Button(tab, text="...", command=lambda: select_directory([entry_widget]), width=10)
    button_widget.grid(row=row, column=2, sticky="e", padx=100)
    return entry_widget

def create_tooltip(widget, text):
    widget.bind("<Enter>", lambda event: show_tooltip(text))
    widget.bind("<Leave>", hide_tooltip)

def show_tooltip(text):
    tooltip.config(text=text)
    tooltip.place(relx=0.5, rely=0.5, anchor="center", bordermode="outside")
    
def hide_tooltip(event):
    tooltip.place_forget()
    
# def show_features_window ():
#     CC= pycc.GetInstance()
#     entities= CC.getSelectedEntities()[0]
    
#     training_pc_name=combo_training.get()
    
#     index=-1
#     for ii, item in enumerate (name_list):
#         if item== training_pc_name:
#             pc_training=entities.getChild(ii)
#             break
#     pcd_training=P2p_getdata(pc_training,False,True,True)
    
#     values_list=[col for col in pcd_training.columns if col != 'Class']
#     labels2include= ['Classification']
    
#     feature_window = tk.Toplevel()
#     feature_window.title("Features of the point cloud")
   
#     canvas = tk.Canvas(feature_window)
#     features_frame = tk.Frame(canvas)
#     features_frame.pack(side="bottom", fill="x")
   
#     features2include.clear()
#     for value in values_list:
#         checked_var = tk.BooleanVar()
#         ttk.Checkbutton(features_frame, text=value, variable=checked_var, onvalue=True, offvalue=False).pack(anchor="w")
#         checked_var.trace_add("write", lambda var, indx, mode, checked_var=checked_var, value=value: on_checkbox_checked(checked_var, value))
   
#     # Add the scroll bar
#     scrollbar = tk.Scrollbar(feature_window, orient="vertical", command=canvas.yview)
#     canvas.configure(yscrollcommand=scrollbar.set)
   
#     scrollbar.config(command = values_list)
#     scrollbar.pack(side="right", fill="y")
#     canvas.pack(side="left", fill="both", expand=True)
   
#     canvas.create_window((0, 0), window=features_frame, anchor="nw")
            
#     # Set the scroll bar while scrolling with mouse
#     def _on_mousewheel(event):
#         canvas.yview_scroll(int(-1*(event.delta/120)), "units")

#     canvas.bind_all("<MouseWheel>", _on_mousewheel)
   
#     # Close the features window after selected
#     def cancel_features_window ():
#         feature_window.destroy ()
        
#     def ok_features_window ():
#         if not features2include:
#             print ("Please, check at least one feature")
#         else:
#             features_names = ', '.join(features2include)
#             if len(features2include) == 1:
#                 print("The feature " + str(features2include) + " has been included for the training")
#             else:
#                 print("The features " + str(features2include) + " have been included for the training")
        
#             # Make list with the numeric features for P.T. classification
#             pcd_training=P2p_getdata(pc_training,False,True,True)
#             values_list_1=[col for col in pcd_training.columns if col != 'Class']
#             df = pd.DataFrame(features2include, columns=["Columns"])
#             values_list = df["Columns"].tolist()
#             selected_indices = [str(values_list_1.index(column)) for column in features2include if column in values_list]
#             indices_as_string = ",".join(selected_indices)
#             with open(output_file_features, "w") as output_file:
#                 output_file.write(indices_as_string)

#         feature_window.destroy ()
      
#     ok_button_features= ttk.Button (features_frame, text="OK", command= ok_features_window, width=10)
#     ok_button_features.pack (side= "right")
#     cancel_button_features= ttk.Button (features_frame, text="Cancel", command= cancel_features_window, width=10)
#     cancel_button_features.pack (side= "right") 
    
# Select all the features in the features window
def select_all_checkbuttons(checkbuttons):
    for widget in checkbuttons:
        if isinstance(widget, ttk.Checkbutton):
            widget.state(['!alternate'])
            widget.state(['selected'])
        
def show_features_window ():
    global features2include
    
    CC= pycc.GetInstance()
    entities= CC.getSelectedEntities()[0]
    
    training_pc_name=combo_training.get()
    
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
    
    checkbutton_frame = tk.Frame(features_frame)
    checkbutton_frame.pack(side="left", fill="y")
   
    features2include.clear()
    for value in values_list:
        checked_var = tk.BooleanVar()
        ttk.Checkbutton(checkbutton_frame, text=value, variable=checked_var, onvalue=True, offvalue=False).pack(anchor="w")
        checked_var.trace_add("write", lambda var, indx, mode, checked_var=checked_var, value=value: on_checkbox_checked(checked_var, value))
   
    # Add the scroll bar
    scrollbar = tk.Scrollbar(checkbutton_frame, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=scrollbar.set)
   
    scrollbar.config(command = values_list)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
   
    canvas.create_window((0, 0), window=features_frame, anchor="nw")
    
    select_all_button = ttk.Button(features_frame, text="Select All", command=lambda: select_all_checkbuttons(checkbutton_frame.winfo_children()))
    select_all_button.pack(side="top", pady=5)
            
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
                
            # Make list with the numeric features for P.T. classification
            pcd_training=P2p_getdata(pc_training,False,True,True)
            values_list_1=[col for col in pcd_training.columns if col != 'Class']
            df = pd.DataFrame(features2include, columns=["Columns"])
            values_list = df["Columns"].tolist()
            selected_indices = [str(values_list_1.index(column)) for column in features2include if column in values_list]
            indices_as_string = ",".join(selected_indices)
            with open(output_file_features, "w") as output_file:
                output_file.write(indices_as_string)

        feature_window.destroy ()
        
    def cancel_features_window ():
        feature_window.destroy ()
    
    button_frame = tk.Frame(features_frame)
    button_frame.pack(side="right", fill="y")    
    ok_button_features= ttk.Button (features_frame, text="OK", command= ok_features_window, width=10)
    ok_button_features.pack (side= "left")
    cancel_button_features= ttk.Button (features_frame, text="Cancel", command= cancel_features_window, width=10)
    cancel_button_features.pack (side= "right")
    
    return features2include
    print(features2include)
    

def on_checkbox_checked(checked_var, value):
    if checked_var.get() and value not in features2include:
        features2include.append(value)
    elif not checked_var.get() and value in features2include:
        features2include.remove(value)     
            
        
def run_algorithm ():
    
    CC= pycc.GetInstance()
    entities= CC.getSelectedEntities()[0]
    
    training_pc_name=combo_training.get()
    testing_pc_name=combo_testing.get()
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
    
    # Save the point clouds
    pcd_training.to_csv(train_file,sep=' ',header=True,index=False)
    pcd_testing.to_csv(test_file,sep=' ',header=True,index=False)
    
    # Selected features to command
    try:
        with open(input_features, "r") as file:
            content=file.read()
    except FileNotFoundError:
        print("The file was not found or the path is incorrect.")
    features= content
    gpu=entry_gpu.get()
    epoch=entry_iterations.get()
    
    values_list_3=[col for col in pcd_training.columns if col != 'Classification']
    class_numeric = None
    for index, column in enumerate(values_list_3):
        if column == "Class":
            class_numeric = index
            break
    classification= class_numeric


#%% COMMAND TRAINING
    os.chdir(directory_bat)
    command = f'{directory_bat}/env/torch_env_38/Scripts/activate.bat && python -u "{output_training}" --train "{train_file}" --test "{test_file}" --features {features} --labels {classification} --size {gpu} --epoch {epoch} --output "{output_directory}" > "{output_log_file}"'
    os.system(command)
    
    print("The process has been finished")
    
#%% FUNCTIONS CLASSIFICATION

# Open directory for selecting trained model   
def show_window ():
    extension= [("Pytorch files","*.pt")]
    selected_folder= filedialog.askopenfilename(filetypes= extension)  
    
# Open directory for selecting the training features   
def show_features_window_2 ():
    extension= [("Text files","*.txt")]
    selected_folder= filedialog.askopenfilename(filetypes= extension)  
        
def run_algorithm_2 ():
    
    CC= pycc.GetInstance()
    entities= CC.getSelectedEntities()[0]
    
    classification_pc_name=combo_classification.get()
    index=-1
    for i, item in enumerate (name_list):
        if item== classification_pc_name:
            pc_classification=entities.getChild(i)
            break  
    pcd_classification=P2p_getdata(pc_classification,False,True,True)
    
    # Save the point clouds
    pcd_classification.to_csv(input_file,sep=' ',header=True,index=False)
    
    # Selected features to command
    try:
        with open(input_features, "r") as file:
            content=file.read()
    except FileNotFoundError:
        print("The file was not found or the path is incorrect.")
    features= content
    
    gpu = entry_gpu.get()
    
    
#%% COMMAND CLASSIFICATION
    os.chdir(directory_bat)
    command = f'{directory_bat}/env/torch_env_38/Scripts/activate.bat && python -u "{output_inference}" --model "{classified_model_file}" --features {features} --input "{input_file}" --size {gpu} --output "{output_class_file}" > "{output_log_file_class}"'
    os.system(command)
    
    print("The process has been finished")

#%% INPUTS AT THE BEGINING
name_list=get_point_clouds_name()
type_data,number=get_istance()
features2include=[]
values_list=[]
features=[]

    
#%% ERROR CONTROL
if type_data=='point_cloud':
    raise RuntimeError ("Please select the folder that contains the point clouds")
if number== 0:
    raise RuntimeError ("The folder does not contain the minimum number of point clouds (1)") 
    
#%% GUI
# Create the main window
root = tk.Tk()
root.title ("Deep Learning Segmentation")
root.resizable (False, False)
# Remove minimize and maximize button 
root.attributes ('-toolwindow',-1)

tabControl = ttk.Notebook(root)

tab1 = tk.Frame(tabControl,padx=10,pady=10)
tab1.pack()

tab2 = tk.Frame(tabControl,padx=10,pady=10)
tab2.pack()

tabControl.add(tab1, text='Training')
tabControl.add(tab2, text='Classification')
tabControl.pack(expand=1, fill="both")

tooltip = tk.Label(root, text="", relief="solid", borderwidth=1)
tooltip.place_forget()

#%% GUI TRAINING

# Labels
label_1=ttk.Label (tab1, text="Point cloud for training")
label_1.grid (row=0,column=0,sticky="w",pady=2)
label_2=ttk.Label (tab1, text="Point cloud for testing")
label_2.grid (row=1,column=0,sticky="w",pady=2)
label_3=ttk.Label (tab1, text="Choose the features to include")
label_3.grid (row=2, column=0, sticky="w",pady=2)
label_4=ttk.Label (tab1, text="GPU comsumption")
label_4.grid (row=3,column=0,sticky="w",pady=2)
label_5=ttk.Label (tab1, text="Number of iterations")
label_5.grid (row=4,column=0,sticky="w",pady=2)
label_6=ttk.Label (tab1, text="Choose output directory").grid (row=5,column=0,sticky="w",pady=2)

# Combobox
combo_training=ttk.Combobox (tab1,values=name_list)
combo_training.grid (row=0,column=1,sticky="w",pady=2)
combo_training.set("Select the point cloud used for training")

combo_testing=ttk.Combobox (tab1,values=name_list)
combo_testing.grid (row=1,column=1,sticky="w",pady=2)
combo_testing.set("Select the point cloud used for testing")

# Entry
entry_gpu=ttk.Entry (tab1,width=5)
entry_gpu.insert (0,20000)
entry_gpu.grid (row=3,column=1,sticky="e",pady=2) 

entry_iterations=ttk.Entry (tab1,width=5)
entry_iterations.insert (0,100)
entry_iterations.grid (row=4,column=1,sticky="e",pady=2)

entry_a = create_entry_button_output(tab1, 5) 

# Button
run_button= ttk.Button (tab1, text="OK", command=run_algorithm, width=10)
cancel_button= ttk.Button (tab1, text="Cancel", command=destroy,width=10)
run_button.grid (row=6,column=1,sticky="e",padx=100) 
cancel_button.grid (row=6,column=1,sticky="e") 

button_features= ttk.Button (tab1, text="...", command=show_features_window, width=10)
button_features.grid(row=2,column=1,sticky="e",pady=2)


#%% GUI CLASSIFICATION

# Create the secondary window
window_2= tk.Tk()
window_2.title ("Features to include")
window_2.resizable (False, False)
window_2.attributes ('-toolwindow',-1)

form_frame_2= tk.Frame (window_2, padx=4,pady=10)
form_frame_2.pack()

# Labels
label_1=ttk.Label (tab2, text="Trained model")
label_1.grid (row=0,column=0,sticky="w",pady=2)
label_2=ttk.Label (tab2, text="Point cloud for classify")
label_2.grid (row=1,column=0,sticky="w",pady=2)
label_3=ttk.Label (tab2, text="GPU comsuption")
label_3.grid (row=2, column=0, sticky="w",pady=2)
label_4=ttk.Label (tab2, text="Choose output directory")
label_4.grid (row=3, column=0, sticky="w",pady=2)

# Combobox
combo_classification=ttk.Combobox (tab2,values=name_list)
combo_classification.grid (row=1,column=1,sticky="e",pady=2)
combo_classification.set("Select the point cloud used for classification")

# Entry
entry_gpu=ttk.Entry (tab2,width=5)
entry_gpu.insert (0,20000)
entry_gpu.grid (row=2,column=1,sticky="e",pady=2) 

entry_b = create_entry_button_output(tab2, 3) 

# Button
run_button= ttk.Button (tab2, text="OK", command=run_algorithm_2, width=10)
cancel_button= ttk.Button (tab2, text="Cancel", command=destroy,width=10)
run_button.grid (row=4,column=1,sticky="e",padx=100) 
cancel_button.grid (row=4,column=1,sticky="e")

button_trained_model= ttk.Button (tab2, text="...", command=show_window, width=10)
button_trained_model.grid(row=0,column=1,sticky="e",pady=2)


# Start the main event loop
root.mainloop()