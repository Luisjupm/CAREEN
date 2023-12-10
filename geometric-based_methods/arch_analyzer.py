# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:57:56 2023

@author: Luisja
"""

#%% LIBRARIES
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import os
import sys

import cccorelib
import pycc

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS

script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'

sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance,extract_longitudinal_axis, minBoundingRect, extract_points_within_tolerance
from ransac import RANSAC


#%% FUNCTIONS
def run_algorithm():
    ## STORE THE INPUT VARIABLES
    Tolerance=float(entry_tolerance.get())
    num_iter_ransac=int(entry_iterations_ransac.get())
    threhold_ransac=float(entry_threshold_ransac.get())    
    type_data, number = get_istance()
    d_min=int(entry_minimum_samples.get())
    if checkbox2_var.get(): 
        if entry_percent_fix.get()=='':
            raise RuntimeError("Please introduce a value for the percent of points to consider. Example: 10 for 10% of the total points")
        else:
            percent=float(entry_percent_fix.get())  
    if type_data=='point_cloud':
        raise RuntimeError("Please select the folder that contains the point clouds")          
    ## EXTRACT THE NUMBER OF CLOUDS IN THE SELECTED FOLDER
    CC = pycc.GetInstance() 
    if number==0:
        raise RuntimeError("There are not entities in the folder")
    else:
        entities = CC.getSelectedEntities()[0]
        number = entities.getChildrenNumber()
    ## LOOP OVER EACH ELEMENT AND PERFORM THE RANSAC
    for i in range(number):
        # Get the point cloud selected as a pandas frame

        pc = entities.getChild(i)
        pcd=P2p_getdata(pc,False,False,True)
        pcd_f,skeleton =extract_points_within_tolerance(pcd[['X','Y','Z']].values, Tolerance,True)
        if checkbox2_var.get(): # In case of having fixed springs
            # Calculate the threshold because we chosen the fix springs option
            difference_height=(pcd_f[:,2].max()-pcd_f[:,2].min())*(percent/100)
            threshold_height = difference_height + (pcd_f[:,2].min())
            # Filter the DataFrame based on the condition
            filtered_pcd_f = pcd_f[pcd_f[:,2] < threshold_height]
            if len (filtered_pcd_f)<6 or len (filtered_pcd_f)<d_min:
               raise RuntimeError("The treshold is too restrictive. At least one of the arches has less than 6 points or less than the number of minimum points for fitting the model. Please increse the percentage value") 
            # Create a istance of RANSAC depending on the type of arch (combo_type.get())
            if combo_type.get()=="Pointed arch":                
                midpoint = sum(pcd_f[:,0]) / len(pcd_f[:,0])
            ransac = RANSAC(filtered_pcd_f[:,0],filtered_pcd_f[:,2],num_iter_ransac,d_min,threhold_ransac,combo_type.get(),midpoint)  
        else: # In case of not having fixed springs

            # Create a istance of RANSAC depending on the type of arch (combo_type.get())
            ransac = RANSAC(pcd_f[:,0],pcd_f[:,2],num_iter_ransac,d_min,threhold_ransac,combo_type.get())  
        # # execute ransac algorithm
        _,outliers,inliers=ransac.execute_ransac() 
        if checkbox2_var.get() and combo_type.get()=="Circular arch" or combo_type.get()=="Quarter arch": # In case of having fixed springs
            # initialize the inliers and outliers lists
            inliers=[]
            outliers=[]
            # get best model from ransac and store the data for plotting the best fit curve
            a, b, r = ransac.best_model[0], ransac.best_model[1], ransac.best_model[2] 
            # compute the error between the whole data and the best model. This model was obtained from the reduced data
            for ii in range(len(pcd_f)):
                dis = np.sqrt((pcd_f[ii,0]-a)**2 + (pcd_f[ii,2]-b)**2)
                if dis >= r:
                    distance=dis - r
                else:
                    distance= r - dis   
                if distance > threhold_ransac:
                    outliers.append(pcd_f[ii,:])
                else:
                    inliers.append(pcd_f[ii,:]) 
            # Creating a NumPy array to be compatible with the rest of the code
            if len (inliers)>0:
                inliers_array = np.array(inliers)
                inliers = inliers_array [:, [0, 2]]
            if len (outliers)>0:
                outliers_array = np.array(outliers)
                outliers = outliers_array [:, [0, 2]]
        ## CREATE THE PLOT
        if len (inliers)>0:
            plt.scatter(inliers[:,0], inliers[:,1],color='g', label='Points consider as inliers')
        if len (outliers)>0:
            plt.scatter(outliers[:,0], outliers[:,1],color='r', label='Points consider as outliers')        
        
        # # Create a scatter plot with blue dots for the best fit curve
        plt.scatter(ransac.best_x_coordinates, ransac.best_y_coordinates, c='b', s=10, label="Estimated arch by using RANSAC")
        plt.axis('scaled')
        plt.xlabel('longitudinal direction')
        plt.ylabel('vertical direction')        
        plt.title('Section of arch_'+str(i))
        plt.legend()
        plt.grid(True)
        
        # Save the plot as a PNG file
        if combo_type.get()=="Circular arch":
            plt.savefig(save_path_textbox.get()+'/circular_arch_'+str(i)+'.png')
        elif combo_type.get()=="Pointed arch":
            plt.savefig(save_path_textbox.get()+'/pointed_arch_'+str(i)+'.png')
        elif combo_type.get()=="Quarter arch":
            plt.savefig(save_path_textbox.get()+'/quarter_arch_'+str(i)+'.png')            
       
        # Clear the plot for the next iteration
        plt.clf()        
 
        npc=pc.clone()
        npc.setName('Arc_'+str(i))
        CC.addToDB(npc)
        if checkbox1_var.get():    
            npc_ske=pycc.ccPointCloud(skeleton[:,0],skeleton[:,1],skeleton[:,2])
            npc_ske.setName('Skeleton_of_Arch_'+str(i))
            CC.addToDB(npc_ske)    
    print('The process has been completed!')  
    window.destroy()  # Close the window       
def destroy():
    window.destroy()  # Close the window    
def select_path():
    # Abrir el diálogo para seleccionar la ruta de guardado
    path = filedialog.askdirectory()
    
    # Mostrar la ruta seleccionada en el textbox correspondiente
    save_path_textbox.delete(0, tk.END)
    save_path_textbox.insert(0, path)
def toggle_entry_state():
    if checkbox2_var.get():
        label_percent_fix.config(state="normal")  # Make the entry editable
        entry_percent_fix.config(state="normal")
    else:
        label_percent_fix.config(state="disabled")  # Make the entry read-only    
        entry_percent_fix.config(state="disabled")
#%% GUI
# Create the main window
window = tk.Tk()

window.title("Arch analyzer")
# Disable resizing the window
window.resizable(False, False)
# Remove minimize and maximize buttons (title bar only shows close button)
window.attributes('-toolwindow', 1)

# Create a frame for the form
form_frame = tk.Frame(window, padx=10, pady=10)
form_frame.pack()

# Labels
label_tolerance = tk.Label(form_frame, text="Thickness threshold:")
label_tolerance.grid(row=0, column=0, sticky="w",pady=2)

label_degree=tk.Label(form_frame, text="Type of arch:")
label_degree.grid(row=1, column=0, sticky="w",pady=2)

label_threshold_ransac=tk.Label(form_frame, text="Threshold value for RANSAC fitting:")
label_threshold_ransac.grid(row=2, column=0, sticky="w",pady=2)

label_iterations_ransac=tk.Label(form_frame, text="Number of iteration for RANSAC fitting:")
label_iterations_ransac.grid(row=3, column=0, sticky="w",pady=2)

label_min_samples=tk.Label(form_frame, text="Minimum number of samples for fitting the model:")
label_min_samples.grid(row=4, column=0, sticky="w",pady=2)

label_fix=tk.Label(form_frame, text="Fixed springing line:")
label_fix.grid(row=5, column=0, sticky="w",pady=2)

label_percent_fix=tk.Label(form_frame, text="Percentage of points to fit the curve from the springs:", state="disabled")
label_percent_fix.grid(row=6, column=0, sticky="w",pady=2)

checkbox1_label = tk.Label(form_frame, text="Load the points of the main section")
checkbox1_label.grid (row=7, column=0, sticky="w",pady=2)

save_path_label = tk.Label(form_frame, text="Path for saving the data:")
save_path_label.grid(row=8, column=0, sticky="w",pady=2)

# Checkboxes

# Variables de control para las opciones
checkbox1_var = tk.BooleanVar()
checkbox_1 = tk.Checkbutton(form_frame, variable=checkbox1_var)
checkbox_1.grid (row=7, column=1, sticky="e",pady=2)

checkbox2_var = tk.BooleanVar()
checkbox_2 = tk.Checkbutton(form_frame, variable=checkbox2_var, command=toggle_entry_state)
checkbox_2.grid (row=5, column=1, sticky="e",pady=2)


# Combox
algorithms = ["Circular arch","Pointed arch","Quarter arch"]
combo_type = ttk.Combobox(form_frame, values=algorithms, state="readonly")
combo_type.current(0)
combo_type.grid(row=1, column=1, sticky="e",pady=2)




# Entries
entry_tolerance = tk.Entry(form_frame,width=5)
entry_tolerance.insert(0,0.02)
entry_tolerance.grid(row=0, column=1, sticky="e",pady=2)

entry_threshold_ransac = tk.Entry(form_frame,width=5)
entry_threshold_ransac.insert(0,0.05)
entry_threshold_ransac.grid(row=2, column=1, sticky="e",pady=2)

entry_iterations_ransac = tk.Entry(form_frame,width=5)
entry_iterations_ransac.insert(0,5000)
entry_iterations_ransac.grid(row=3, column=1, sticky="e",pady=2)

entry_minimum_samples = tk.Entry(form_frame,width=5)
entry_minimum_samples.insert(0,100)
entry_minimum_samples.grid(row=4, column=1, sticky="e",pady=2)

entry_percent_fix = tk.Entry(form_frame,width=5, state="disabled")
entry_percent_fix.insert(0,10)
entry_percent_fix.grid(row=6, column=1, sticky="e",pady=2)

save_path_textbox = tk.Entry(form_frame,width=30)
save_path_textbox.grid(row=8,column=1, sticky="e",pady=2)

save_path_button = tk.Button(form_frame, text="...", command=select_path,width=2)
save_path_button.grid(row=8, column=2, sticky="e",pady=2)


# Buttons
run_button = tk.Button(form_frame, text="OK", command=run_algorithm,width=10)
cancel_button = tk.Button(form_frame, text="Cancel", command=destroy,width=10)
run_button.grid(row=9, column=1, sticky="e",padx=100)
cancel_button.grid(row=9, column=1, sticky="e")


#%% START THE GUI
# Start the main event loop
window.mainloop()


