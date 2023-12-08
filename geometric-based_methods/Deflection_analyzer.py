# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 18:00:57 2023

@author: Luisja
"""

#%% LIBRARIES
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import cccorelib
import pycc

import pandas as pd
import numpy as np

from scipy.spatial import ConvexHull
from scipy.spatial import distance
from scipy.optimize import fsolve

from itertools import combinations
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import os
import sys

#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS

script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'
sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance,extract_longitudinal_axis, minBoundingRect, extract_points_within_tolerance

#%% FUNCTIONS
def select_path():
    # Abrir el diálogo para seleccionar la ruta de guardado
    path = filedialog.askdirectory()
    
    # Mostrar la ruta seleccionada en el textbox correspondiente
    save_path_textbox.delete(0, tk.END)
    save_path_textbox.insert(0, path)
def run_algorithm():
    ## STORE THE INPUT VARIABLES
    Tolerance=float(entry_tolerance.get())
    Relative_threshold=int(entry_relative_deflection.get())
    Degree=int(entry_degree.get())
    cal_type=str(combo_type.get())
    
    type_data, number = get_istance()
    if type_data=='point_cloud':
        raise RuntimeError("Please select the folder that contains the point clouds")          
    ## EXTRACT THE NUMBER OF CLOUDS IN THE SELECTED FOLDER
    CC = pycc.GetInstance() 
    if number==0:
        raise RuntimeError("There are not entities in the folder")
    else:
        entities = CC.getSelectedEntities()[0]
        number = entities.getChildrenNumber()

    ## CREATE A EMPTY VARIABLE FOR STORING RESULTS
        data = []
    ## LOOP OVER EACH ELEMENT
        for i in range(number):
            # Get the point cloud selected as a pandas frame
            pc = entities.getChild(i)
            pcd=P2p_getdata(pc,False,True,True)
            pcd_f,skeleton =extract_points_within_tolerance(pcd[['X','Y','Z']].values, Tolerance,True)
            
            # FIT TO A POLINOMIAL CURVE
            coefficients = np.polyfit(pcd_f[:,0], pcd_f[:,2], Degree)
            curve = np.poly1d(coefficients)
            # Find the inflection point (second derivative equal to 0)
            second_derivative= np.polyder (curve,2)
            second_derivative_roots=np.roots(second_derivative)
            filter_arr_2 = []
            
            for element in second_derivative_roots:
              if element>min(pcd_f[:,0]) and element<max(pcd_f[:,0]):
                filter_arr_2.append(True)
              else:
                filter_arr_2.append(False)
            second_derivative_roots_filtered = second_derivative_roots[filter_arr_2]
            z_second_derivative_roots_filtered = [curve(x) for x in second_derivative_roots_filtered]
            
            
            ## PLOTTING
            # Generate points on the curve for plotting
            x_curve = np.linspace(pcd_f[:,0].min(), pcd_f[:,0].max(), 100)
            z_curve = curve(x_curve)
            # Find the corresponding x values for the maximum and minimum z points
            x_max_z_data = pcd_f[:,0][np.argmax(pcd_f[:,2])]
            x_min_z_data = pcd_f[:,0][np.argmin(pcd_f[:,2])]
            x_max_z_fit = x_curve[np.argmax(z_curve)]
            x_min_z_fit = x_curve[np.argmin(z_curve)]
            
            # Find the maximum and minimum z values in the original data
            max_z_data = np.max(pcd_f[:,2])
            min_z_data = np.min(pcd_f[:,2])
            
            # Find the maximum and minimum z values from the polynomial fitting
            max_z_fit = np.max(z_curve)
            min_z_fit = np.min(z_curve)
            # Calculate the distances along the x-axis
            x_distance = pcd_f[:,0].max() - pcd_f[:,0].min()
            
            # Calculate the distances between the maximum and minimum z points
            z_distance_data = max_z_data - min_z_data
            z_distance_fit = max_z_fit - min_z_fit
            
            #Calculate the relative deflection
            Relative_data=z_distance_data/x_distance
            Maximum_deflection=x_distance/Relative_threshold
            Relative_fit=z_distance_fit/x_distance
            # Create the plot
            plt.scatter(pcd_f[:,0], pcd_f[:,2], label='Data Points')
            plt.plot(x_curve, z_curve, 'r', label='Polynomial Curve')


            plt.scatter(x_min_z_data, min_z_data, color='green', marker='o', label='Min Z (Data)')
            plt.scatter(x_max_z_data, max_z_data, color='blue', marker='o', label='Max Z (Data)')
            plt.scatter(x_min_z_fit, min_z_fit, color='yellow', marker='o', label='Min Z (Fit)')
            plt.scatter(x_max_z_fit, max_z_fit, color='purple', marker='o', label='Max Z (Fit)')
            plt.scatter(second_derivative_roots_filtered, z_second_derivative_roots_filtered, color='red', marker='o', label='Inflection point')
            
            plt.xlabel('longitudinal direction')
            plt.ylabel('vertical direction')
            plt.title('Deflection analysis of Beam_'+str(i))
            plt.legend()
            plt.grid(True)
            # Save the plot as a PNG file
            plt.savefig(save_path_textbox.get()+'/Beam_'+str(i)+'.png')
            # Clear the plot for the next iteration
            plt.clf()

            if cal_type=='Data':
                if z_distance_data<=Maximum_deflection:
                    arr = np.full((len(pcd),), 0)
                    verified= True
                    arr_1=np.full((len(pcd),), z_distance_data)
                    arr_2=np.full((len(pcd),), Relative_data)
                else:
                    arr = np.full((len(pcd),), 1)
                    verified= False
                    arr_1=np.full((len(pcd),), z_distance_data)
                    arr_2=np.full((len(pcd),), Relative_data)
            else:
                if z_distance_fit<=Maximum_deflection:
                    arr = np.full((len(pcd),), 0)
                    verified= True
                    arr_1=np.full((len(pcd),), z_distance_fit)
                    arr_2=np.full((len(pcd),), Relative_fit)
                else:
                    arr = np.full((len(pcd),), 1)
                    verified= False
                    arr_1=np.full((len(pcd),), z_distance_fit)
                    arr_2=np.full((len(pcd),), Relative_fit)
             # Store the data as a tuple
            data.append(('Beam_'+str(i),x_distance, z_distance_data, z_distance_fit, x_min_z_data-min(pcd_f[:,0]), x_min_z_fit-min(pcd_f[:,0]),z_second_derivative_roots_filtered,second_derivative_roots_filtered,Relative_data,Relative_fit,verified))
            npc=pc.clone()
            npc.setName('Beam_'+str(i))
            CC.addToDB(npc)
            npc.addScalarField("Is deflected", arr)       
            npc.addScalarField("Relative_deflection", arr_2)       
            npc.addScalarField("Maximum deflection", arr_1)
            
            if checkbox1_var.get():    
                npc_ske=pycc.ccPointCloud(skeleton[:,0],skeleton[:,1],skeleton[:,2])
                npc_ske.setName('Skeleton_of_Beam_'+str(i))
                CC.addToDB(npc_ske)
        # Open the file in write mode
        with open(save_path_textbox.get()+'/deflection_analysis.txt', 'w') as file:
        # Write the header
            file.write("Identifier\tLength\tDeflection from point data\tDeflection from polynomial data\tDistance to maximum deflection point from point data\tDistance to maximum deflection point from polynomial data\tInflection points (vertical coordinates)\tInflection points (horizontal coordinates)\tRelative deflection from point data\tRelative deflection from polynomial data\tIs within the relative deflection tolerante?\n")
            
            # Write the data to the file
            for item in data:
                file.write(f"{item[0]}\t{item[1]:.3f}\t{item[2]:.3f}\t{item[3]:.3f}\t{item[4]:.3f}\t{item[5]:.3f}\t{item[6]}\t{item[7]}\t{item[8]:.3f}\t{item[9]:.3f}\t{item[10]}\n")
        print('The process has been finished')  
    
    
    window.destroy()  # Close the window    
def destroy():
    window.destroy()  # Close the window    

#%% GUI
# Create the main window
window = tk.Tk()

window.title("Deflection analyzer")
# Disable resizing the window
window.resizable(False, False)
# Remove minimize and maximize buttons (title bar only shows close button)
window.attributes('-toolwindow', 1)

# Create a frame for the form
form_frame = tk.Frame(window, padx=10, pady=10)
form_frame.pack()

# Variables de control para las opciones
checkbox1_var = tk.BooleanVar()

# Labels
label_tolerance = tk.Label(form_frame, text="Thickness threshold:")
label_tolerance.grid(row=0, column=0, sticky="w",pady=2)

label_degree=tk.Label(form_frame, text="Polinomic degree:")
label_degree.grid(row=1, column=0, sticky="w",pady=2)

label_relative_deflection = tk.Label(form_frame, text="Maximum relative deflection (L/300; L/500):")
label_relative_deflection.grid(row=2, column=0, sticky="w",pady=2)

label_type = tk.Label(form_frame, text="Type of input for the scalar field")
label_type.grid(row=3, column=0, sticky="w",pady=2)

checkbox1_label = tk.Label(form_frame, text="Load the points of the main axis")
checkbox1_label.grid (row=4, column=0, sticky="w",pady=2)

checkbox_1 = tk.Checkbutton(form_frame, variable=checkbox1_var)
checkbox_1.grid (row=4, column=1, sticky="e",pady=2)

save_path_label = tk.Label(form_frame, text="Path for saving the data:")
save_path_label.grid(row=5, column=0, sticky="w",pady=2)


# Entries
entry_tolerance = tk.Entry(form_frame,width=5)
entry_tolerance.insert(0,0.02)
entry_tolerance.grid(row=0, column=1, sticky="e",pady=2)

entry_degree = tk.Entry(form_frame,width=5)
entry_degree.insert(0,4)
entry_degree.grid(row=1,column=1, sticky="e",pady=2)

entry_relative_deflection = tk.Entry(form_frame,width=5)
entry_relative_deflection.insert(0,300)
entry_relative_deflection.grid(row=2, column=1, sticky="e",pady=2)

save_path_textbox = tk.Entry(form_frame,width=30)
save_path_textbox.grid(row=5,column=1, sticky="e",pady=2)

# Combox
algorithms = ["Data", "Fit"]
combo_type = ttk.Combobox(form_frame, values=algorithms, state="readonly")
combo_type.current(0)
combo_type.grid(row=3, column=1, sticky="e",pady=2)


# Buttons
save_path_button = tk.Button(form_frame, text="...", command=select_path,width=2)
save_path_button.grid(row=5, column=1, sticky="e",pady=2)

run_button = tk.Button(form_frame, text="OK", command=run_algorithm,width=10)
cancel_button = tk.Button(form_frame, text="Cancel", command=destroy,width=10)
run_button.grid(row=6, column=1, sticky="e",padx=100)
cancel_button.grid(row=6, column=1, sticky="e")


#%% START THE GUI
# Start the main event loop
window.mainloop()