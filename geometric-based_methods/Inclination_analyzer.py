# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:27:04 2023

@author: Luisja
"""
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import cccorelib
import pycc
import os
import pandas as pd
import numpy as np


import scipy.spatial
from scipy.spatial import ConvexHull
from scipy import optimize
from scipy.stats import linregress
import math
import cv2
import matplotlib.pyplot as plt
def P2p_getdata (pc,nan_value=False,sc=True):
    ## CREATE A DATAFRAME WITH THE POINTS OF THE PC
   pcd = pd.DataFrame(pc.points(), columns=['X', 'Y', 'Z'])
   if (sc==True):       
   ## ADD SCALAR FIELD TO THE DATAFRAME
       for i in range(pc.getNumberOfScalarFields()):
           scalarFieldName = pc.getScalarFieldName(i)  
           scalarField = pc.getScalarField(i).asArray()[:]              
           pcd.insert(len(pcd.columns), scalarFieldName, scalarField) 
   ## DELETE NAN VALUES
   if (nan_value==True):
       pcd.dropna(inplace=True)
   return pcd
def select_path():
    # Abrir el diálogo para seleccionar la ruta de guardado
    path = filedialog.askdirectory()
    
    # Mostrar la ruta seleccionada en el textbox correspondiente
    save_path_textbox.delete(0, tk.END)
    save_path_textbox.insert(0, path)
# Calculate the area using the shoelace formula
def calculate_polygon_area(points):
    n = len(points)
    area = 0
    for i in range(n):
        x1, y1 = points.iloc[i]
        x2, y2 = points.iloc[(i + 1) % n]
        area += (x1 * y2 - x2 * y1)
    area = abs(area) / 2
    return area
def run_algorithm():
    ## STORE THE INPUT VARIABLES
    Tolerance=float(entry_tolerance.get())
    Step=float(entry_step.get())
    cal_type=str(combo_type.get())    
    limit=float (entry_maximum_inclination.get())
    
    ## EXTRACT THE NUMBER OF CLOUDS IN THE SELECTED FOLDER
    CC = pycc.GetInstance() 
    if not CC.haveSelection():
        raise RuntimeError("No folder selected")
    else:
        entities = CC.getSelectedEntities()[0]
        number = entities.getChildrenNumber()
        if hasattr(entities, 'points'):
            raise RuntimeError("Please select the folder that contains the point clouds")        
        
    if number==0:
        raise RuntimeError("There are not entities in the folder")
    else:

    ## CREATE A EMPTY VARIABLE FOR STORING RESULTS
        data = []
    ## LOOP OVER EACH ELEMENT
        for i in range(number):
            pc = entities.getChild(i)
            pcd=P2p_getdata(pc,False,True)
    ## LOOP WITHIN EACH ELEMENT FOR EXTRACTING THE SECTIONS
            j=Step
            # Create an empty list to store the data
            data_element = [] # for storing the data of each section
            while j<max (pcd['Z']):# For stoping at the maximum z
                if j>=min (pcd['Z']):# For starting at the minimum z
                    
                    upper_bound= j + Tolerance
                    lower_bound= j - Tolerance
                    section= pcd[(pcd['Z'] >= lower_bound) & (pcd['Z'] <= upper_bound)]
                    section_f=section[['X','Y']] # Deleting the Z coordinate for processing
                    if cal_type=="Points":
                        area= calculate_polygon_area(section_f[['X', 'Y']])
                        centroid_x = section_f['X'].mean()
                        centroid_y = section_f['Y'].mean()
    
                    elif cal_type=="Convex Hull":
                        hull=ConvexHull(section_f[['X', 'Y']].values)
                        # Extract the indices of the convex hull vertices
                        hull_indices = hull.vertices
                        # Calculate the area of the polygon defined by the convex hull
                        hull_points = section_f[['X', 'Y']].values[hull_indices]
                        area = ConvexHull(hull_points).area
                        centroid_x = np.mean(hull.points[:,0])
                        centroid_y = np.mean(hull.points[:,1])
                    elif cal_type=="Rectangle fitting":
                        # Fit the points to a rectangle
                        rect = cv2.minAreaRect(section_f[['X', 'Y']].values)
                        # Extract the centroid and area of the fitted rectangle
                        centroid_x, centroid_y = rect[0]

                        area = rect[1][0] * rect[1][1]
                        box=cv2.boxPoints(rect)
                    elif cal_type=="Circle fitting":
                        # Fit the points to a circle
                        (centroid_x, centroid_y), radius = cv2.minEnclosingCircle(section_f[['X', 'Y']].values)
                        
                        area = np.pi * radius**2

                    # Append the data as a dictionary to the list
                    data_element.append((j,centroid_x, centroid_y,area))
                    # WRITE AND IMAGE WITH THE CROSS SECTION
                    figure, axes = plt.subplots()
                    axes.set_aspect( 1 )
                    # Create the plot along x-axis
                    
                    plt.scatter(section_f['X'].values, section_f['Y'].values, label='Data Points')
                    # Plot the centroid point
                    plt.scatter(centroid_x, centroid_y, color='red', marker='x', label='Centroid')
                    #Plot circle in case of estimating with circle fitting
                    if cal_type =='Points':
                        pass
                    elif cal_type=='Circle fitting':
                        Circle=plt.Circle((centroid_x, centroid_y), radius, fill=False, color='green')
                        axes.add_artist(Circle)
                        plt.xlim(centroid_x-radius*2 , centroid_x+radius*2 )
                        plt.ylim(centroid_y-radius*2 ,centroid_y+radius*2 )
                    elif cal_type=='Rectangle fitting':
                        # We plot by means of the four corners since it is easier
                        # Extract x and y coordinates from the points
                        x_coords, y_coords = zip(*box)
                        x_coords = list(x_coords)  # Convert to a list
                        y_coords = list(y_coords)  # Convert to a list
                        # Plot the rectangle by connecting the points with lines
                        axes.plot(x_coords + [x_coords[0]], y_coords + [y_coords[0]], marker='o', linestyle='-', color='g')
                    elif cal_type=="Convex Hull":
                        # Plot the convex hull polygon
                        for simplex in hull.simplices:
                            axes.plot(hull.points[simplex, 0], hull.points[simplex, 1], 'g-')                        
                    plt.xlabel('x-axis')
                    plt.ylabel('y-axis')
                    plt.title('Centroid estimation of Element_'+str(i) + ' at height ' + str(j))
                    plt.legend(loc="upper right")
                    plt.grid(True)
                    # Save the plot as a PNG file
                    newdir=save_path_textbox.get()+'/Element_'+str(i)+'_sections'
                    # Check if the folder already exists
                    if not os.path.exists(newdir):
                    # If it doesn't exist, create it
                        os.mkdir(newdir)
                    plt.savefig(newdir +'/Element_'+str(i)+ ' at height ' + str(j) + '.png')
                    # Clear the plot for the next iteration
                    plt.clf()                     
                j= j + Step #Updating the Step
            # WRITE FILE WITH THE GENERAL DATA
            with open(save_path_textbox.get()+'/Element_'+str(i)+'.txt', 'w') as file:
            # Write the header
                file.write("Height\tCentroid along x-axis\tCentroid along y-axis\tArea\n")
                
                # Write the data to the file
                for item in data_element:
                    file.write(f"{item[0]}\t{item[1]:.3f}\t{item[2]:.3f}\t{item[3]:.3f}\n")
            
            
            #Best fit line from j and centroid_x
            column_0 = [row[0] for row in data_element]  # List comprehension to extract the first column
        
            column_1 = [row[1] for row in data_element]  # List comprehension to extract the first column
            min_value = min(column_1)
            # Normalize the values in column_1
            normalize_c_1 = [x - min_value for x in column_1]

            column_2 = [row[2] for row in data_element]  # List comprehension to extract the second column
            min_value = min(column_2)
            # Normalize the values in column_1
            normalize_c_2 = [x - min_value for x in column_2]
            # Along x-axis
            slope_x, intercept_x, r_value_x, p_value_x, std_err_x = linregress(normalize_c_1, column_0)
            angle_rad_x = math.atan(slope_x)  # Calculate the angle in radians
            angle_deg_x = 90-math.degrees(angle_rad_x)  # Convert the angle to degrees
            arg_1=np.full((len(pcd),), angle_deg_x)

            # Along y-axis
            slope_y, intercept_y, r_value_y, p_value_y, std_err_y = linregress(normalize_c_2, column_0)
            angle_rad_y = math.atan(slope_y)  # Calculate the angle in radians
            angle_deg_y = 90-math.degrees(angle_rad_y)  # Convert the angle to degrees
            arg_2=np.full((len(pcd),), angle_deg_y)
            
            
            
            # # Extract the first column (column 0) and calculate the mean
            area_element = [row[3] for row in data_element]  # List comprehension to extract the first column
            mean_area_element = sum(area_element) / len(area_element)
            #PRINT FIGURES FOR EACH SECTION
            
            # Create the plot along x-axis
            plt.scatter(normalize_c_1, column_0, label='Data Points')
            # Generate x values for the best-fit line
            best_fit_x = np.linspace(min(normalize_c_1), max(normalize_c_1), 100)
            # Calculate corresponding y values for the best-fit line
            best_fit_y = slope_x * best_fit_x + intercept_x
            plt.plot(best_fit_x, best_fit_y, 'r', label='Best fit line')

          
            plt.xlabel('deviation of the center of gravity along x-axis')
            plt.ylabel('height')
            plt.title('Inclination analysis along x-axis of Element_'+str(i))
            plt.legend()
            plt.grid(True)
            # Save the plot as a PNG file
            plt.savefig(save_path_textbox.get()+'/Element_'+str(i)+'_x_axis.png')
            # Clear the plot for the next iteration
            plt.clf()
            
            # Create the plot along y-axis
            plt.scatter(normalize_c_2, column_0, label='Data Points')
            # Generate x values for the best-fit line
            best_fit_xx = np.linspace(min(normalize_c_2), max(normalize_c_2), 100)
            # Calculate corresponding y values for the best-fit line
            best_fit_yy = slope_y * best_fit_xx + intercept_y
            plt.plot(best_fit_xx, best_fit_yy, 'r', label='Best fit line')

          
            plt.xlabel('deviation of the center of gravity along y-axis')
            plt.ylabel('height')
            plt.title('Inclination analysis along y-axis of Element_'+str(i))
            plt.legend()
            plt.grid(True)
            # Save the plot as a PNG file
            plt.savefig(save_path_textbox.get()+'/Element_'+str(i)+'_y_axis.png')
            # Clear the plot for the next iteration
            plt.clf()  
            # empty list to store new elements
            data_element = []
            # Cheking if the current inclination is lower that the maximum one
            if angle_deg_x>=limit:
                 safe_x=True
                 arg_3=np.full((len(pcd),), 1)
            else:
                 safe_x=False
                 arg_3=np.full((len(pcd),), 0)
            if angle_deg_y>=limit:
                 safe_y=True
                 arg_4=np.full((len(pcd),), 1)
            else:
                 safe_y=False
                 arg_4=np.full((len(pcd),), 0)               
            data.append(('Element_'+str(i),max(pcd['Z'])-min(pcd['Z']),mean_area_element,angle_deg_x,angle_deg_y,safe_x,safe_y))            
            
            # NEW POINTS CLOUD WITH THE SCALAR FIELDS
            npc=pc.clone()
            npc.setName('Element_'+str(i))
            CC.addToDB(npc)
            npc.addScalarField("Excesive inclination along x-axis", arg_3)       
            npc.addScalarField("Deflection along x-axis", arg_1)       
            npc.addScalarField("Excesive inclination along y-axis", arg_4)       
            npc.addScalarField("Deflection along y-axis", arg_2)       
            
             #WRITE FILE WITH THE GENERAL DATA
             # Open the file in write mode
            with open(save_path_textbox.get()+'/inclination_analysis.txt', 'w') as file:
             # Write the header
                 file.write("Identifier\tLenght\tMean area\tInclination angle along x-axis\tInclination angle along y-axis,\tIs within the inclination tolerante along x-axis?\tIs within the inclination tolerante along y-axis?\n")
                
            # Write the data to the file
                 for item in data:
                     file.write(f"{item[0]}\t{item[1]:.3f}\t{item[2]:.3f}\t{item[3]:.3f}\t{item[4]:.3f}\t{item[5]}\t{item[6]}\n")   
    print('The process has been finished')
    window.destroy()  # Close the window    


def destroy():
    window.destroy()  # Close the window   
    
# Create the main window
window = tk.Tk()

window.title("Inclination analyzer")
# Disable resizing the window
window.resizable(False, False)
# Remove minimize and maximize buttons (title bar only shows close button)
window.attributes('-toolwindow', 1)


# Create a frame for the form
form_frame = tk.Frame(window, padx=10, pady=10)
form_frame.pack()

# Etiquetas de los algoritmos
label_tolerance = tk.Label(form_frame, text="Thickness threshold:")
label_tolerance.grid(row=0, column=0, sticky="w",pady=2)

label_step=tk.Label(form_frame, text="Step between sections:")
label_step.grid(row=1, column=0, sticky="w",pady=2)

label_maximum_inclination = tk.Label(form_frame, text="Maximum inclination allowed:")
label_maximum_inclination.grid(row=2, column=0, sticky="w",pady=2)

label_type = tk.Label(form_frame, text="Type strategy used to compute the center of gravity")
label_type.grid(row=3, column=0)


save_path_label = tk.Label(form_frame, text="Path for saving the data:")
save_path_label.grid(row=4, column=0, sticky="w",pady=2)

# Entries
entry_tolerance = tk.Entry(form_frame)
entry_tolerance.grid(row=0, column=1, sticky="w",pady=2)
entry_tolerance.insert(0,0.02)

entry_step = tk.Entry(form_frame)
entry_step.grid(row=1,column=1, sticky="w",pady=2)
entry_step.insert(0,0.5)

entry_maximum_inclination = tk.Entry(form_frame)
entry_maximum_inclination.grid(row=2,column=1, sticky="w",pady=2)
entry_maximum_inclination.insert(0,2)

save_path_textbox = tk.Entry(form_frame)
save_path_textbox.grid(row=4,column=1, sticky="w",pady=2)

algorithms = ["Points","Convex Hull", "Rectangle fitting", "Circle fitting"]
combo_type = ttk.Combobox(form_frame, values=algorithms, state="readonly")
combo_type.current(0)
combo_type.grid(row=3, column=1, sticky="w",pady=2)

save_path_button = tk.Button(form_frame, text="...", command=select_path)
save_path_button.grid(row=4, column=2, sticky="e",pady=2)

run_button = tk.Button(form_frame, text="        OK        ", command=run_algorithm)
cancel_button = tk.Button(form_frame, text="     Cancel     ", command=destroy)
dummie_label = tk.Label(form_frame, text="")
dummie_label.grid(row=5, column=0, sticky="w",pady=2)
run_button.grid(row=6, column=1, sticky="e",padx=10)
cancel_button.grid(row=6, column=2, sticky="w")




# Start the main event loop
window.mainloop()