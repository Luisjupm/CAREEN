# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 18:00:57 2023

@author: Luisja
"""
import tkinter as tk

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
from tkinter import ttk
from tkinter import filedialog
import os
import sys

# ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS

script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'
print (additional_modules_directory)
sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance


def minBoundingRect(hull_points_2d):
    #print "Input convex hull points: "
    #print hull_points_2d

    # Compute edges (x2-x1,y2-y1)
    edges = np.zeros( (len(hull_points_2d)-1,2) ) # empty 2 column array
    for i in range( len(edges) ):
        edge_x = hull_points_2d[i+1,0] - hull_points_2d[i,0]
        edge_y = hull_points_2d[i+1,1] - hull_points_2d[i,1]
        edges[i] = [edge_x,edge_y]
    #print "Edges: \n", edges

    # Calculate edge angles   atan2(y/x)
    edge_angles = np.zeros( (len(edges)) ) # empty 1 column array
    for i in range( len(edge_angles) ):
        edge_angles[i] = np.math.atan2( edges[i,1], edges[i,0] )
    #print "Edge angles: \n", edge_angles

    # Check for angles in 1st quadrant
    for i in range( len(edge_angles) ):
        edge_angles[i] = abs( edge_angles[i] % (np.math.pi/2) ) # want strictly positive answers
    #print "Edge angles in 1st Quadrant: \n", edge_angles

    # Remove duplicate angles
    edge_angles = np.unique(edge_angles)
    #print "Unique edge angles: \n", edge_angles

    # Test each angle to find bounding box with smallest area
    min_bbox = (0, 100000, 0, 0, 0, 0, 0, 0) # rot_angle, area, width, height, min_x, max_x, min_y, max_y
    for i in range( len(edge_angles) ):

        # Create rotation matrix to shift points to baseline
        # R = [ cos(theta)      , cos(theta-PI/2)
        #       cos(theta+PI/2) , cos(theta)     ]
        R = np.array([ [ np.math.cos(edge_angles[i]), np.math.cos(edge_angles[i]-(np.math.pi/2)) ], [ np.math.cos(edge_angles[i]+(np.math.pi/2)), np.math.cos(edge_angles[i]) ] ])
        #print "Rotation matrix for ", edge_angles[i], " is \n", R

        # Apply this rotation to convex hull points
        rot_points = np.dot(R, np.transpose(hull_points_2d) ) # 2x2 * 2xn
        #print "Rotated hull points are \n", rot_points

        # Find min/max x,y points
        min_x = np.nanmin(rot_points[0], axis=0)
        max_x = np.nanmax(rot_points[0], axis=0)
        min_y = np.nanmin(rot_points[1], axis=0)
        max_y = np.nanmax(rot_points[1], axis=0)
        #print "Min x:", min_x, " Max x: ", max_x, "   Min y:", min_y, " Max y: ", max_y

        # Calculate height/width/area of this bounding rectangle
        width = max_x - min_x
        height = max_y - min_y
        area = width*height
        #print "Potential bounding box ", i, ":  width: ", width, " height: ", height, "  area: ", area 

        # Store the smallest rect found first (a simple convex hull might have 2 answers with same area)
        if (area < min_bbox[1]):
            min_bbox = ( edge_angles[i], area, width, height, min_x, max_x, min_y, max_y )
        # Bypass, return the last found rect
        #min_bbox = ( edge_angles[i], area, width, height, min_x, max_x, min_y, max_y )

    # Re-create rotation matrix for smallest rect
    angle = min_bbox[0]   
    R = np.array([ [ np.math.cos(angle), np.math.cos(angle-(np.math.pi/2)) ], [ np.math.cos(angle+(np.math.pi/2)), np.math.cos(angle) ] ])

    #print "Project hull points are \n", proj_points

    # min/max x,y points are against baseline
    min_x = min_bbox[4]
    max_x = min_bbox[5]
    min_y = min_bbox[6]
    max_y = min_bbox[7]
    #print "Min x:", min_x, " Max x: ", max_x, "   Min y:", min_y, " Max y: ", max_y

    # Calculate center point and project onto rotated frame
    center_x = (min_x + max_x)/2
    center_y = (min_y + max_y)/2
    center_point = np.dot( [ center_x, center_y ], R )
    #print "Bounding box center point: \n", center_point

    # Calculate corner points and project onto rotated frame
    corner_points = np.zeros( (4,2) ) # empty 2 column array
    corner_points[0] = np.dot( [ max_x, min_y ], R )
    corner_points[1] = np.dot( [ min_x, min_y ], R )
    corner_points[2] = np.dot( [ min_x, max_y ], R )
    corner_points[3] = np.dot( [ max_x, max_y ], R )
    
    
    # Calculate the midpoints on each side of the rectangle
    midpoints = np.zeros( (4,2) ) # empty 2 column array
    midpoints[0] = (corner_points[0] + corner_points[3]) / 2.0
    midpoints[1] = (corner_points[0] + corner_points[1]) / 2.0
    midpoints[2] = (corner_points[1] + corner_points[2]) / 2.0
    midpoints[3] = (corner_points[2] + corner_points[3]) / 2.0

    #print "Angle of rotation: ", angle, "rad  ", angle * (180/math.pi), "deg"

    return (angle, min_bbox[1], min_bbox[2], min_bbox[3], center_point, corner_points,midpoints) # rot_angle, area, width, height, center_point, corner_points, mid_points

def extract_longitudinal_axis(midpoints):
    # Get all combinations of two midpoints
    point_combinations = list(combinations(midpoints, 2))
    
    # Calculate the distances between the point combinations
    distances = [np.linalg.norm(p2 - p1) for p1, p2 in point_combinations]
    
    # Find the index of the largest distance
    max_distance_index = np.argmax(distances)
    
    # Extract the two midpoints corresponding to the largest distance
    p1, p2 = point_combinations[max_distance_index]
    
    # Get the longitudinal axis (line connecting the two parallel sides)
    longitudinal_axis = p2 - p1
    longitudinal_axis_norm = distance.euclidean(p1, p2)
    
    #Calculate the angle between the longitudinal axis and the x-axis
    angle_rad = np.arctan2(longitudinal_axis[1], longitudinal_axis[0])
    angle_deg = np.degrees(angle_rad)

    return  p1,p2,longitudinal_axis, longitudinal_axis_norm, angle_deg


def extract_points_within_tolerance(point_cloud, tolerance,rot):
    # Calculate the minimum bounding rectangle
    point_cloud_red=point_cloud[:,:2]

    _, _, _, _, center_point, corner_points,midpoints= minBoundingRect(point_cloud_red)
    
    p1,p2,longitudinal_axis, longitudinal_axis_norm, angle_deg=extract_longitudinal_axis(midpoints)
    perpendicular = np.array([-longitudinal_axis[1], longitudinal_axis[0]])
    perpendicular /= distance.euclidean([0, 0], perpendicular)
    # Calculate the dot product of each point with the direction vector
    dot_products = np.dot(point_cloud_red-p1, perpendicular)
        
    # Determine the points within the tolerance
    points_within_tolerance = point_cloud[np.abs(dot_products) <= tolerance]
    skeleton=points_within_tolerance
    if rot==True:
        points_within_tolerance=rotate_point_cloud_3d(points_within_tolerance, angle_deg)

    return points_within_tolerance, skeleton

def rotate_point_cloud_3d(point_cloud, angle_deg):
    # Convert the rotation angle from degrees to radians
    angle_rad = np.radians(angle_deg)
    
    # Compute the rotation matrix around the z-axis
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    rotation_matrix = np.array([[cos_theta, -sin_theta, 0],
                                [sin_theta, cos_theta, 0],
                                [0, 0, 1]])
    
    # Apply the rotation matrix to the point cloud
    rotated_point_cloud = np.dot(point_cloud, rotation_matrix)
    
    return rotated_point_cloud

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

# Labels for the algorithms
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

algorithms = ["Data", "Fit"]
combo_type = ttk.Combobox(form_frame, values=algorithms, state="readonly")
combo_type.current(0)
combo_type.grid(row=3, column=1, sticky="e",pady=2)



save_path_button = tk.Button(form_frame, text="...", command=select_path,width=2)
save_path_button.grid(row=5, column=1, sticky="e",pady=2)

run_button = tk.Button(form_frame, text="OK", command=run_algorithm,width=10)
cancel_button = tk.Button(form_frame, text="Cancel", command=destroy,width=10)
run_button.grid(row=6, column=1, sticky="e",padx=100)
cancel_button.grid(row=6, column=1, sticky="e")



# Start the main event loop
window.mainloop()