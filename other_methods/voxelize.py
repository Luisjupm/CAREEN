# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 22:00:43 2023

@author: LuisJa
"""
import cccorelib
import pycc
import os
import sys
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import open3d as o3d
import numpy as np


#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS

script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'
sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance,get_point_clouds_name

#%% SET-UP THE TEMPORAL FOLDER
current_directory=os.path.dirname(os.path.abspath(__file__))
input_file=os.path.join(os.path.dirname(current_directory),'temp_folder\\','voxelized_point_cloud.ply')   

#%% CREATE A INSTANCE WITH THE ELEMENT SELECTED
CC = pycc.GetInstance() 
entities= CC.getSelectedEntities()[0]
#%% INPUTS AT THE BEGINING
name_list=get_point_clouds_name()

#%% VOXELIZATION OF THE POINT CLOUD
v_size=0.02
def run_algorithm():
    # Get the input values
    v_size=float(entry_voxel_size.get()) 
    # Error if there is not selection
    if combo1.get()=="Not selected":
        raise RuntimeError("Please select a point cloud to process the data")
    if not CC.haveSelection():
        raise RuntimeError("No folder or entity selected")
    else:
        # Load the selected point cloud. If the entity has the attribute points is a folder. Otherwise is a point cloud
        if hasattr (entities, 'points'):
            
            pc=entities
        else:            
            for ii, item in enumerate (name_list):
                if item== combo1.get():
                    pc = entities.getChild(ii)
                    break 
            print ("hola")
        if hasattr(pc, 'points'): # If there is selection and the selected entity is a point
            # Transform the selected point cloud to a open3d point cloud
            pcd = o3d.geometry.PointCloud()            
            pcd.points = o3d.utility.Vector3dVector(pc.points())
            # pcd.color= o3d.utility.Vector3dVector(pc.colors()[:,0],pc.colors()[:,1],pc.colors()[:,2])
            # Create the voxel model from the pcd point cloud
            voxel_grid=o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=v_size)
            voxels=voxel_grid.get_voxels()
            vox_mesh=o3d.geometry.TriangleMesh()
            for v in voxels:
                cube=o3d.geometry.TriangleMesh.create_box(width=1, height=1,
                depth=1)
                cube.paint_uniform_color(v.color)
                cube.translate(v.grid_index, relative=False)
                vox_mesh+=cube
            vox_mesh.translate([0.5,0.5,0.5], relative=True)
            vox_mesh.scale(v_size, [0,0,0])
            vox_mesh.translate(voxel_grid.origin, relative=True)
            vox_mesh.merge_close_vertices(0.0000001)
            # Save the file and then load the file with cloudcompare. It is used a temporal folder for this process. Finally the temporal file is deleted
            o3d.io.write_triangle_mesh(input_file,vox_mesh)        
            params = pycc.FileIOFilter.LoadParameters()
            params.alwaysDisplayLoadDialog=False
            CC.loadFile(input_file, params)
            os.remove(input_file)
        else: # If there is selection is a folder perform a for loop chosing the children and checking if the children is a point cloud
            raise RuntimeError("The selected entity is not a point cloud")
    #%% UPDATE THE DB    
    CC.updateUI()        
    print('The process has been completed!')  
    window.destroy()  # Close the window   
def destroy():
    window.destroy()  # Close the window        
#%% GUI
# Create the main window
window = tk.Tk()

window.title("Voxelize point cloud")
# Disable resizing the window
window.resizable(False, False)
# Remove minimize and maximize buttons (title bar only shows close button)
window.attributes('-toolwindow', 1)

# Create a frame for the form
form_frame = tk.Frame(window, padx=10, pady=10)
form_frame.pack()

# Labels
label_pc = tk.Label(form_frame, text="Select a point cloud:")
label_pc.grid(row=0, column=0, sticky="w",pady=2)
label_tolerance = tk.Label(form_frame, text="Select the voxel size:")
label_tolerance.grid(row=1, column=0, sticky="w",pady=2)


# Combobox
combo1=ttk.Combobox (form_frame,values=name_list)
combo1.grid(column=1, row=0, sticky="e", pady=2)
combo1.set("Not selected")
# Entries
entry_voxel_size = tk.Entry(form_frame,width=5)
entry_voxel_size.insert(0,0.02)
entry_voxel_size.grid(row=1, column=1, sticky="e",pady=2)

# Buttons
run_button = tk.Button(form_frame, text="OK", command=run_algorithm,width=10)
cancel_button = tk.Button(form_frame, text="Cancel", command=destroy,width=10)
run_button.grid(row=2, column=1, sticky="e",padx=100)
cancel_button.grid(row=2, column=1, sticky="e")


#%% START THE GUI
# Start the main event loop
window.mainloop()
