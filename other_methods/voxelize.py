# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 22:00:43 2023

@author: LuisJa
"""
import cccorelib
import pycc
import os
import sys
import traceback

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
from main import P2p_getdata,get_istance,get_point_clouds_name, check_input, write_yaml_file
from main_gui import show_features_window, definition_of_labels_type_1,definition_of_entries_type_1, definition_of_combobox_type_1,definition_ok_cancel_buttons_type_1,definition_run_cancel_buttons_type_1, definition_of_buttons_type_1

#%% CREATE A INSTANCE WITH THE ELEMENT SELECTED
CC = pycc.GetInstance() 
entities= CC.getSelectedEntities()[0]

#%% INITIAL OPERATIONS
name_list=get_point_clouds_name()

#%% GUI
class GUI_voxelize(tk.Frame):
    def __init__(self, master=None, **kwargs): # Initial parameters. It is in self because we can update during the interaction with the user
        super().__init__(master, **kwargs)
        
        # Directory to save the files (output)
        self.output_directory=os.getcwd() # The working directory
    
    def main_frame (self, window):    # Main frame of the GUI  

        # Function to create tooltip
        def create_tooltip(widget, text):
            widget.bind("<Enter>", lambda event: show_tooltip(text))
            widget.bind("<Leave>", hide_tooltip)

        def show_tooltip(text):
            tooltip.config(text=text)
            tooltip.place(relx=0.5, rely=0.5, anchor="center", bordermode="outside")
            
        def hide_tooltip(event):
            tooltip.place_forget()

        tooltip = tk.Label(window, text="", relief="solid", borderwidth=1)
        tooltip.place_forget()
        
        # Destroy the window
        def destroy (self): 
            window.destroy ()
            
        window.title("Voxelize point cloud")
        # Disable resizing the window
        window.resizable(False, False)
        # Remove minimize and maximize buttons (title bar only shows close button)
        window.attributes('-toolwindow', 1)
        
        # Create a frame for the form
        form_frame = tk.Frame(window, padx=10, pady=10)
        form_frame.pack()
        
        # Labels
        label_texts = [
            "Select a point cloud:",
            "Select the voxel size:",
        ]
        row_positions = [0,1]        
        definition_of_labels_type_1 ("window",label_texts,row_positions,form_frame,0)
        
        # Combobox
        combo_point_cloud=ttk.Combobox (form_frame,values=name_list)
        combo_point_cloud.grid(column=1, row=0, sticky="e", pady=2)
        combo_point_cloud.set("Not selected")
        
        # Entries
        entry_voxel_size = ttk.Entry(form_frame,width=5)
        entry_voxel_size.insert(0,0.02)
        entry_voxel_size.grid(row=1, column=1, sticky="e",pady=2)
        
        # Buttons
        _=definition_run_cancel_buttons_type_1("window",
                                     [lambda:run_algorithm_1(self,name_list,combo_point_cloud.get(),float(entry_voxel_size.get())),lambda:destroy(self)],
                                     2,
                                     form_frame,
                                     1
                                     )

        def run_algorithm_1(self,name_list,pc_name,v_size):
            # Check if the selection is a point cloud
            pc=check_input(name_list,pc_name)
            
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
            o3d.io.write_triangle_mesh(pc_name,vox_mesh)        
            params = pycc.FileIOFilter.LoadParameters()
            params.alwaysDisplayLoadDialog=False
            CC.loadFile(pc_name, params)
            os.remove(pc_name)
            
                
            CC.updateUI()        
            print('The process has been completed')  
            window.destroy()  # Close the window
            
    def show_frame(self,window):
        self.main_frame(window)
        self.grid(row=1, column=0, pady=10)

    def hide_frame(self):
        self.grid_forget()

#%% RUN THE GUI
if __name__ == "__main__":        
    try:
        # START THE MAIN WINDOW        
        window = tk.Tk()
        app = GUI_voxelize()
        app.main_frame(window)
        window.mainloop()    
    except Exception as e:
        print("An error occurred during the computation of the algorithm:", e)
        # Optionally, print detailed traceback
        traceback.print_exc()
        window.destroy()
