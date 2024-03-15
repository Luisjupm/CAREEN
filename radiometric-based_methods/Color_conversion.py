# -*- coding: utf-8 -*-
"""
Created on Mon May 15 23:00:29 2023

@author: Luisja
"""

import tkinter as tk
import pandas as pd
import cccorelib
import pycc
import colorsys
import os
import sys
import traceback

#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS

script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'
print (additional_modules_directory)
sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance
from main_gui import show_features_window, definition_of_labels_type_1,definition_of_entries_type_1, definition_of_combobox_type_1,definition_ok_cancel_buttons_type_1,definition_run_cancel_buttons_type_1, definition_of_buttons_type_1

#%% GUI
class GUI_cc(tk.Frame):
    def __init__(self, master=None, **kwargs): # Initial parameters. It is in self because we can update during the interaction with the user
        super().__init__(master, **kwargs)
        
        # Directory to save the files (output)
        self.output_directory=os.getcwd() # The working directory
    
    # Convert RGB to HSV function
    def rgb_to_hsv(self,row):
        r, g, b = row['R'], row['G'], row['B']
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return pd.Series([h, s, v])
    
    # Convert RGB to YCBCR function
    def rgb_to_ycbcr(self,row):
        r, g, b = row['R'], row['G'], row['B']
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 0.564 * (b - y)
        cr = 0.713 * (r - y)
        return pd.Series([y, cb, cr])
    
    # Convert RGB to YIQ function
    def rgb_to_yiq(self,row):
        r, g, b = row['R'], row['G'], row['B']
        y = 0.299 * r + 0.587 * g + 0.114 * b
        i = 0.596 * r - 0.274 * g - 0.322 * b
        q = 0.211 * r - 0.523 * g + 0.312 * b
        return pd.Series([y, i, q])
    
    # Convert RGB to YUV function
    def rgb_to_yuv(self,row):
        r, g, b = row['R'], row['G'], row['B']
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.147 * r - 0.289 * g + 0.436 * b
        v = 0.615 * r - 0.515 * g - 0.100 * b
        return pd.Series([y, u, v])
    # Create Scalar fields for the different color functions
    def color_conversion (self,selected_algorithms,pc,pcd):
         for algorithm in selected_algorithms:
             print(f"Coverting to: {algorithm}")
             if algorithm == 'HSV':
                 ## ADD YCBCR TO THE DATAFRAME
                 pcd[['H(HSV)', 'S(HSV)', 'V(HSV)']] = pcd.apply(self.rgb_to_hsv, axis=1)
                 ## ADD HSV AS SCALAR FIELDS
                 pc.addScalarField("H(HSV)", pcd['H(HSV)'])
                 pc.addScalarField("S(HSV)", pcd['S(HSV)'])
                 pc.addScalarField("V(HSV)", pcd['V(HSV)'])
             elif algorithm == 'YCbCr':            
                 ## ADD YCBCR TO THE DATAFRAME
                 pcd[['Y(YCbCr)', 'Cb(YCbCr)', 'Cr(YCbCr)']] = pcd.apply(self.rgb_to_ycbcr, axis=1)
                 ## ADD HSV AS SCALAR FIELDS
                 pc.addScalarField("Y(YCbCr)", pcd['Y(YCbCr)'])
                 pc.addScalarField("Cb(YCbCr)", pcd['Cb(YCbCr)'])
                 pc.addScalarField("Cr(YCbCr)", pcd['Cr(YCbCr)'])   
             elif algorithm == 'YIQ': 
                 ## ADD YIQ TO THE DATAFRAME
                 pcd[['Y(YIQ)', 'I(YIQ)', 'Q(YIQ)']] = pcd.apply(self.rgb_to_yiq, axis=1)
                 ## ADD HSV AS SCALAR FIELDS
                 pc.addScalarField("Y(YIQ)", pcd['Y(YIQ)'])
                 pc.addScalarField("I(YIQ)", pcd['I(YIQ)'])
                 pc.addScalarField("Q(YIQ)", pcd['Q(YIQ)'])   
             elif algorithm == 'YUV': 
             ## ADD YUV TO THE DATAFRAME
                 pcd[['Y(YUV)', 'U(YUV)', 'V(YUV)']] = pcd.apply(self.rgb_to_yuv, axis=1)
                 ## ADD HSV AS SCALAR FIELDS
                 pc.addScalarField("Y(YUV)", pcd['Y(YUV)'])
                 pc.addScalarField("U(YUV)", pcd['U(YUV)'])
                 pc.addScalarField("V(YUV)", pcd['V(YUV)'])  
    
    def main_frame (self,window):
        
        def destroy(self):
            window.destroy()  # Close the window
  
        window.title("Color maps")
        # Disable resizing the window
        window.resizable(False, False)
        # Remove minimize and maximize buttons (title bar only shows close button)
        window.attributes('-toolwindow', 1)
        # Create a frame for the form
        form_frame = tk.Frame(window, padx=10, pady=10)
        form_frame.pack()
    
        # Control variables
        algorithm1_var = tk.BooleanVar()
        algorithm2_var = tk.BooleanVar()
        algorithm3_var = tk.BooleanVar()
        algorithm4_var = tk.BooleanVar()
    
        # Labels
        label_texts = [
            "HSV (Hue-Saturation-Value)",
            "YCbCr",
            "YIQ",
            "YUV"
        ]
        row_positions = [0,1,2,3]        
        definition_of_labels_type_1 ("form_frame",label_texts,row_positions,form_frame,0)
    
        # Checkboxes
        algorithm1_checkbox = tk.Checkbutton(form_frame, variable=algorithm1_var)
        algorithm1_checkbox.grid(row=0, column=1, sticky="e")
        
        algorithm2_checkbox = tk.Checkbutton(form_frame, variable=algorithm2_var)
        algorithm2_checkbox.grid(row=1, column=1, sticky="e")       
        
        algorithm3_checkbox = tk.Checkbutton(form_frame, variable=algorithm3_var)
        algorithm3_checkbox.grid(row=2, column=1, sticky="e")       
        
        algorithm4_checkbox = tk.Checkbutton(form_frame, variable=algorithm4_var)
        algorithm4_checkbox.grid(row=3, column=1, sticky="e")        
        
        # Buttons
        _=definition_run_cancel_buttons_type_1("form_frame",
                                     [lambda:run_algorithm_1(self,bool(algorithm1_var.get()),bool(algorithm2_var.get()),bool(algorithm3_var.get()),bool(algorithm4_var.get())),lambda:destroy(self)],
                                     4,
                                     form_frame,
                                     1
                                     )
            
        def run_algorithm_1(self,hsv,ycbcr,yiq,yuv):
            
            ## CONTROL THE SELECTION OF THE INPUT
            CC = pycc.GetInstance() 
            if not CC.haveSelection():
                raise RuntimeError("No folder or point cloud has been selected")
                
            selected_algorithms = []

            if hsv:
                selected_algorithms.append("HSV")
            
            if ycbcr:
                selected_algorithms.append("YCbCr")
            
            if yiq:
                selected_algorithms.append("YIQ")
                
            if yuv:
                selected_algorithms.append("YUV") 
           #Get data
            CC = pycc.GetInstance()  
            type_data, number= get_istance()
            if type_data=='folder':
               entities = CC.getSelectedEntities()[0]
               ## LOOP OVER EACH ELEMENT
               for i in range(number):
                   pc = entities.getChild(i)  
                   pcd=P2p_getdata(pc,False, True, True) 
                   print(pcd.columns)
                   self.color_conversion(selected_algorithms,pc,pcd)  
                   CC.addToDB(pc)
                   CC.updateUI() 
            elif type_data=='point_cloud':
               entities = CC.getSelectedEntities()
               pc = entities[0]
               pcd=P2p_getdata(pc,False, True, True)
               print(pcd.columns)
               self.color_conversion(selected_algorithms,pc,pcd)
               CC.addToDB(pc)
               CC.updateUI()            
    
    
            print('The color scales has been added to the scalar fields of the point cloud')  
            self.destroy()  # Close the window
        
    def show_frame(self,window):
        self.main_frame(window)
        self.grid(row=1, column=0, pady=10)

    def hide_frame(self):
        self.grid_forget()
        
#%% START THE GUI        
if __name__ == "__main__":
    try:        
        window = tk.Tk()
        app = GUI_cc()
        app.main_frame(window)
        window.mainloop()    
    except Exception as e:
        print("An error occurred during the computation of the algorithm:", e)
        
        # Optionally, print detailed traceback
        traceback.print_exc()
        window.destroy()