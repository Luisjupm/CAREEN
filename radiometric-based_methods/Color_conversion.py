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

# ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS

script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'
print (additional_modules_directory)
sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance



# DEFINE A CLASS FOR THE WINDOW (IT HAS ANOTHER POP-UP WINDOWS)
class GUI_cc(tk.Frame):

   
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
    def __init__(self):
        
        ## CONTROL THE SELECTION OF THE INPUT
        CC = pycc.GetInstance() 
        if not CC.haveSelection():
            raise RuntimeError("No folder or point cloud has been selected")
      
        
        super().__init__()
        self.title("Color maps")
        # Disable resizing the window
        self.resizable(False, False)
        # Remove minimize and maximize buttons (title bar only shows close button)
        self.attributes('-toolwindow', 1)
        # Create a frame for the form
        form_frame = tk.Frame(self, padx=10, pady=10)
        form_frame.pack()

        # Control variables
        self.algorithm1_var = tk.BooleanVar()
        self.algorithm2_var = tk.BooleanVar()
        self.algorithm3_var = tk.BooleanVar()
        self.algorithm4_var = tk.BooleanVar()

        # Labels
        self.algorithm1_label = tk.Label(form_frame, text="HSV (Hue-Saturation-Value")
        self.algorithm1_label.grid(row=0, column=0, sticky="w",pady=2)
       
        self.algorithm2_label = tk.Label(form_frame, text="YCbCr")
        self.algorithm2_label.grid(row=1, column=0, sticky="w",pady=2)
       
        self.algorithm3_label = tk.Label(form_frame, text="YIQ")
        self.algorithm3_label.grid(row=2, column=0, sticky="w",pady=2) 
        
        self.algorithm4_label = tk.Label(form_frame, text="YUV")
        self.algorithm4_label.grid(row=3, column=0, sticky="w",pady=2)
        
        # Checkboxes
        self.algorithm1_checkbox = tk.Checkbutton(form_frame, variable=self.algorithm1_var)
        self.algorithm1_checkbox.grid(row=0, column=1, sticky="e")
        
        self.algorithm2_checkbox = tk.Checkbutton(form_frame, variable=self.algorithm2_var)
        self.algorithm2_checkbox.grid(row=1, column=1, sticky="e")       
        
        self.algorithm3_checkbox = tk.Checkbutton(form_frame, variable=self.algorithm3_var)
        self.algorithm3_checkbox.grid(row=2, column=1, sticky="e")       
        
        self.algorithm4_checkbox = tk.Checkbutton(form_frame, variable=self.algorithm4_var)
        self.algorithm4_checkbox.grid(row=3, column=1, sticky="e")        
        
        # Buttons       
        self.run_button = tk.Button(form_frame, text="OK", command=self.run_algorithms,width=10)
        self.cancel_button = tk.Button(form_frame, text="Cancel", command=self.destroy,width=10)
        self.run_button.grid(row=4, column=1, sticky="e",padx=100)
        self.cancel_button.grid(row=4, column=1, sticky="e")        

        
    def run_algorithms(self):
        selected_algorithms = []
       
     
        if self.algorithm1_var.get():
            selected_algorithms.append("HSV")
        
        if self.algorithm2_var.get():
            selected_algorithms.append("YCbCr")
        
        if self.algorithm3_var.get():
            selected_algorithms.append("YIQ")
            
        if self.algorithm4_var.get():
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
               self.color_conversion(selected_algorithms,pc,pcd)  
               CC.addToDB(pc)
               CC.updateUI() 
        elif type_data=='point_cloud':
           entities = CC.getSelectedEntities()
           pc = entities[0]
           pcd=P2p_getdata(pc,False, True, True)
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