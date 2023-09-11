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

# DEFINE A CLASS FOR THE WINDOW (IT HAS ANOTHER POP-UP WINDOWS)
class App(tk.Tk):
    def P2p_getdata (pc,nan_value=False,sc=True):
    ## CREATE A DATAFRAME WITH THE POINTS OF THE PC
       pcd = pd.DataFrame(pc.points(), columns=['X', 'Y', 'Z'])
       ## GET THE RGB COLORS
       pcd['R']=pc.colors()[:,0]
       pcd['G']=pc.colors()[:,1] 
       pcd['B']=pc.colors()[:,2] 
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
   
    ## LOAD THE SELECTED POINT CLOUD
    CC = pycc.GetInstance() 
    entities = CC.getSelectedEntities()
    print(f"Selected entities: {entities}")
    if not entities:
        raise RuntimeError("No entities selected")
    else:
        pc = entities[0]
        pcd=P2p_getdata(pc,False,True)
    
 
    def __init__(self):
        CC = pycc.GetInstance()
        
        
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
        self.run_button = tk.Button(form_frame, text="        OK        ", command=self.run_algorithms)
        self.cancel_button = tk.Button(form_frame, text="     Cancel     ", command=self.destroy)
        self.run_button.grid(row=4, column=0, sticky="e",padx=10)
        self.cancel_button.grid(row=4, column=1, sticky="w")        

        
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
            
        # Aquí puedes ejecutar los algoritmos seleccionados
        for algorithm in selected_algorithms:
            print(f"Coverting to: {algorithm}")
            if algorithm == 'HSV':
                ## ADD HSV TO THE DATAFRAME
                self.pcd[['H(HSV)', 'S(HSV)', 'V(HSV)']] = self.pcd.apply(self.rgb_to_hsv, axis=1)
                ## ADD HSV AS SCALAR FIELDS
                self.pc.addScalarField("H(HSV)", self.pcd['H(HSV)'])
                self.pc.addScalarField("S(HSV)", self.pcd['S(HSV)'])
                self.pc.addScalarField("V(HSV)", self.pcd['V(HSV)'])
            elif algorithm == 'YCbCr':            
                ## ADD YCBCR TO THE DATAFRAME
                self.pcd[['Y(YCbCr)', 'Cb(YCbCr)', 'Cr(YCbCr)']] = self.pcd.apply(self.rgb_to_ycbcr, axis=1)
                ## ADD HSV AS SCALAR FIELDS
                self.pc.addScalarField("Y(YCbCr)", self.pcd['Y(YCbCr)'])
                self.pc.addScalarField("Cb(YCbCr)", self.pcd['Cb(YCbCr)'])
                self.pc.addScalarField("Cr(YCbCr)", self.pcd['Cr(YCbCr)'])   
            elif algorithm == 'YIQ': 
                ## ADD YIQ TO THE DATAFRAME
                self.pcd[['Y(YIQ)', 'I(YIQ)', 'Q(YIQ)']] = self.pcd.apply(self.rgb_to_yiq, axis=1)
                ## ADD HSV AS SCALAR FIELDS
                self.pc.addScalarField("Y(YIQ)", self.pcd['Y(YIQ)'])
                self.pc.addScalarField("I(YIQ)", self.pcd['I(YIQ)'])
                self.pc.addScalarField("Q(YIQ)", self.pcd['Q(YIQ)'])   
            elif algorithm == 'YUV': 
            ## ADD YUV TO THE DATAFRAME
                self.pcd[['Y(YUV)', 'U(YUV)', 'V(YUV)']] = self.pcd.apply(self.rgb_to_yuv, axis=1)
                ## ADD HSV AS SCALAR FIELDS
                self.pc.addScalarField("Y(YUV)", self.pcd['Y(YUV)'])
                self.pc.addScalarField("U(YUV)", self.pcd['U(YUV)'])
                self.pc.addScalarField("V(YUV)", self.pcd['V(YUV)']) 
        self.CC.addToDB(self.pc)
        self.CC.updateUI() 
        print('The color scales has been added to the scalar fields of the point cloud')  
        self.destroy()  # Close the window
        
app = App()
app.mainloop()