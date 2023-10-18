# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:07:07 2023

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

ç



