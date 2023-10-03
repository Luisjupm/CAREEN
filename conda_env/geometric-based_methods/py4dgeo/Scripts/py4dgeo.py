# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 12:47:21 2023

@author: Pablo
"""
#import cccorelib
#import pycc
#import subprocess

import py4dgeo
import os
import sys

import pandas as pd
import numpy as np
import open3d as o3d

from py4dgeo.m3c2ep import *
from py4dgeo.util import Py4DGeoError
from py4dgeo import write_m3c2_results_to_las

import pytest
import tempfile
import os

#script_directory = os.path.abspath(__file__)
#path_parts = script_directory.split(os.path.sep)

#additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'
#print (additional_modules_directory)
# sys.path.insert(0, additional_modules_directory)
# from main import P2p_getdata,get_istance

# type_data, number = get_istance()

# CC = pycc.GetInstance() 
current_directory=os.path.dirname(os.path.abspath(__file__))
# params = pycc.FileIOFilter.LoadParameters()
# params.parentWidget = CC.getMainWindow()
i1 = os.path.join(current_directory,'i1.laz')
i2 = os.path.join(current_directory,'i2.laz')

epoch1, epoch2 = py4dgeo.read_from_las(i1, i2)

json = os.path.join(current_directory,'metadata.json')

with open(py4dgeo.util.find_file(json), "rb") as load_f:
    scanpos_info_dict = eval(load_f.read())
epoch1.scanpos_info = scanpos_info_dict
epoch2.scanpos_info = scanpos_info_dict

corepoints = py4dgeo.read_from_las("i1.laz").cloud

# Cxx = np.loadtxt(py4dgeo.util.find_file("Cxx.csv"), dtype=np.float64, delimiter=",")
# tfM = np.loadtxt(py4dgeo.util.find_file("tfM.csv"), dtype=np.float64, delimiter=",")
# refPointMov = np.loadtxt(
#     py4dgeo.util.find_file("redPoint.csv"), dtype=np.float64, delimiter=","
# )

m3c2_ep = py4dgeo.M3C2EP(
    epochs=(epoch1, epoch2),
    corepoints=corepoints,
    normal_radii=(0.5, 1.0, 2.0),
    cyl_radii=(0.5,),
    max_distance=3.0,
#     Cxx=Cxx,
#     tfM=tfM,
#     refPointMov=refPointMov,
)

distances, uncertainties, covariance = m3c2_ep.run()

distances[0:8]
uncertainties["lodetection"][0:8]
uncertainties["spread1"][0:8]
uncertainties["num_samples1"][0:8]
covariance["cov1"][0, :, :]


