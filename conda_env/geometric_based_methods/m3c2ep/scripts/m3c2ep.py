# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 18:48:34 2024

@author: LuisJa
"""
import numpy as np
import py4dgeo
epoch1, epoch2 = py4dgeo.read_from_las(
    "C:\\Users\\LuisJa\\Desktop\\m3c2ep_testdata\\ahk_2017_652900_5189100_gnd_red.laz",
    "C:\\Users\\LuisJa\\Desktop\\m3c2ep_testdata\\ahk_2018A_652900_5189100_gnd_red.laz",
    additional_dimensions={"point_source_id": "scanpos_id"},
)

with open("C:\\Users\\LuisJa\\Desktop\\m3c2ep_testdata\\sps.json", "r") as load_f:
    scanpos_info_dict = eval(load_f.read())
epoch1.scanpos_info = scanpos_info_dict
epoch2.scanpos_info = scanpos_info_dict
corepoints = py4dgeo.read_from_las("C:\\Users\\LuisJa\\Desktop\\m3c2ep_testdata\\ahk_cp_652900_5189100_red.laz").cloud
Cxx = np.loadtxt(py4dgeo.find_file("C:\\Users\\LuisJa\\Desktop\\m3c2ep_testdata\\Cxx.csv"), dtype=np.float64, delimiter=",")
tfM = np.loadtxt(py4dgeo.find_file("C:\\Users\\LuisJa\\Desktop\\m3c2ep_testdata\\tfM.csv"), dtype=np.float64, delimiter=",")
refPointMov = np.loadtxt("C:\\Users\\LuisJa\\Desktop\\m3c2ep_testdata\\redPoint.csv", dtype=np.float64, delimiter=","
)

m3c2_ep = py4dgeo.M3C2EP(
    epochs=(epoch1, epoch2),
    corepoints=corepoints,
    normal_radii=(0.5, 1.0, 2.0),
    cyl_radii=(0.5,),
    max_distance=3.0,
    Cxx=Cxx,
    tfM=tfM,
    refPointMov=refPointMov,
)

m3c2_ep.run()

print ("hola")