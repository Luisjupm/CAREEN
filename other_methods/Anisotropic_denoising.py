# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 19:06:20 2023

@author: Luisja
"""

import cccorelib
import pycc
import os
import subprocess

import pandas as pd
import numpy as np
import open3d as o3d



CC = pycc.GetInstance() 
current_directory=os.path.dirname(os.path.abspath(__file__))
params = pycc.FileIOFilter.LoadParameters()
params.parentWidget = CC.getMainWindow()
input_file=os.path.join(os.path.dirname(current_directory),'temp\\','input.ply')
output_file=os.path.join(os.path.dirname(current_directory),'temp\\','output.ply')
processing_file=os.path.join(current_directory,'Anisotropic_denoising-1.0.4\\Anisotropic_denoising-1.0.4.exe')
if not CC.haveSelection():
    raise RuntimeError("No folder or entity selected")
else:
    
    entities = CC.getSelectedEntities()[0]

    if hasattr(entities, 'points'):

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(entities.points())
        
        o3d.io.write_point_cloud(input_file,pcd, write_ascii=True)
        #run the cmd of anisotropic filter 
        command= processing_file + ' --i ' + input_file + ' --o ' + output_file
        os.system(command)
        pcd = o3d.io.read_point_cloud(output_file)
        # Convert Open3D.o3d.geometry.PointCloud to numpy array
        xyz_load = np.asarray(pcd.points)
        point_cloud = pycc.ccPointCloud(xyz_load[:,0], xyz_load[:,1], xyz_load[:,2])
        point_cloud.setName(entities.getName()+'_denoised')
        CC.addToDB(point_cloud)

    else:
        entities = CC.getSelectedEntities()[0]
        number = entities.getChildrenNumber()  
        for i in range (number):
            pc = entities.getChild(i)
            pcd = o3d.geometry.PointCloud()            
            pcd.points = o3d.utility.Vector3dVector(pc.points())
            
            o3d.io.write_point_cloud(input_file,pcd, write_ascii=True)
            #run the cmd of anisotropic filter 
            command= processing_file + ' --i ' + input_file + ' --o ' + output_file
            os.system(command)
            pcd = o3d.io.read_point_cloud(output_file)
            # Convert Open3D.o3d.geometry.PointCloud to numpy array
            xyz_load = np.asarray(pcd.points)
            point_cloud = pycc.ccPointCloud(xyz_load[:,0], xyz_load[:,1], xyz_load[:,2])
            point_cloud.setName(pc.getName()+'_denoised')
            CC.addToDB(point_cloud)
    CC.updateUI()

        
