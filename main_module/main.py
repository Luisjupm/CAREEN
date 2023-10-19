# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 19:16:07 2023

@author: Utilizador
"""


import pandas as pd
import pycc


## CREATE A DATAFRAME WITH THE POINTS OF THE PC
def P2p_getdata (pc,nan_value=False,sc=True,color=True):
## CREATE A DATAFRAME WITH THE POINTS OF THE PC
   pcd = pd.DataFrame(pc.points(), columns=['X', 'Y', 'Z'])
   if color==True:
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

def get_istance ():
    CC = pycc.GetInstance() 
    #CHECKING THE FOLDER
    if not CC.haveSelection():
        raise RuntimeError("You need to select a folder or a point cloud")
    else:            
        entities = CC.getSelectedEntities()[0]        
    if hasattr(entities, 'points'):
        type_data='point_cloud'
        number=1
    else:
        type_data='folder'
        number = entities.getChildrenNumber()   

    return type_data,number

def get_point_clouds ():
    CC = pycc.GetInstance() 
    #CHECKING THE FOLDER
    if not CC.haveSelection():
        raise RuntimeError("You need to select a folder or a point cloud")
    else:            
        entities = CC.getSelectedEntities()[0]        
    if hasattr(entities, 'points'):
        type_data='point_cloud'
        number=1
    else:
        type_data='folder'
        number = entities.getChildrenNumber()   

    return type_data,number  
  
def get_point_clouds_name ():
    CC = pycc.GetInstance() 
    name_list = []  # Create an empty list to store names
    #CHECKING THE FOLDER
    if not CC.haveSelection():
        raise RuntimeError("You need to select a folder or a point cloud")
    else:            
        entities = CC.getSelectedEntities()[0]
            
    if hasattr(entities, 'points'):
        type_data='point_cloud'
        name_list= entities.getName()
        number=1
    else:
        type_data='folder'
        number = entities.getChildrenNumber()   
        for i in range (number):
            new_name=entities.getChild(i).getName()
            name_list.append(new_name)           
    return name_list    