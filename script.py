# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 12:43:14 2023

@author: Pablo
"""

import os
import sys
# ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS

script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'
print (additional_modules_directory)
sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance

type_data, number = get_istance()

for i in range(number):
            pc = entities.getChild(i)
            pcd=P2p_getdata(pc,False,True,True)

