# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 17:40:42 2023

@author: Luisja
"""
"""
Batch proccess of the algorithm proposed in:  

Winiwarter, L., Anders, K., & Höfle, B. (2021). M3C2-EP: Pushing the limits of 3D topographic point cloud change detection by error propagation. ISPRS Journal of Photogrammetry and Remote Sensing, 178, 240-258.    

"""

import argparse


import numpy as np
import py4dgeo
from py4dgeo.m3c2ep import *

