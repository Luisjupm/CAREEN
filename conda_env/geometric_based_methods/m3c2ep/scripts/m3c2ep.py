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
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--e1',type=str,help='Path for epoch_1 point cloud in laz format')
    parser.add_argument('--e2',type=str,help='Path for epoch_1 point cloud in laz format')
    parser.add_argument('--sp',type=str,help='Name of the scalar field that store the position of the scanner')
    parser.add_argument('--c',type=str,help='Path for the corepoints in laz format. If is None the core points will not be used')
    parser.add_argument('--pos',type=str,help='Path of the position file wihtt scanner undertainities in json format')
    parser.add_argument('--nr',type=str,help=' Path of the normal radii txt file ex. (0.5, 1.0, 2.0) type "None if the normals are not required')
    parser.add_argument('--cr',type=str,help=' Path of the cylinder radii txt file ex. (0.5,)')
    parser.add_argument('--md',type=str,help=' Maximum search distance as float')
    parser.add_argument('--cxx',type=str,help=' Path for covariance matrix in csv format 12 columsn, 12 rows. Type "None" if we dont use it')
    parser.add_argument('--tfm',type=str,help='Path for the transformation matrix in csv format 4 columns, 3 rows Type "None" if we dont use it')
    args=parser.parse_args()  
    
    epoch1, epoch2 = py4dgeo.read_from_las(
        args.e1,
        args.e2,
        additional_dimensions={"point_source_id": args.sp},
    )
    with open(py4dgeo.find_file(args.pos), "r") as load_f:
        scanpos_info_dict = eval(load_f.read())
    epoch1.scanpos_info = scanpos_info_dict
    epoch2.scanpos_info = scanpos_info_dict
    
    if args.c!="None":
        corepoints = py4dgeo.read_from_las(args.c).cloud
    else:
        corepoints = py4dgeo.read_from_las(args.e1).cloud     
    if args.nr=="None":
        normal_radii=[]
    else:
        normal_radii=args.nr    
    m3c2ep = M3C2EP(
            epochs=(epoch1, epoch2),
            corepoints=corepoints,
            corepoint_normals=corepoint_normals,
            normal_radii=normal_radii,
            cyl_radii=args.cr,
            max_distance=args.md,
            Cxx=args.cxx,
            tfM=args.tfm)    
    # Run it and check results exists with correct shapes
    distances, uncertainties, covariance = m3c2ep.run()      
            

if __name__=='__main__':
	main()