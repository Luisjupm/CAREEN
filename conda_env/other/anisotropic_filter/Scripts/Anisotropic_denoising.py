# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 12:02:38 2023

@author: Luisja
"""

import argparse


from anisofilter import utilities as UTI
from anisofilter import anisofilter 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--i',type=str,help='Path for the input file')
    parser.add_argument('--o',type=str,help='Path for the output file')
    args=parser.parse_args()  
    
    
    
    
    pcd = UTI.read_ply_single_class(args.i)
    sigma_pcd, dens_pcd = UTI.pcd_std_est(pcd)
    pcd_de_m2c = anisofilter.anisofilter(pcd, sigma_pcd, dens_pcd)
    # write to ply file
    UTI.write_ply_only_pos(pcd_de_m2c, args.o)




if __name__=='__main__':
	main()