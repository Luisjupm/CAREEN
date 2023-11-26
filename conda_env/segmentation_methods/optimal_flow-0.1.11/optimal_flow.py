# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:34:55 2023

@author: Digi_2
"""

import argparse

from optimalflow.autoFS import dynaFS_clf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--i',type=str,help='Path for the input file')
    parser.add_argument('--o',type=str,help='Path for the output file')
    parser.add_argument('--s',type=str,help='Path for the selectors')
    parser.add_argument('--f',type=str,help='Path for the features')
    parser.add_argument('--cv',type=str,help='Path for the cross validation')
    
    args=parser.parse_args()  
    
    
    input_file=args.i
    print("Input file located in " + input_file)
    output_directory=args.o
    print("Output file located in " + output_directory)
    selectors=args.s
    print("Selectors chosen = " + selectors)
    features=args.f
    print("Features chosen = " + features)
    cross_val=args.cv
    print("Value chosen for cross validation = " + cross_val)


    reg_fs_demo = dynaFS_clf(
                            custom_selectors=selectors, 
                            fs_num = features, 
                            random_state = None, 
                            cv = cross_val, 
                            input_from_file = False
                            )

    _,features_selected=reg_fs_demo.fit(tr_features_filled,tr_labels_2d)



if __name__=='__main__':
	main()