# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:34:55 2023

@author: Digi_2
"""

import argparse


import pandas as pd
import optimalflow

def main():
    
    
    
    dataset='dataset.xyz'
    selectors = ['kbest_f','rfe_lr','rfe_tree','rfe_rf','rfecv_tree','rfecv_rf']
    features = 25
    cross_val = 5
    
    
    tr = pd.read_csv(dataset,delimiter=' ')
    #Creates a Features_DF from the original one by dropping Classification Column
    tr_features_copy=tr
    tr_features_clean= tr_features_copy.drop('Classification', axis=1)
    print('Features', tr_features_clean.columns,'\n')
    
    #Creates a Labels_DF from the original one by picking Classification Column
    tr_labels_clean=tr[['Classification']]
    #Creates an Array from Labels_DF
    tr_labels=tr_labels_clean.values
    tr_labels_2d = tr_labels.ravel()
    print('Labels',tr_labels_clean.columns,'\n')
    
    
    # PROCESSING
    reg_fs_demo = dynaFS_clf(custom_selectors=selectors, fs_num = features ,random_state = None,cv = cross_val,input_from_file = False)

    _,features_selected=reg_fs_demo.fit(tr_features_filled,tr_labels_2d)

if __name__=='__main__':
	main()