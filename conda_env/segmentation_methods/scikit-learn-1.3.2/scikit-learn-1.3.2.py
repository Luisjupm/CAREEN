# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 19:42:08 2023

@author: LuisJa
"""
import pandas as pd

import sklearn

from sklearn.pipeline import *
from sklearn.ensemble import *
from sklearn.ensemble import *
from sklearn.kernel_approximation import *
from sklearn.naive_bayes import *
from sklearn.neural_network import *
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
from tpot import TPOTClassifier
from tpot import *
import tpot

import matplotlib.pyplot as plt
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.model_selection import FeatureImportances

import joblib

import argparse

import os
import yaml

def main():
    
    #%% CMD
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--i',type=str,help='Yaml configuration file')
    # parser.add_argument('--o',type=str,help='Output_directory')    
    # args=parser.parse_args() 
    
    #%% INITIAL READING
    with open(r"C:\Users\LuisJa\Desktop\logistic_regression\algorithm_configuration.yaml", 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    algo =  config_data.get('ALGORITHM')
    output_directory= config_data.get('OUTPUT_DIRECTORY')
    if algo=='Prediction':
        
        input_file= config_data.get('INPUT_POINT_CLOUD')
        features2include= config_data['CONFIGURATION']['f']   
        pkl_file=config_data['CONFIGURATION']['p']    
        
        print("The input file is taken from " + str(input_file))  
        print("The features are taken from " + str(features2include)) 
        print("The pkl file is taken from " + str(pkl_file)) 
        print("Output directory is " + str(output_directory))
        
        # Store in a Pandas dataframe the content of the file
        pcd_testing=pd.read_csv(input_file,delimiter=' ')
        with open(features2include, "r") as file:
            f2i = [line.strip().split(',') for line in file]    
        X_test= pcd_testing[f2i[0]].ffill()
        
        # Load the model from the file
        model = joblib.load(pkl_file)
        
    else:
        
        # Read the neccesary information from the YAML file
        test_file= config_data.get('INPUT_POINT_CLOUD_TESTING')
        train_file= config_data.get('INPUT_POINT_CLOUD_TRAINING')
        features2include_path= config_data.get('INPUT_FEATURES')
              
        # Store in a Pandas dataframe the content of the files
        pcd_training=pd.read_csv(train_file,delimiter=' ')        
        pcd_testing=pd.read_csv(test_file,delimiter=' ')
        
        # Clean the dataframe, and drop all the line that contains a NaN (Not a Number) value.
        pcd_training.dropna(inplace=True)
        pcd_testing.dropna(inplace=True)
        
        # Extracting the classification labels for training and testing
        labels2include=['Classification']
        labels_training=pcd_training[labels2include]
        labels_testing=pcd_testing[labels2include]
        
        # Extracting the features of interest for training and testing        
        with open(output_directory + "\\features.txt", "r") as file:
            features2include = [line.strip().split(',') for line in file]    
        features=pcd_training[features2include[0]]
        features_training=features    
        features=pcd_testing[features2include[0]]        
        features_testing=features
        
        # Creation of the matrices for running the algorithm
        X_test=features_testing
        y_test=labels_testing.to_numpy()
        X_train=features_training
        y_train=labels_training.to_numpy()
        
    #%% SPECIFIC PARAMETERS FOR EACH ALGORITHM
        
    if algo=="Random_forest":        
        
        ne=config_data['CONFIGURATION']['ne']
        c=config_data['CONFIGURATION']['c']
        md=config_data['CONFIGURATION']['md']
        ms=config_data['CONFIGURATION']['ms']
        mns=config_data['CONFIGURATION']['mns']
        mwf=config_data['CONFIGURATION']['mwf']
        mf=config_data['CONFIGURATION']['mf']
        bt=config_data['CONFIGURATION']['bt']
        n_jobs=config_data['CONFIGURATION']['nj'] 
        
        # Restrictions
        if not bt:
            obb = False
        else:
            obb = True  
            
        print("Test file located in " + test_file)        
        print("Train file located in " + train_file)        
        print("Output directory is " + output_directory)        
        print("Features to include = " + features2include_path)        
        print("Number of estimators = " + str(ne))        
        print("Criterion chosen = " + str(c))        
        print("Maximum depth of the tree = " + str(md))        
        print("Minimum number of samples required to split an internal node = " + str(ms))        
        print("Minimum number of samples required to be at a leaf node = " + str(mns))        
        print("Minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node = " + str(mwf))        
        print("Number of features to consider = " + str(mf))        
        print("Bootstrap = " + str(bt)) 
        print("Number of jobs = " + str(n_jobs))
        
  
        model = RandomForestClassifier(
            n_estimators=ne,
            max_depth=md,
            max_features=mf,
            random_state=42,
            oob_score=obb,
            bootstrap=bt,
            criterion=c,
            min_samples_split=ms,
            min_samples_leaf=mns,
            min_weight_fraction_leaf=mwf,
            n_jobs=n_jobs
            )
        
    elif algo=="Support Vector Machine":
        
        C=config_data['CONFIGURATION']['C']
        kernel=config_data['CONFIGURATION']['kernel']
        degree=config_data['CONFIGURATION']['degree']
        gamma=config_data['CONFIGURATION']['gamma']
        coef0=config_data['CONFIGURATION']['coef0']
        shrinking=config_data['CONFIGURATION']['shrinking']
        probability=config_data['CONFIGURATION']['probability']
        tol=config_data['CONFIGURATION']['tol']
        class_weight=config_data['CONFIGURATION']['class_weight']
        max_iter=config_data['CONFIGURATION']['max_iter']
        decision_function_shape=config_data['CONFIGURATION']['decision_function_shape']
        break_ties=config_data['CONFIGURATION']['break_ties']     

        # Restrictions
        if C<0:
            C=0
        if degree<=0:
            degree=1
        if shrinking=="True":
            shrinking=True
        else:
            shrinking=False 
        if class_weight=="No":
            class_weight=None
        if decision_function_shape=="ovo":
            decision_function_shape="ovr"
        if break_ties=="True":
            break_ties=True
        else:
            break_ties=False
        if probability=="True":
            probability=True
        else:
            probability=False
            
        print("Test file located in " + test_file)        
        print("Train file located in " + train_file)        
        print("Output directory is " + output_directory)        
        print("Features to include = " + features2include_path)        
        print("Regularization parameter = " + str(C))        
        print("Kernel type = " + kernel)      
        print("Degree of the polynomial kernel function = " + str(degree))        
        print("Kernel coefficient = " + gamma)     
        print("Independent ter in the kernel function = " + str(coef0))       
        print("Use the shirinking herustics = " + str(shrinking))        
        print("Enable probability estimation = " + str(probability))        
        print("Tolerace for stopping criterios = " + str(tol)) 
        print("Weights for the classes = " + str(class_weight))
        print("Maximum number of iterations = " + str(max_iter))        
        print("Function shape to be returned= " + decision_function_shape)        
        print("Break ties = " + str(break_ties)) 
      
        model= sklearn.svm.SVC(
            C=C,kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            class_weight=class_weight,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties)
        
    elif algo=="Logistic Regression":
        
        penalty=config_data['CONFIGURATION']['penalty']
        dual=config_data['CONFIGURATION']['dual']
        tol=config_data['CONFIGURATION']['tol']
        C=config_data['CONFIGURATION']['c']
        fit_intercept=config_data['CONFIGURATION']['fit_intercept']
        intercept_scaling=config_data['CONFIGURATION']['intercept_scaling']
        class_weight=config_data['CONFIGURATION']['class_weight']
        solver=config_data['CONFIGURATION']['solver']
        max_iter=config_data['CONFIGURATION']['max_iter']
        multi_class=config_data['CONFIGURATION']['multi_class']
        l1_ratio=config_data['CONFIGURATION']['l1_ratio']
        
        # Restrictions
        if penalty=="No":
            penalty=None
        if dual=='True' and penalty=="l2" and solver=="liblinear":
            dual=True
        else:
            dual=False
        if fit_intercept=='True':
            fit_intercept=True
        else:
            fit_intercept=False
        if class_weight=="No":
            class_weight=None
        if solver=="lbfgs" and (penalty=="l2" or penalty==None):
            pass
        elif solver=="liblinear" and (penalty=="l2" or penalty=="l1"):
            pass
        elif solver=="newton-cg" and (penalty=="l2" or penalty==None):
            pass
        elif solver=="newton-cholesky" and (penalty=="l2" or penalty==None):
            pass
        elif solver=="sag" and (penalty=="l2" or penalty==None):
            pass
        else:
            solver="saga"
        
        if multi_class=="multinomial" and solver=="liblinear":
            multi_class=='auto'
        if l1_ratio<0:
            l1_ratio=0
        elif l1_ratio>1:
            l1_ratio=1
        
        print("Test file located in " + test_file)        
        print("Train file located in " + train_file)        
        print("Output directory is " + output_directory)        
        print("Features to include = " + features2include_path)        
        print("Norm of the penalty = " + str(penalty))        
        print("Constrained formulation = " + str(dual))      
        print("Inverse of regularization strength = " + str(C))        
        print("Scaling of the intercept = " + str(intercept_scaling))
        print("Classes weights = " + str(class_weight))
        print("Solver used = " + str(solver))
        print("Maximum number of iterations = " + str(max_iter))
        print("Multiclass strategy = " + multi_class)
        print("Elastic-Net mixing paramter = " + str(l1_ratio)) 
        
        model=sklearn.linear_model.LogisticRegression(
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            l1_ratio=l1_ratio)
        
    #%% RUNNING THE MODEL AND PREDICT THE RESULTS
    if algo != "Prediction":
        model.fit(X_train,y_train.ravel())
    y_pred = model.predict(X_test)  
    
    #%% SAVE THE MODEL
    if algo != "Prediction":
        joblib.dump(model, os.path.join(output_directory, 'model.pkl'))
    
    #%% CREATION OF THE CONFUSION MATRIX
    
        cm= ConfusionMatrix(model, cmap='Blues')
        cm.score (X_test,y_test)
        cm.show(outpath=os.path.join(output_directory, 'confusion_matrix.png'))  # Save the confusion matrix to a file        
        plt.close()  # Close the plot 
        
    #%% CREATION OF THE CLASSIFICATION REPORT
    
        report=classification_report(y_test, y_pred)
        # Write the report to a file
        with open(os.path.join(output_directory,'classification_report.txt'), 'w') as file:
            file.write(report)

    #%% CREATION OF THE IMPORTANCE GRAPH IN CASE OF CHOSING RANDOM FOREST
    if algo == "Random_forest":
        # Determine the number of features for automatic sizing
        num_features = len(X_train.columns)
        height = min(num_features * 0.3, 100000)  # Limit maximum height to 10000 inches, adjust multiplier as needed

        # Set the size of the figure dynamically based on the number of features
        plt.figure(figsize=(10, height))  # Set width to 10 inches and adjust height dynamically
        # Create the FeatureImportances visualizer with the trained classifier
        viz = FeatureImportances(model)

        # Fit the visualizer to your data
        viz.fit(X_train, y_train.ravel())
        
        # Save the importance graph
        viz.show(outpath=os.path.join(output_directory,'feature_importance.png'))
        plt.close()  # Close the plot
        
        # Extract feature importances
        feature_importances = viz.feature_importances_
        
        # Create a DataFrame with feature names and importances
        importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
        
        # Save to a text file separated by tab
        output_file = os.path.join(output_directory, 'feature_importance.txt')
        importance_df.to_csv(output_file, sep='\t', index=False)
    #%% CREATION OF THE FINAL POINT CLOUD WITH THE PREDICTIONS FOR FURTHER LOADING
    
    pcd_testing_subset = pcd_testing[['X', 'Y', 'Z']].copy()
    pcd_testing_subset['Predictions'] = y_pred
    # Saving the DataFrame to a CSV file
    pcd_testing_subset.to_csv(os.path.join(output_directory, 'predictions.txt'), index=False)       
        
    
if __name__=='__main__':
	main()