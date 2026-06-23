#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 22:30:05 2023

@author: romibarde

This script splits the data into train, val and test sets in the ratio 80:10:10
using train_test_split method

"""

from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(df_to_split, start_col, end_col):
    
    datasplits=[]
    
    X = df_to_split.iloc[:, start_col:end_col]   #feature columns
    y = df_to_split.iloc[:, end_col]      # target column

    train_ratio = 0.80
    validation_ratio = 0.10
    test_ratio = 0.10

    # train is now 80% of the entire data set
    x_train_df, x_remaining_df, y_train_arr, y_remaining = train_test_split(X, y, test_size=1 - train_ratio, stratify=y, random_state=0)

    # test is now 10% of the initial data set
    # validation is now 10% of the initial data set
    x_val_df, x_test_df, y_val, y_test = train_test_split(x_remaining_df, y_remaining, test_size=test_ratio/(test_ratio + validation_ratio), stratify=y_remaining, random_state=0)

    '''    
    #Combine the x_test and y_test to generate common test set for comparison
    y_test_df = pd.DataFrame(y_test)
    combined_test_df = pd.concat([x_test_df,y_test_df],axis=1)
    #print("combined_test_df:", combined_test_df)
    
    combined_test_df.to_excel("/Users/romibarde/Library/CloudStorage/OneDrive-Personal/Southern/Research/Cancer_Vaccine_Design/working_files/Datasets/Analysis/New/Old_Test_Dataset.xlsx","Sheet1", index=False)
    
    #combine train and validation
    y_train_df = pd.DataFrame(y_train_arr)
    combined_train_df = pd.concat([x_train_df,y_train_df],axis=1)
    
    y_val_df = pd.DataFrame(y_val)
    combined_val_df = pd.concat([x_val_df,y_val_df],axis=1)
    
    remaining_df = pd.concat([combined_train_df,combined_val_df],axis=0)
    
    remaining_df.to_excel("/Users/romibarde/Library/CloudStorage/OneDrive-Personal/Southern/Research/Cancer_Vaccine_Design/working_files/Datasets/Analysis/New/Old_Remaining_Dataset.xlsx","Sheet1", index=False)
    '''
    
    #add splits to a list to be used in the models
    datasplits.append(x_train_df)
    datasplits.append(y_train_arr)
    datasplits.append(x_val_df)
    datasplits.append(x_test_df)
    datasplits.append(y_val)
    datasplits.append(y_test)
    
    return datasplits