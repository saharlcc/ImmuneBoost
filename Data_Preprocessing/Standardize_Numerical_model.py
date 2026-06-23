#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 00:50:29 2022

@author: romibarde

This script has the code for standardizing the input data used in numerical model.
Following operations are being performed in the given sequence:
    1. Normalization
    2. Data Imputation
    3. One-hot encoding mutation position column

Dec  19
    1. Added two more feature columns Score_EL_Mutated, Score_EL_WT
    
Jan 10
    1. Included all reamaining feature columns
    2. Renamed affinity and stability columns for better clarity
    
Jan 19
    1. Added separate function for one hot encoding mutation position because
    this is only needed for XGBoost. 
    (Using different technique for one hot encoding mutation position in neural network model)

Feb 1
    1. Included Foreignness feature in feature list

"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# This function normalizes and imputes the data 
def numerical_data_standardizer(dataframe):
        
    #------------------------------ Normalize Data --------------------------
    
    #normalize data
    scaler = preprocessing.MinMaxScaler()
    
    
    #Use below columns for version 8 script
    #df_to_normalize = dataframe[["Aff(nM)_Mutated", "Aff(nM)_WT", "Pred_Mutated", "Pred_WT", "Agretopicity"]]
    
    #use below columns for version 9 script
    #df_to_normalize = dataframe[["Aff(nM)_Mutated", "Score_EL_Mutated", "Score_BA_Mutated", "Aff(nM)_WT", "Score_EL_WT", "Score_BA_WT", "Pred_Mutated", "Thalf(h)_Mutated", "Pred_WT", "Thalf(h)_WT" , "Agretopicity", "Hydrophobicity" ]]
    
    #use below columns for version 10 and 11 and 14, 17 script
    df_to_normalize = dataframe[["Aff(nM)_Mutated", "Aff_Score_EL_Mutated", "Aff_Score_BA_Mutated", "Aff(nM)_WT", "Aff_Score_EL_WT", 
                                 "Aff_Score_BA_WT", "Stab_Pred_Mutated", "Stab_Thalf(h)_Mutated", "Stab_Pred_WT", "Stab_Thalf(h)_WT" , "Agretopicity",
                                 "Hydrophobicity" ]]
    
    #use below columns for version 12, 15 and 16, 17, 18 script
    #df_to_normalize = dataframe[["Aff(nM)_Mutated", "Aff_Score_EL_Mutated", "Aff_Score_BA_Mutated", "Aff(nM)_WT", "Aff_Score_EL_WT", 
    #                             "Aff_Score_BA_WT", "Stab_Pred_Mutated", "Stab_Thalf(h)_Mutated", "Stab_Pred_WT", "Stab_Thalf(h)_WT" , "Agretopicity",
    #                             "Hydrophobicity", "Foreignness_combined"]]
    
    col_names = df_to_normalize.columns
    transformed_arr = scaler.fit_transform(df_to_normalize)
    normalized_df = pd.DataFrame(transformed_arr, columns=col_names)
    
    #columns not normalized 
    #use below columns for version 8 and 9 script
    #df1 = dataframe[["Peptide_Length", "Mutation_Position", "Validation_Result"]]
    
    #use below columns for version 10, 11 and 12 script
    df1 = dataframe[["Aff_%Rank_EL_Mutated", "Aff_%Rank_BA_Mutated", "Aff_%Rank_EL_WT", "Aff_%Rank_BA_WT",
      "%Rank_Stab_Mutated", "%Rank_Stab_WT", "Distoself_Value", "Peptide_Length", "Mutation_Position", "Validation_Result"]]
    
    
    #axis=1 parameter indicates that it is the columns that need to be joined together    
    final_df = pd.concat([normalized_df,df1],axis=1)
    
    
    #------------------------------ Impute Missing Values --------------------------
    
    #Impute the values by class
    df_pos = final_df.loc[final_df['Validation_Result'] == 1 ]
    df_neg = final_df.loc[final_df['Validation_Result'] == 0 ]
    
    imputer = KNNImputer(n_neighbors=10)
    
    imputed_pos_arr = imputer.fit_transform(df_pos)
    imputed_neg_arr = imputer.fit_transform(df_neg)
    
    imputed_arr = np.vstack((imputed_pos_arr,imputed_neg_arr))
    
    imputed_df = pd.DataFrame(imputed_arr, columns=final_df.columns)
    #print("imputed_df:",imputed_df.columns)
    
    return imputed_df
    

# This function onehot encodes mutation position for XGBoost model
def onehot_mut_pos_xgb(imputed_df):
    
    #------------------------------ One hot encode mutation position --------------------------
    
    
    #resampled_df.to_excel("/Users/romibarde/Library/CloudStorage/OneDrive-Personal/Southern/Research/Cancer_Vaccine_Design/working_files/Datasets/XGB_Resampled_Dataset.xlsx","Sheet1", index=False)
    
    #one hot encode the column
    ohe = OneHotEncoder()
    transformed_mutpos = ohe.fit_transform(imputed_df[['Mutation_Position']])
    
    transformed_mutpos_arr = transformed_mutpos.toarray()
    #print(transformed_mutpos_arr[0])
    print("Mutation Position categories after one-hot encoding: ", ohe.categories_[0])
    
    
    #------------------------------ Add one hot mutation position category columns to dataframe --------------------------
    
    #create one-hot category column names list to be added to dataframe
    cat_arr = ohe.categories_[0].astype(int)
    
    one_hot_cols = []
    for i in range(0,len(cat_arr)):
        one_hot_cols.append("Mut_Pos_" + str(cat_arr[i]))
    
    #add one-hot encoded column to dataframe
    #it will create new columns in dataframe for one-hot categories
    imputed_df[one_hot_cols] = transformed_mutpos.toarray()
    
    
    #create new dataframe to have target column at the end with renamed onehot encoded mut_pos category columns and Mutation_Position column dropped
    
    #use below columns for version 8 script
    '''
    rearranged_cols = ['Aff(nM)_Mutated', 'Aff(nM)_WT', 'Pred_Mutated' ,'Pred_WT', 'Agretopicity',
                       'Peptide_Length', 'Mut_Pos_1', 'Mut_Pos_2', 'Mut_Pos_3', 'Mut_Pos_4', 'Mut_Pos_5', 'Mut_Pos_6', 'Mut_Pos_7', 
                       'Mut_Pos_8', 'Mut_Pos_9', 'Mut_Pos_10', 'Mut_Pos_11', 'Mut_Pos_13', 'Mut_Pos_14','Validation_Result']
    '''
    
    
    #use below columns for version 9 script
    '''
    rearranged_cols = ['Aff(nM)_Mutated', 'Score_EL_Mutated', 'Score_BA_Mutated', 'Aff(nM)_WT', 'Score_EL_WT', 
                       'Score_BA_WT' ,'Pred_Mutated','Thalf(h)_Mutated' ,'Pred_WT', 'Thalf(h)_WT' ,'Agretopicity', 'Hydrophobicity',
                       'Peptide_Length', 'Mut_Pos_1', 'Mut_Pos_2', 'Mut_Pos_3', 'Mut_Pos_4', 'Mut_Pos_5', 'Mut_Pos_6', 'Mut_Pos_7', 
                       'Mut_Pos_8', 'Mut_Pos_9', 'Mut_Pos_10', 'Mut_Pos_11', 'Mut_Pos_13', 'Mut_Pos_14','Validation_Result']
    '''
    
    #use below columns for version 10 and 11 script
    #34 columns total
    
    rearranged_cols = ['Aff_Score_EL_Mutated', 'Aff_%Rank_EL_Mutated','Aff_Score_BA_Mutated', 'Aff_%Rank_BA_Mutated', 'Aff(nM)_Mutated', 
            'Aff_Score_EL_WT', 'Aff_%Rank_EL_WT', 'Aff_Score_BA_WT', 'Aff_%Rank_BA_WT', 'Aff(nM)_WT', 'Stab_Pred_Mutated', 'Stab_Thalf(h)_Mutated', 
            '%Rank_Stab_Mutated', 'Stab_Pred_WT', 'Stab_Thalf(h)_WT', '%Rank_Stab_WT', 'Agretopicity', 'Hydrophobicity', 'Distoself_Value', 
            'Peptide_Length', 'Mut_Pos_1', 'Mut_Pos_2', 'Mut_Pos_3', 'Mut_Pos_4', 'Mut_Pos_5', 'Mut_Pos_6', 'Mut_Pos_7', 
            'Mut_Pos_8', 'Mut_Pos_9', 'Mut_Pos_10', 'Mut_Pos_11', 'Mut_Pos_13', 'Mut_Pos_14', 'Validation_Result']
    
    
    #use below columns for version 12, 15 and 16, 17, 18 script
    #35 columns total
    '''
    rearranged_cols = ['Aff_Score_EL_Mutated', 'Aff_%Rank_EL_Mutated','Aff_Score_BA_Mutated', 'Aff_%Rank_BA_Mutated', 'Aff(nM)_Mutated', 
            'Aff_Score_EL_WT', 'Aff_%Rank_EL_WT', 'Aff_Score_BA_WT', 'Aff_%Rank_BA_WT', 'Aff(nM)_WT', 'Stab_Pred_Mutated', 'Stab_Thalf(h)_Mutated', 
            '%Rank_Stab_Mutated', 'Stab_Pred_WT', 'Stab_Thalf(h)_WT', '%Rank_Stab_WT', 'Agretopicity', 'Hydrophobicity', 'Distoself_Value', 
            'Peptide_Length', "Foreignness_combined", 'Mut_Pos_1', 'Mut_Pos_2', 'Mut_Pos_3', 'Mut_Pos_4', 'Mut_Pos_5', 'Mut_Pos_6', 'Mut_Pos_7', 
            'Mut_Pos_8', 'Mut_Pos_9', 'Mut_Pos_10', 'Mut_Pos_11', 'Mut_Pos_13', 'Mut_Pos_14', 'Validation_Result']
    '''
    
    rearranged_df = imputed_df[rearranged_cols]
    
    
    (rearranged_df.Mut_Pos_11 == 1).sum()
    
    return rearranged_df

    