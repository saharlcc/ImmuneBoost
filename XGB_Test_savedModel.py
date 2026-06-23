#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 17:55:49 2023

@author: romibarde
"""

import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score,  f1_score, precision_score, recall_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as pyplot


#----------------------------- Load the model -----------------------------

path="./"
filename = path + "finalized_xbg_model_without_foreigness.sav"

# load the model from disk
model = pickle.load(open(filename, 'rb'))


#----------------------------- Load test data -----------------------------

#test_df = pd.read_excel("/Users/romibarde/Library/CloudStorage/OneDrive-Personal/Southern/Research/Cancer_Vaccine_Design/working_files/Datasets/My_Data_Splits/test_data.xlsx", sheet_name="Sheet1")
#test_df = pd.read_csv("./TEST_DATA3.csv")
test_df = pd.read_excel("./TEST_DATA3.xlsx", sheet_name="Sheet1")

x_test_df = test_df.iloc[:,2:35]
y_test = test_df.iloc[:,35]
#x_test_df = test_df.iloc[:,31:64]
#y_test = test_df.iloc[:,64]

x_test = x_test_df.to_numpy()

#----------------------------- Making predictions ----------------------

# make class predictions with the model
y_pred = model.predict(x_test)
predictions = (y_pred > 0.5).astype(int)

test_accuracy = accuracy_score(y_test, predictions)
test_f1_score = f1_score(y_test, predictions)
test_precision = precision_score(y_test, predictions)
test_recall = recall_score(y_test, predictions)

print('\nTest accuracy =',test_accuracy)
print("Precision = ", test_precision)
print("Recall = ", test_recall)
print("F1 Score = ", test_f1_score)

print("\n\n",classification_report(y_test, predictions, labels=[0, 1]))


# summarize the first 5 cases
#for i in range(23):
#    print('%s => %d (expected %d)' % (x_test[i].tolist(), predictions[i], y_test[i]))

cm = confusion_matrix(y_test, predictions)
print(cm) 

#from sklearn.metrics import precision_recall_fscore_support as score
#precision, recall, fscore, support = score(y_test, predictions)

#true negatives is C(0,0), false negatives is C(1,0) 
#true positives is C(1,1) and false positives is C(0,1)

#specificity =  tn / (tn + fp)
#sensitivity = recall  = tp / (tp + fn)

tn, fp, fn, tp = cm.ravel()

print("\nTrue Negatives: ", tn)
print("False Positives: ", fp)
print("False Negatives: ", fn)
print("True Positives: ", tp)

test_specificity = tn / (tn + fp)
print('Specificity : ', test_specificity)

cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])

cm_display.plot()
pyplot.show()

test_df["y_pred"] = y_pred.flatten()
test_df["prediction"] = predictions.flatten()
test_df.to_csv("./TEST_DATA3_with_predictions.csv", index=False)