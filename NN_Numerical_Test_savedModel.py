#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 17:55:49 2023

@author: romibarde
"""

#from keras.models import load_model
from keras.models import load_model
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score,  f1_score, precision_score, recall_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as pyplot
from keras import backend as K

#------------------------------ Define evaluation metric functions --------------------------

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


#----------------------------- Load the model -----------------------------

#path="/Users/romibarde/Library/CloudStorage/OneDrive-Personal/PythonScripts/ThesisScripts/Saved_Best_Models/"
path="./"
filename = path + "finalized_nn_numerical_model_without_foreigness.h5"


# load the model from disk
#model = load_model(filename, custom_objects={'f1_m': f1_m,'recall_m': recall_m, 'precision_m': precision_m})
model = load_model(filename, compile=False)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[precision_m, recall_m, f1_m]
)

#----------------------------- Load test data -----------------------------

test_df = pd.read_csv("./TEST_DATA3.csv")


x_test_df = test_df.iloc[:,2:35]
y_test = test_df.iloc[:,35]
print("y_test")
print(y_test)

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