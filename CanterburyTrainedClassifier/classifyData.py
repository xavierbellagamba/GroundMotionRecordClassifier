import GMClassifierFx as gmc
import numpy as np
import csv
import os


#########################################################################
#########################################################################
# User-defined paramters
#########################################################################
#Load data with discrete score
data_path = './data/data_test.csv'
n_row_skip = 1
n_col_skip = 3
categoricalOutput = False
hasScore=True

#Neural network name
model_name = 'GMClassifier_Cant'
#########################################################################
#########################################################################


#########################################################################
# Evaluate given database
#########################################################################
#Configure loading format
isInput = False
if hasScore:
    isInput = True
    
#Create directory if not already existing
dir_path = './Results'
if not os.path.exists(dir_path):
	os.mkdir(dir_path)

#Load data
data, y = gmc.loadCSV(data_path, row_ignore=n_row_skip, col_ignore=n_col_skip, isInput=isInput)
data = gmc.preprocessData(data)

#Neural Net
NN = gmc.neuralNet()
NN.loadNN(model_name)

y_hat = []
data_score = []
data_header = gmc.loadCSV(data_path, row_ignore=0, col_ignore=0, isInput=False, isCategorical=False)[0]
data_header.append('yhat_low')
data_header.append('yhat_high')
data_save = gmc.loadCSV(data_path, row_ignore=1, col_ignore=0, isInput=False, isCategorical=False)

for i in range(len(data)):
    y_hat.append(NN.useNN(data[i]))
    data_score.append(data_save[i] + y_hat[-1][0].tolist())

#########################################################################

#########################################################################
# Save the vector of results
#########################################################################
with open("./Results/data_test_pred.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(data_header)
    writer.writerows(data_score)
f.close()
#########################################################################
