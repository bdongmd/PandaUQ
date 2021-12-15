import numpy as np
import h5py
import model
import lib_plotting
import tensorflow as tf
from itertools import compress
from scipy import stats

import os
import sys
import timeit
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(description='Options for doing the evaluations.')
parser.add_argument('-i', '--input', type=str, default='input/bdong_total_shuffled.h5')
parser.add_argument('-o', '--output', type=str, default='output')
parser.add_argument('-m', '--model', type=str, default='output/training_e40.h5')
args = parser.parse_args()

n_predictions = 10000 # number of predictions, got from stability study
performanceCheck = True # check simple performance without Droput enabled

#### model parameters
InputShape = 10
h_layers = [40, 25, 8]
drops = [0.3, 0.3, 0.3]
dropout=True
lr = 0.0001
batch_size = 524 

test_model = model.DLModel(InputShape=InputShape, h_layers=h_layers, lr=lr, drops=drops, dropout=False)
test_model_Droput = model.DLModel(InputShape=InputShape, h_layers=h_layers, lr=lr, drops=drops, dropout=True)
test_model.load_weights(args.model)
test_model_Droput.load_weights(args.model)

f = h5py.File(args.input, 'r')
X_test = f['X_train'][:]
labels = f['labels'][:]

nodrop = test_model.predict(X_test) # evaluaiton without Dropout
if performanceCheck:
	lib_plotting.plotOutputScore(nodrop.flatten(), labels)


#sig = np.array(list(compress(X_test, labels==1)))
#bkg = np.array(list(compress(X_test, labels==0)))
