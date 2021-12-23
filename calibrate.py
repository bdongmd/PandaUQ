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
parser.add_argument('-s', '--s2', type=str)
parser.add_argument('-o', '--output', type=str, default='output')
parser.add_argument('-m', '--model', type=str, default='output/training_e40.h5')
args = parser.parse_args()

n_predictions = 10000 # number of predictions, got from stability study
performanceCheck = False # check simple performance without Droput enabled
compareTwoSig = True and args.s2 # if comapre performance between two signals

#### model parameters
InputShape = 10
h_layers = [20, 15, 8]
#h_layers = [40, 25, 8]
drops = [0.2, 0.2, 0.2]
dropout=True
lr = 0.0005
batch_size = 524 

test_model = model.DLModel(InputShape=InputShape, h_layers=h_layers, lr=lr, drops=drops, dropout=False)
test_model_Droput = model.DLModel(InputShape=InputShape, h_layers=h_layers, lr=lr, drops=drops, dropout=True)
test_model.load_weights(args.model)
test_model_Droput.load_weights(args.model)

f = h5py.File(args.input, 'r')
if args.s2:
	f2 = h5py.File(args.s2, 'r')
if compareTwoSig:
	labels = f['labels'][:]
	X_test = f['X_train'][:]
	X_test = X_test[labels==1]
	sig2_label = f2['labels'][:]
	sig2_test = f2['X_train'][:] 
	sig2_test = sig2_test[sig2_label==1]
else:
	halfEvents = int(len(f['X_train'])*0.7)
	X_test = f['X_train'][:halfEvents]
	labels = f['labels'][:halfEvents]

nodrop = test_model.predict(X_test) # evaluaiton without Dropout
if compareTwoSig:
	s2_nodrop = test_model.predict(sig2_test)
	lib_plotting.compare2Sig(nodrop, s2_nodrop, 'output/')

if performanceCheck:
	lib_plotting.plotOutputScore(nodrop.flatten(), labels)


#sig = np.array(list(compress(X_test, labels==1)))
#bkg = np.array(list(compress(X_test, labels==0)))
