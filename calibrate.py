import numpy as np
import h5py
from numpy.lib.npyio import save
import model
import lib_plotting
from tqdm import tqdm
import utils

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
performanceCheck = False # check simple performance without Droput enabled if True, else not.
compareTwoSig = True and args.s2 # if comapre performance between two signals
cut = 0.9
saveDropoutScore = False # if True save output score with Dropout enabled, else no.

#### model parameters
InputShape = 23
h_layers = [60, 30, 15, 8]
drops = [0.2, 0.2, 0.2, 0.2]
dropout=True
lr = 0.0005
batch_size = 524 

#### model loading
test_model = model.DLModel(InputShape=InputShape, h_layers=h_layers, lr=lr, drops=drops, dropout=False)
test_model_Dropout = model.DLModel(InputShape=InputShape, h_layers=h_layers, lr=lr, drops=drops, dropout=True)
test_model.load_weights(args.model)
test_model_Dropout.load_weights(args.model)

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
	halfEvents = 156380
	X_test = f['X_train'][halfEvents:]
	labels = f['labels'][halfEvents:]

nodrop = test_model.predict(X_test) # evaluaiton without Dropout
if compareTwoSig:
	s2_nodrop = test_model.predict(sig2_test)
	lib_plotting.compare2Sig(nodrop, s2_nodrop, '{}/'.format(args.output))

elif performanceCheck:
	lib_plotting.plotOutputScore(nodrop.flatten(), labels)
else:
	objAcc = []
	median = []
	uncer = []
	significance = []
	score_drop = []
	init = timeit.default_timer()
	for j in tqdm(range(int(X_test.size / InputShape)), desc="===== Evaluating each objects"):
		input_data = X_test[j]
		if j < 10:
			sanityCheck = True
		else:
			sanityCheck = False
		tmpAcc, tmpSig, tmpUncer, tmpMedian, tmpScore = utils.evaluation(model=test_model_Dropout, input_data=input_data, n_predictions=n_predictions, cut=cut, title='event_{}'.format(str(j)), sanityCheck=sanityCheck)

		objAcc.append(tmpAcc)
		median.append(tmpMedian)
		significance.append(tmpSig)
		uncer.append(tmpUncer)
		if saveDropoutScore:
			score_drop.append(tmpScore)
	probability = stats.norm.cdf(significance)
	final = timeit.default_timer()
	print("===== Time used to evaluate ojects: {}s.".format(final-init))

	fout = h5py.File('{}/DUQ_out.h5'.format(args.output), 'w')
	fout.create_dataset('probability', data=np.array(probability))
	fout.create_dataset('acc', data=np.array(objAcc))
	fout.create_dataset('median', data=np.array(median))
	fout.create_dataset('uncertainty', data=np.array(uncer))
	fout.create_dataset('labels', data=np.array(labels))
	fout.create_dataset('score_notDrop', data=np.array(nodrop))
	if saveDropoutScore:
		fout.create_dataset('score', data=np.array(score_drop))
	fout.close()

	lib_plotting.plot_DUQ(labels=np.array(labels), var1=np.array(probability), x_label="predicted probability", title="prob")
	lib_plotting.plot_DUQ(labels=np.array(labels), var1=np.array(uncer), x_label="uncertainty", title="uncertainty")
	lib_plotting.plot_DUQ(labels=np.array(labels), var1=np.array(median), x_label="median", title="median")
	acc_bins = np.linspace(0,1,100).tolist()
	lib_plotting.plot_DUQ(var1=np.array(nodrop).flatten(), var2=np.array(objAcc), x_label="output score", y_label="accuracy", bins1=acc_bins, bins2=acc_bins, title="acc_vs_score")
	lib_plotting.plot_DUQ(var1=np.array(probability).flatten(), var2=np.array(objAcc), x_label="DUQ predicted score", y_label="accuracy", bins1=acc_bins, bins2=acc_bins, title="acc_vs_duq")