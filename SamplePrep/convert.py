from json import dump, load
from keras.utils import np_utils
import uproot3 as up
import numpy as np
import h5py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pickle import dump, load

from var_mapping import mapping
import sys

sys.path.append('../')
import lib_plotting

import argparse

def get_max(tree, branch, index):
	max_value = []
	var = tree.array(branch)
	ith = tree.array(index)
	for i in range(tree.numentries):
		max_value.append(var[i][ith[i]])
	return max_value
vector_varible = ["qS2Tenth", "qS2FWHM1", "qS2FWHM2","qS2FWHM3", "widthTenS2", "wS2CDF5", "wS2CDF50", "wS2CDF90", "wS2FWHM", "nPMTS2", "nPeakS2", "ratioqS2PrePeak", "qS2maxHitCharge", "qS2maxChannelCharge", "qS2BmaxHitCharge", "qS2hitStdev", "qS2channelStdev"]

parser = argparse.ArgumentParser(description='Add input and output files.')
parser.add_argument('-b', '--bkgfile', type=str,
	default='/Users/binbindong/Desktop/work/PandaX/PandaUQ/input/total_pair_test_2hit.root')
parser.add_argument('-s', '--sigfile', type=str,
	default='/Users/binbindong/Desktop/work/PandaX/PandaUQ/input/simu_2hit_bdt.root')
parser.add_argument('--s2', type=str)
parser.add_argument('--saveScale', action='store_true')
parser.add_argument('-c', '--config', type=str)
parser.add_argument('-o', '--outputfile', type=str)
args = parser.parse_args()

saveSig2 = True and args.s2 # Save s2 for training if True, else only for comparing the distributions between two distributions but do not save.

signal = up.open(args.sigfile)["event_tree"]
bkg = up.open(args.bkgfile)["event_tree"]

var_list = list(mapping.keys())

## load scalar variables
df_S = signal.pandas.df(var_list)
df_B = bkg.pandas.df(var_list)
## load vectorized variables
print("======= Processing to get max value of the vectorized variables")
for i in range(len(vector_varible)):
	print("::: {}_max".format(vector_varible[i]))
	df_S["{}_max".format(vector_varible[i])] = get_max(signal, vector_varible[i], index='iS2_max')
	df_B["{}_max".format(vector_varible[i])] = get_max(bkg, vector_varible[i], index='iS2_max')
df_S = df_S.dropna()
df_B = df_B.dropna()

print("======= Signal shape: {}".format(df_S.shape))
print("======= Background shape: {}".format(df_B.shape))

if args.s2:
	sig2 = up.open(args.s2)['event_tree']
	df_S2 = sig2.pandas.df(var_list)
	for i in range(len(vector_varible)):
		df_S2["{}_max".format(vector_varible[i])] = get_max(sig2, vector_varible[i], index='iS2_max')
	df_S2 = df_S2.dropna()
	print("======= Second signal shape: {}".format(df_S2.shape))

######## plotting input variables #########
## note: no scaling is applied here
if args.s2:
	lib_plotting.variable_plotting(df_S, df_B, sig2=df_S2, noname=False,variables=args.config, outputFile='plots/inputVar_noscale.pdf')
else:
	lib_plotting.variable_plotting(df_S, df_B, noname=False, variables=args.config, outputFile='plots/inputVar_noscale.pdf')

sWeight = 10. 
bWeight = 1.
if saveSig2:
	X_train = np.concatenate((pd.DataFrame(df_B), pd.DataFrame(df_S), pd.DataFrame(df_S2)))
else:
	X_train = np.concatenate((pd.DataFrame(df_B), pd.DataFrame(df_S)))
print("======= X_train shape before scaling: {}".format(X_train.shape))

scaler = StandardScaler()
if args.saveScale:
	scaler.fit(X_train)
	dump(scaler, open('VariableScaler.pkl', 'wb'))
else:
	scaler = load(open('VariableScaler.pkl', 'rb'))
X_train = scaler.transform(X_train)
df_S = pd.DataFrame(scaler.transform(df_S[:20])).dropna()
df_B = pd.DataFrame(scaler.transform(df_B)).dropna()

if saveSig2:
	X_train = np.concatenate((df_B, df_S, df_S2))
	labels = np.concatenate((np.zeros(len(df_B), dtype=int), np.ones(len(df_S), dtype=int), 2*np.ones(len(df_S2), dtype=int)))
	Y_train = np_utils.to_categorical(labels, 3, dtype=int)
	s2Weight = 10.
	weight = np.concatenate((np.ones(len(df_B))*bWeight, np.ones(len(df_S))*sWeight, np.ones(len(df_S2)*s2Weight)))
else:
	X_train = np.concatenate((df_B, df_S))
	labels = np.concatenate((np.zeros(len(df_B), dtype=int), np.ones(len(df_S), dtype=int)))
	weight = np.concatenate((np.ones(len(df_B))*bWeight, np.ones(len(df_S))*sWeight))
print("======= X_train shape after scaling: {}".format(X_train.shape))
print("======= Y_train shape after scaling: {}".format(labels.shape))

#if args.s2:
#	lib_plotting.variable_plotting(df_S, df_B, df_S2, noname=True, variables=args.config, outputFile='plots/inputVar_scaled.pdf')
#else:
#	lib_plotting.variable_plotting(df_S, df_B, noname=True, variables=args.config, outputFile='plots/inputVar_scaled.pdf')

rng_state = np.random.get_state()
np.random.shuffle(X_train)
np.random.set_state(rng_state)
np.random.shuffle(labels)
np.random.set_state(rng_state)
np.random.shuffle(weight)
if saveSig2:
	np.random.set_state(rng_state)
	np.random.shuffle(Y_train)
#assert X_train.shape[1] == 10 ## number of input variables 

outputfile = h5py.File(args.outputfile, 'w')
outputfile.create_dataset('X_train', data=X_train, compression='gzip')
outputfile.create_dataset('labels',  data=labels,  compression='gzip')
outputfile.create_dataset('weight', data=weight, compression='gzip')
if saveSig2:
	outputfile.create_dataset('Y_train', data=Y_train, compression='gzip')
outputfile.close()