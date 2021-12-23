from json import dump, load
from numpy.core.fromnumeric import compress
import uproot3 as up
import numpy as np
import h5py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pickle import dump, load

#from var_mapping import mapping
import sys

from uproot3.source.compressed import Compression
sys.path.append('../')
import lib_plotting

import argparse

parser = argparse.ArgumentParser(description='Add input and output files.')
parser.add_argument('-s', '--sigfile', type=str,
	default='/Users/binbindong/Desktop/work/PandaX/PandaUQ/input/simu_2hit_bdt.root')
parser.add_argument('-b', '--bkgfile', type=str,
	default='/Users/binbindong/Desktop/work/PandaX/PandaUQ/input/total_pair_test_2hit.root')
parser.add_argument('--s2', type=str)
parser.add_argument('--saveScale', action='store_true')
parser.add_argument('-c', '--config', type=str)
parser.add_argument('-o', '--outputfile', type=str)
args = parser.parse_args()

signal = up.open(args.sigfile)['miniTree']
bkg = up.open(args.bkgfile)['miniTree']

#var_list = list(mapping.keys())

df_S = signal.pandas.df()
df_B = bkg.pandas.df()
if args.s2:
	sig2 = up.open(args.s2)['miniTree']
	df_S2 = sig2.pandas.df()
	lib_plotting.variable_plotting(df_S, df_B, sig2=df_S2, noname=False, variables=args.config, outputFile='plots/inputCompare.pdf')

######## plotting input variables #########
## note: no scaling is applied here
lib_plotting.variable_plotting(df_S, df_B, noname=False, variables=args.config, outputFile='plots/inputVar_noscale.pdf')

sWeight = 25.5/73010
bWeight = 456.9/946892
X_train = np.concatenate((pd.DataFrame(df_B), pd.DataFrame(df_S)))
labels = np.concatenate((np.zeros(len(df_B), dtype=int), np.ones(len(df_S), dtype=int)))
weight = np.concatenate((np.ones(len(df_B))*bWeight, np.ones(len(df_S))*sWeight))
print("X_train shape: {}".format(X_train.shape))

scaler = StandardScaler()
scaler.fit(X_train)
if args.saveScale:
	scaler.fit(X_train)
	dump(scaler, open('VariableScaler.pkl', 'wb'))
else:
	scaler = load(open('VariableScaler.pkl', 'rb'))
X_train = scaler.transform(X_train)
lib_plotting.variable_plotting(pd.DataFrame(scaler.transform(df_S)), pd.DataFrame(scaler.transform(df_B)), noname=True, variables=args.config, outputFile='plots/inputVar_scaled.pdf')

rng_state = np.random.get_state()
np.random.shuffle(X_train)
np.random.set_state(rng_state)
np.random.shuffle(labels)
np.random.set_state(rng_state)
np.random.shuffle(weight)
#assert X_train.shape[1] == 10 ## number of input variables 
print('Training input shape: {}'.format(X_train.shape))

outputfile = h5py.File(args.outputfile, 'w')
outputfile.create_dataset('X_train', data=X_train, compression='gzip')
outputfile.create_dataset('labels',  data=labels,  compression='gzip')
outputfile.create_dataset('weight', data=weight, compression='gzip')
outputfile.close()
