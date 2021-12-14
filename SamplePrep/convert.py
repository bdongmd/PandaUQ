from numpy.core.fromnumeric import compress
import uproot3 as up
import numpy as np
import h5py
import pandas as pd

#from var_mapping import mapping
import sys
sys.path.append('../')
import lib_plotting

import argparse

parser = argparse.ArgumentParser(description='Add input and output files.')
parser.add_argument('-s', '--sigfile', type=str,
	default='/Users/binbindong/Desktop/work/PandaX/PandaUQ/input/simu_2hit_bdt.root')
parser.add_argument('-b', '--bkgfile', type=str,
	default='/Users/binbindong/Desktop/work/PandaX/PandaUQ/input/total_pair_test_2hit.root')
parser.add_argument('-o', '--outputfile', type=str)
args = parser.parse_args()

signal = up.open(args.sigfile)['miniTree']
bkg = up.open(args.bkgfile)['miniTree']

#var_list = list(mapping.keys())

df_S = signal.pandas.df()
df_B = bkg.pandas.df()

######## plotting input variables #########
## note: no scaling is applied here
lib_plotting.variable_plotting(df_S, df_B, outputFile='../plots/inputVar.pdf')

X_train = np.concatenate((pd.DataFrame(df_B), pd.DataFrame(df_S)))
labels = np.concatenate((np.zeros(len(df_B), dtype=int), np.ones(len(df_S), dtype=int)))
print("X_train shape: {}".format(X_train.shape))

rng_state = np.random.get_state()
np.random.shuffle(X_train)
np.random.set_state(rng_state)
np.random.shuffle(labels)
assert X_train.shape[1] == 10 ## number of input variables 

outputfile = h5py.File(args.outputfile, 'w')
outputfile.create_dataset('X_train', data=X_train, compression='gzip')
outputfile.create_dataset('labels',  data=labels,  compression='gzip')
outputfile.close()
