import sys
import numpy as np
import os
import h5py
import model

### import callbacks for early stoppting...
from keras.callbacks import EarlyStopping, ModelCheckpoint 

### import self defined modules...
import lib_plotting

import argparse

parser = argparse.ArgumentParser(description='Options for launching trianings...')
parser.add_argument('-i', '--input', type=str, default="input/bdong_total_shuffled.h5")
parser.add_argument('-o', '--outputDir', type=str, default="output")
args = parser.parse_args()

#####
# Parameter setting here
InputShape = 10
h_layers = [20, 10, 3]
drops = [0.2, 0.2, 0.2]
dropout=True
lr = 0.0001
batch_size = 128
epochs = 4

doPlotting = True


##### Data loading
trainfile = h5py.File(args.input, 'r')
totalEvents = len(trainfile['X_train'])
halfEvents = int(0.5 * totalEvents)
X_train = trainfile['X_train'][:halfEvents]
Y_train = trainfile['labels'][:halfEvents]
X_test = trainfile['X_train'][halfEvents:]
Y_test = trainfile['labels'][halfEvents:]

##### Training model parameter setting and loading
TrainModel = model.DLModel(InputShape=InputShape, h_layers=h_layers, lr=lr, drops=drops, dropout=dropout)
TrainModel.summary()

##### Hello!! Adding useful early stop...
callbacks = EarlyStopping(monitor='val_loss', patience=5)
history = TrainModel.fit(X_train, Y_train,
			batch_size = batch_size,
			epochs = epochs,
			validation_data = (X_test, Y_test),
			callbacks = callbacks,
			verbose = 1)
TrainModel.save('{}/training_e{}.h5'.format(args.outputDir,epochs))


train_loss = history.history['loss']
train_acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']

if doPlotting:
	lib_plotting.plotAccLoss(train_loss, val_loss, putVar='Loss', output_dir=args.outputDir)
	lib_plotting.plotAccLoss(train_acc, val_acc, putVar='Acc', output_dir=args.outputDir)
