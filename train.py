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
######
# add argument here
######

#####
# Parameter setting here
InputShape = 8
doPlotting = True

##### Data loading

##### Training model parameter setting and loading
Model.summary()

callbacks = EarlyStopping(monitor='val_loss', practice=5)
history = Model.fit(X_train, Y_train,
			batch_size = args.batch_size,
			epochs = args.epoch,
			validation_data = (X_test, Y_test),
			callbacks = callbacks,
			verbose = 2)


train_loss = history.history['loss']
train_acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']

if doPlotting:
	lib_plotting.plotAccLoss(train_loss, val_loss, output_dir=args.output)
	lib_plotting.plotAccLoss(train_acc, val_acc, putVar='Acc', output_dir=args.output)