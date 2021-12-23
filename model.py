from keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Input, Dropout
from keras.models import Model
from keras import metrics
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import sys

def significance(y_true, y_pred):
	y_pred_pos = K.round(K.clip(y_pred, 0, 1))   
	
	y_pos = K.round(K.clip(y_true, 0, 1))
	y_neg = 1 - y_pos
	
	tp = K.sum(y_pos * y_pred_pos)
	fp = K.sum(y_neg * y_pred_pos)

	sWeight = 1.95/(73010*0.7)
	bWeight = 456.9/(946892*0.7)

	sig = tp*sWeight/K.sqrt(tp*sWeight+fp*bWeight)

	return sig

def DLModel(InputShape, h_layers, lr=0.001, drops=None, dropout=True, met='acc'):
	'''
	InputShape = number of input variables
	h_layers = number of nodes in each layer
	lr = learning rate
	drops = dropout rate in each hidden layer
	dropout = if you want to turn on dropout in trianing and testing'''
	
	In = Input(shape=[InputShape,])
	x = In
	for i, h in enumerate(h_layers[:]):
		x = Dense(h)(x)
		#x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = Dropout(drops[i])(x, training=dropout)
	predictions = Dense(1, activation='sigmoid')(x)

	model = Model(inputs=In, outputs=predictions)

	model_optimizer = Adam(learning_rate=lr)
	if met=='acc':
		metr = ['accuracy']
	elif met=='recall':
		metr = metrics.Recall(thresholds=0.6)
	elif met=='prec':
		metr = metrics.Precision(thresholds=0.6)
	elif met=='sig':
		metr = [significance]
	else:
		print("ERROR: no such metrics available. Available metrics in this script: acc (for accuracy), recall (for recall), prec (for precision) and sig (for significance). For other keras metrics or custom metrics, please add it by yourself.")
		sys.exit()

	model.compile(
		loss = 'binary_crossentropy',
		optimizer = model_optimizer,
		metrics=metr,
	)

	return model
