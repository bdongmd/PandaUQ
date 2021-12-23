from keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Input, Dropout
from keras.models import Model
from keras import metrics
from tensorflow.keras.optimizers import Adam
import sys

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
	if met='acc':
		metr = ['accuracy']
	elif met='recall':
		metr = metrics.Recall(thresholds=0.6)
	elif met='prec':
		metr = metrics.Precision(thresholds=0.6)
	else:
		print("ERROR: no such metrics available. Please select betwee: acc, recall and prec.")
		sys.exit()

	model.compile(
		loss = 'binary_crossentropy',
		optimizer = model_optimizer,
		metrics=['accuracy'],
	)

	return model
