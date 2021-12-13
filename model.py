from keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Input, Dropout, add
from keras.models import Model
from keras.optimizers import Adam

def DLModel(InputShape, h_layers, lr=0.001, drops=None, dropout=True):
	In = Input(shape=[InputShape,])
	x = In
	for i, h in enumerate(h_layers[:]):
		x = Dense(h, kernel_initializer='glorot_uniform')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		if dropout:
			x = Dropout(drops[i])(x)
	predictions = Dense(1, activation='sigmoid')(x)

	model = Model(input=In, outputs=predictions)

	model_optimizer = Adam(lr=lr)

	model.complile(
		loss = 'binary_crossentropy',
		optimizer = model_optimizer,
		metrics=['accuracy'],
	)

	return model