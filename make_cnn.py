# coding: utf-8

"""
	make_cnn.py
	~~~~~~~~~~~~~~~~

	Make CNN models.

	Dependency::
		python : 3.6.*  
		Package : Please see requirements.txt

	Usage::
		- Using pre-trained model  
			>>> python make_cnn.py --debug --file XY_224.txt -F -M VGG
		or 
		- Using own model 
			>>> python make_cnn.py --debug --file XY_224.txt
"""
from keras.applications import VGG16, ResNet50, InceptionV3
from keras import models
from keras import layers
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard, EarlyStopping
import argparse, os, glob
from PIL import Image
import numpy as np
from keras.utils import np_utils
import joblib

LABEL = ["ALKALINE", "LIION", "NIMH", "NICD"]
IMAGE_SIZE = 224

VGG = 'VGG'	# VGG16
RN = 'RN'	# ResNet50
I = 'I' 	# InceptionV3
X = 'X' # Xception


def main(args):
	if not os.path.isdir(args.traindata):
		print('Plese choose valid path : ' + args.traindata)
		exit(1)

	if args.debug:
		print('Make data...')

	X_train, X_test, y_train, y_test = make_data(args.traindata, args.traindata + '/' + args.file)

	model = models.Sequential()
	if args. fine:
		print(args.model)
		if args.model == VGG:
			conv = VGG16(weights='imagenet', include_top=False, input_shape=X_train.shape[1:])
		elif args.model == RN:
			conv = ResNet50(weights='imagenet', include_top=False, input_shape=X_train.shape[1:])
		elif args.model == I:
			conv = InceptionV3(weights='imagenet', include_top=False, input_shape=X_train.shape[1:])
		elif args.model == X:
			conv = Xception(weights='imagenet', include_top=False, input_shape=X_train.shape[1:])
		else:
			print('You can not choose ' + args.model + '!')
			exit(1)
		model = make_model_fine(model, conv, args.debug)	
	else:
		model.add(layers.Conv2D(32, (3, 3) , padding='same', input_shape=X_train.shape[1:]))
		model = make_own_model(model, args.debug)

	if args.debug:
		print('\nModel summary : ')
		model.summary()
		print()
	trained_model = train(X_train, y_train, model, args.name, (X_test, y_test), args.debug)
	if args.debug:
		print(trained_model.evaluate(X_test, y_test))

	# Save the model
	trained_model.save('models/' + args.name + '.h5')


def make_model_fine(model, conv, debug=False):
	""" make CNN model using by Fine-tuning.

		:param model: basic model
		:param conv: pretrained model 
		:return model: compiled model
	"""
	layer_num = len(conv.layers)
	for layer in conv.layers[:int(layer_num * 0.8)]:
		if 'BatchNormalization' in str(layer):
			...
		else:
			layer.trainable = False

	if debug:
		print('Layer : ')
		for layer in conv.layers:
			print(layer, layer.trainable)

	 
	# Add new layers
	model.add(conv)
	model.add(layers.Flatten())
	model.add(layers.Dense(1024, activation='relu'))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(len(LABEL), activation='softmax'))

	# Compile the model
	model.compile(loss='categorical_crossentropy',
				  optimizer=optimizers.RMSprop(lr=1e-4),
				  metrics=['acc'])


	return model


def make_own_model(model, debug=False):
	""" make CNN model

		:param model: basic model
		:return model: compiled model
	"""

	# Add new layers
	model.add(layers.Activation('relu'))
	model.add(layers.Conv2D(32, (3, 3)))
	model.add(layers.Activation('relu'))
	model.add(layers.MaxPooling2D(pool_size=(2, 2)))
	model.add(layers.Dropout(0.25))

	model.add(layers.Flatten())
	model.add(layers.Dense(1024))
	model.add(layers.Activation('relu'))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(len(LABEL)))
	model.add(layers.Activation('softmax'))

	# Compile the model
	model.compile(loss='categorical_crossentropy',
				  optimizer='SGD',
				  metrics=['accuracy'])

	return model


def make_data(train_data_path, file):
	""" Processe the train data for CNN training.

		:param train_data_path: path to the train data :type: str
		:param file: If you want to use processed data, please enter file name. :type: str
		:return : processed data, X_train, X_test, y_train, y_test
	"""
	X = []
	Y = []
	if not file:
		# X is the data of image, Y is the label of correct.
		for index, name in enumerate(LABEL):
			files = glob.glob(train_data_path + '/' + name + "/*.jpg")
			for i, file in enumerate(files):
				image = Image.open(file)
				image = image.convert("RGB")
				image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
				data = np.asarray(image)
				X.append(data)
				Y.append(index)

		# Change the data from 0 to 1
		# The original data is (0,0,0) to (255,255,255) because it is RGB.  
		# But the data should be 0 to 1.
		X = np.array(X, dtype=np.float32)
		X = X / 255.0

		# Change the type of label
		# The correct label should have a vector with 0 or 1.  
		# So change the type.
		Y = np_utils.to_categorical(Y, len(LABEL))

	else:
		if not os.path.isfile(file):
			print('Plese choose valid path : ' + file)
			exit(1)
		X, Y = joblib.load(open(file, 'rb'))

	# Devide into training data and test data
	return train_test_split(X, Y, test_size=0.20, random_state = 24)


def train(X_train, y_train, model, name, valid, debug=False):
	""" train the model

		:param X_train: training image data :type: np.array
		:param y_train: the label of X_train  :type: np.array
		:param model: Model to train
		:param name: the name of model. Use for saving log. :type: str
		:param valid: valid data for validation_data, (x_text, y_test)
		:return model: trained model
	"""
	logDir = 'models/log_' + name
	if not os.path.isdir(logDir):
		os.makedirs(logDir)
	es_cb = EarlyStopping(monitor='val_loss', patience=3, mode='auto', save_best_only=True)
	tb_cb = TensorBoard(log_dir=logDir)
	history = model.fit(X_train, y_train, batch_size=32, epochs=100, callbacks=[tb_cb, es_cb], validation_data=valid, initial_epoch=0)

	return model


if __name__ == '__main__':
	# Make parser.
	parser = argparse.ArgumentParser(
				prog='make_cnn.py', 
				usage='Make NN by Fine-tuning.', 
				description='description...',
				epilog='end',
				add_help=True,
				)
	parser.add_argument('-F', '--fine', help='Using fine-tuning.', action='store_true', required=False, default=False)
	parser.add_argument('-M', '--model', help="You can choose pretrained model when you use fine-tuning option. ['VGG', 'RN', 'I', 'X']. VGG is VGG16, RN is ResNet50, I is InceptionV3 and X is Xception." ,
							 required=False, choices=[VGG, RN, I, X], default=VGG)
	parser.add_argument('--file', help='If you want to use processed data. Please enter the name of data file. It should be used joblib.', required=False, default='')
	parser.add_argument('-D', '--traindata', help="Path to the traindata. That directory should have label name. Default is './train_data'.", 
							required=False, default='./train_data')
	parser.add_argument('-N', '--name', help="The neme of model. Default is 'model'.'", 
							required=False, default='model')
	parser.add_argument('--debug', help='Debug mode.', action='store_true', required=False, default=False)

	# parse thearguments.
	args = parser.parse_args()
	main(args)
