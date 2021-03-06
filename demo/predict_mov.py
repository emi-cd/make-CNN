# -*- coding: utf-8 -*-
"""
	predict_mov.py
	Copyright (c) 2018 emi
	~~~~~~~~~~~~~~~~

	Demonstrate using by CNN.
	This script use USB camera and show posibility.
	If you want to stop this program, please enter 'q'.

	Test environment::
		python : 3.6.6 
		Package : Please see requirements.txt

	Usage::
	>>> python predict.py
"""
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image
import copy

LABEL = ["ALKALINE", "LIION", "NIMH", "NICD"]
IMAGE_SIZE=224


def convert_data(image):
	""" Processing data for CNN.

		:param image: image data :type: `Image` object
		:return X: processed data : typt: np.array
	"""
	X = []
	image = image.convert("RGB")
	image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
	data = np.asarray(image)
	X.append(data)
		
	X = np.array(X, dtype=np.float32)
	X = X / 255.0

	return X



cap = cv2.VideoCapture(1)
# Please choice your model!!
model = load_model('../Models/model/model.h5')

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	# frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
	width = (frame.shape[1] if frame.shape[1] < frame.shape[0] else frame.shape[0]) // 2
	x = frame.shape[1] // 2 
	y = frame.shape[0] // 2
	frame = frame[y - width:y + width, x - width:x + width]

	data = convert_data(Image.fromarray(frame))
	result = list(model.predict(data, verbose=0)[0])
	sorted_result = copy.deepcopy(result)
	sorted_result.sort()
	sorted_result.reverse()

	text = ''
	for label in sorted_result:
		text += LABEL[result.index(label)] + ' : ' + str(round(label*100, 2)) + ', '

	# Display the resulting frame 
	cv2.putText(frame, text, (5, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
	cv2.imshow('frame',frame)


	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
