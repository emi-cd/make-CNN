# coding: utf-8

import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import glob


# Make data
folder = ["Alkaline", "LIION", "NIMH", "NICD"]
image_size = 150

# X is the data of image, Y is the label of correct.
X = []
Y = []
for index, name in enumerate(folder):
    files = glob.glob(name + "/*")s
    for i, file in enumerate(files):
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
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
Y = np_utils.to_categorical(Y, len(folder))


# Devide into training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state = 24)


# Make CNN
# I'm not sure with this part, especialy compaile part.
model = Sequential()

model.add(Conv2D(32, (3, 3) , padding='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#model.add(Conv2D(64, (3, 3), padding="same"))
#model.add(Activation('relu'))
#model.add(Conv2D(64,(3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(len(folder)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
             metrics=['accuracy'])


# Training
es_cb = EarlyStopping(monitor='val_loss', patience=2, mode='auto', restore_best_weights=True)
tb_cb = TensorBoard(log_dir='./logs/')
history = model.fit(X_train, y_train, batch_size=32, epochs=40, callbacks=[tb_cb, es_cb], validation_data=(X_test, y_test), initial_epoch=0)

# Evaluation
print(model.evaluate(X_test, y_test))
# Save and download (If you want)
model.save("models/model.h5")

