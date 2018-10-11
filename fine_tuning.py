# coding: utf-8

import joblib
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping

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


#Load the VGG model
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=X_train.shape[1:]

# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False
 
# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)

# Create the model
model = models.Sequential()
 
# Add the vgg convolutional base model
model.add(vgg_conv)
 
# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(folder), activation='softmax'))
 
# Show a summary of the model. Check the number of trainable parameters
model.summary()

## Training
# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# Training
es_cb = EarlyStopping(monitor='val_loss', patience=2, mode='auto')
tb_cb = TensorBoard(log_dir='./logs/')
history = model.fit(X_train, y_train, batch_size=32, epochs=30, callbacks=[tb_cb, es_cb], validation_data=(X_test, y_test), initial_epoch=0)

# Evaluation
print(model.evaluate(X_test, y_test))
# Save the model
model.save('models/model.h5')

