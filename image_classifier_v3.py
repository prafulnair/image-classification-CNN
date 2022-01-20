# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 18:52:29 2022

@author: Praful Nair
An implementation of Image classification using : Keras.
"""
# Importing essential libraries and Packages.
#from keras.model import Sequential
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
"""
using Sequential from keras.model to use sequential network.
using Conv2D to perform convolution on our 2 dimensional data.
using maxPooling to perform pooling / downscaling of the image. 
using flatten from keras.layer to create one single linear vector of 2d image data. 
"""

# Creating an object of the sequential class
classifier = Sequential()

# Coding the convolution. Thanks to keras for the simplicity.
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Performing pooling to reduce the image/ downscale images
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Let's flatten/convert the data into single long linear vector
classifier.add(Flatten())

# The dense from Keras.layers is used to create completed connected layers
classifier.add(Dense(units=128, activation='relu'))

# Now, creating our output layer, which contains one single node.
classifier.add(Dense(units=1, activation='sigmoid'))

# We have now completed building our CNN model. Next thing to do is to compile
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

"""
The following code is to handle the issue of over-fitting. Over-fitting happens
when our model fits the data too well, which results in poor prediction.
This is pre-processing of our data.
"""

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('dataset1/training_set', target_size=(64, 64), batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory('dataset1/test_set', target_size=(64, 64), batch_size=32, class_mode='binary')

"""
Fitting the data to our model
"""
classifier.fit_generator(training_set, steps_per_epoch=8000, epochs=25, validation_data=test_set, validation_steps=2000)

# This is the last part, in which we will make new predictions using our trained model
test_image = image.load_img('dataset1/single_prediction/dog (11).jpg')
target_size = (64, 64)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'Dog'
else:
    prediction = 'Cat'
    
print(prediction)