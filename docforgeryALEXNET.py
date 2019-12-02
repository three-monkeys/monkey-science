# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 11:45:49 2019

@author: johvicente
"""


# Import necessary components to build LeNet
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

def alexnet_model(img_shape=(224, 224, 3), n_classes=10, l2_reg=0.,
	weights=None):

	# Initialize model
	alexnet = Sequential()

	# Layer 1
	alexnet.add(Conv2D(96, (11, 11), input_shape=img_shape,
		padding='same', kernel_regularizer=l2(l2_reg)))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 2
	alexnet.add(Conv2D(256, (5, 5), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 3
	alexnet.add(ZeroPadding2D((1, 1)))
	alexnet.add(Conv2D(512, (3, 3), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 4
	alexnet.add(ZeroPadding2D((1, 1)))
	alexnet.add(Conv2D(1024, (3, 3), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))

	# Layer 5
	alexnet.add(ZeroPadding2D((1, 1)))
	alexnet.add(Conv2D(1024, (3, 3), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 6
	alexnet.add(Flatten())
	alexnet.add(Dense(3072))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(Dropout(0.1))

	# Layer 7
	alexnet.add(Dense(4096))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(Dropout(0.1))

	# Layer 8
	alexnet.add(Dense(n_classes))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('sigmoid'))

	if weights is not None:
		alexnet.load_weights(weights)

	return alexnet

batch_size = 32

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   zoom_range = 0.1,
                                   brightness_range=[0.5,1.0])

test_datagen = ImageDataGenerator(rescale = 1./255,
                                  brightness_range=[0.5,1.0])

training_set = train_datagen.flow_from_directory('C:/Users/Johvicente/Documents/DocumentForgeryProject/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = batch_size,
                                                 class_mode = 'binary',
                                                 color_mode = "grayscale")

test_set = test_datagen.flow_from_directory('C:/Users/Johvicente/Documents/DocumentForgeryProject/test_set',
                                            target_size = (64, 64),
                                            batch_size = batch_size,
                                            class_mode = 'binary',
                                            color_mode = "grayscale")
                                        

alexnet = alexnet_model(img_shape=(64, 64, 1), n_classes=1, l2_reg=0., weights=None)

alexnet.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

alexnet.fit_generator(training_set,
                         steps_per_epoch = 157/batch_size,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 21)

alexnet.save("docforgALEXNET.h5")


image_file = image.load_img('C:/Users/Johvicente/Documents/DocumentForgeryProject/single prediction/Sdefra2p20818091113210_0001 original.jpg', color_mode="grayscale", target_size = (64, 64))

''' from image to 3D array '''
image_file = image.img_to_array(image_file)

''' add new dimension so that the predics method can be applied '''
image_file = np.expand_dims(image_file, axis=0)

# make prediction
result = alexnet.predict(image_file)

# result - what corresponds to 0 and what corresponds to 1? Named after the underlying folder (execute below)
training_set.class_indices
# test_set.class_indices
# get result
if result[0][0] == 1:
    print('There is a Gehalt on the image') 
else:
    print('There is a dog on the image') 
    
    
# TODO: Combine with SVM Classifier