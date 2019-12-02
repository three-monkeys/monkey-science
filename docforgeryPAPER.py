
# https://projet.liris.cnrs.fr/imagine/pub/proceedings/ICPR-2014/data/5209d168.pdf

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

def paper_model(img_shape=(150, 150, 1), n_classes=1, l2_reg=0.,
	weights=None):

	# Initialize model
	paper = Sequential()

	# Layer 1
	paper.add(Conv2D(20, (7, 7), input_shape=img_shape,
	padding='same', kernel_regularizer=l2(l2_reg)))
	paper.add(Activation('relu'))
	paper.add(MaxPooling2D(pool_size=(4, 4)))

	# Layer 2
	paper.add(Conv2D(50, (5, 5), padding='same'))
	paper.add(Activation('relu'))
	paper.add(MaxPooling2D(pool_size=(4, 4)))

	# Layer 6
	paper.add(Flatten())
	paper.add(Dense(1000))
	paper.add(Activation('relu'))
	paper.add(Dropout(0.1))

	# Layer 7
	paper.add(Dense(1000))
	paper.add(Activation('relu'))
	paper.add(Dropout(0.1))

	# Layer 8
	paper.add(Dense(n_classes))
	paper.add(Activation('sigmoid'))

	if weights is not None:
		paper.load_weights(weights)

	return paper

batch_size = 32

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   zoom_range = 0.1,
                                   brightness_range=[0.5,1.0])

test_datagen = ImageDataGenerator(rescale = 1./255,
                                  brightness_range=[0.5,1.0])

training_set = train_datagen.flow_from_directory('C:/Users/Johvicente/Documents/DocumentForgeryProject/training_set',
                                                 target_size = (150, 150),
                                                 batch_size = batch_size,
                                                 class_mode = 'binary',
                                                 color_mode = "grayscale")

test_set = test_datagen.flow_from_directory('C:/Users/Johvicente/Documents/DocumentForgeryProject/test_set',
                                            target_size = (150, 150),
                                            batch_size = batch_size,
                                            class_mode = 'binary',
                                            color_mode = "grayscale")
                                        

papernet = paper_model(img_shape=(150, 150, 1), n_classes=1, l2_reg=0., weights=None)

papernet.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

papernet.fit_generator(training_set,
                         steps_per_epoch = 157/batch_size,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 21)

papernet.save("docforgPAPER.h5")


image_file = image.load_img('C:/Users/Johvicente/Documents/DocumentForgeryProject/single prediction/test_img.PNG', color_mode="grayscale", target_size = (150, 150))

''' from image to 3D array '''
image_file = image.img_to_array(image_file)

''' add new dimension so that the predics method can be applied '''
image_file = np.expand_dims(image_file, axis=0)

# make prediction
result = papernet.predict(image_file)

# result - what corresponds to 0 and what corresponds to 1? Named after the underlying folder (execute below)
training_set.class_indices
# test_set.class_indices
# get result
if result[0][0] == 1:
    print('There is a Gehalt on the image') 
else:
    print('There is a dog on the image') 