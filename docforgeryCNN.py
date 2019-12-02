# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:46:01 2019

@author: bgalesic
"""

%reset

import os
os.getcwd()
os.chdir("C:/Users/bgalesic/Documents/work/ECB NLP/NLP Use Case")

###########
import glob
# package for parse PDFs
from tika import parser
import nltk

import pandas as pd
import numpy as np
import textblob

'''
# Reading PDF files from folder

def ReadPDFfromFolder():
    path_folder = r'.\documents'
    # get all file names and store it in "allFiles"
    allFiles = glob.iglob(path_folder + "/*.pdf")

    # create empty DataFrames to fill up
    df = pd.DataFrame(columns=['file_name','full_text'])
    df_temp = pd.DataFrame(columns=['file_name','full_text'])

    for file in allFiles:
        # get file name and content from PDF File
        file_name = os.path.basename(file)
        parsed_data = parser.from_file(file)['content']

        # fill up temp dataframe
        df_temp["file_name"] = [file_name]
        df_temp["full_text"] = parsed_data

        # append it to the final DataFrame
        df = df.append(df_temp)

    return(df)


df = ReadPDFfromFolder()

doc = df.iloc[1,1]
'''

###############################################################################

# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense




# data
''' Here we don't import data, we just store it in different folders (train/test) and Keras will recognize it '''
''' See below '''

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution - add convolutional layers
''' 32 feature detectors of 3 by 3 dimensions ; 3D dim input when colored picture '''
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 1), activation = 'relu'))

# Step 2 - Max Pooling - reduce feature maps
'''  Reduce feature map to 2 by 2 while still keeping most important information '''
classifier.add(MaxPooling2D(pool_size = (2, 2)))

''''
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
'''

# Step 3 - Flattening - flatten feature maps into single input vector for NN
classifier.add(Flatten())

# Step 4 - Full connection - create "classic" ANN using 'flattened' input vector / fully connected layer = hidden layer
''' units = # of nodes in hidden layer '''
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
''' stochastic gradient descent algorithm = adam, bin_crossentr. loss function because binary classification problem, accuracy performance metric '''
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
''' keras.io - Keras documentation for image preprocessing - code is from there '''
''' there is also sample code for text preprocessing '''
''' Preprocess images by applying image augmentation '''
''' Image augmentation - flipping, shifting etc. images to get more training data out of images '''

''' import ImageDataGenerator class for image augmentation '''
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

''' tell Keras where training data is '''
''' target size needs to match to input dimensions chosen above in convolutional layer '''
training_set = train_datagen.flow_from_directory('C:/Users/Johvicente/Documents/DocumentForgeryProject/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary',
                                                 color_mode = "grayscale")

'''
training_set = train_datagen.flow_from_directory('C:/Users/bgalesic/Documents/work/Document Forgery/Test Data/Printed/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

C:/Users/bgalesic/Documents/work/Document Forgery/Test Data/Printed/Train Set
'''
''' tell Keras where test data is '''
test_set = test_datagen.flow_from_directory('C:/Users/Johvicente/Documents/DocumentForgeryProject/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary',
                                            color_mode = "grayscale")

''' steps_per_epoch = 8000 needs to match to number of images in training set '''
''' validation_steps = 2000 needs to match to number of images in test set '''
classifier.fit_generator(training_set,
                         steps_per_epoch = 5,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 21)

classifier.save("model.h5")
'loaded_model.load_weights("model.h5")


# Part 3 - Single Prediction
''' is pet in cat_or_dog_1 image a cat or a dog? '''

import numpy as np
from keras.preprocessing import image

image_file = image.load_img('C:/Users/Johvicente/Documents/DocumentForgeryProject/single prediction/IMG_0337bw.jpg', color_mode="grayscale", target_size = (64, 64))

''' from image to 3D array '''
image_file = image.img_to_array(image_file)

''' add new dimension so that the predics method can be applied '''
image_file = np.expand_dims(image_file, axis=0)

# make prediction
result = classifier.predict(image_file)

# result - what corresponds to 0 and what corresponds to 1? Named after the underlying folder (execute below)
training_set.class_indices
# test_set.class_indices
# get result
if result[0][0] == 1:
    print('There is a Gehalt on the image') 
else:
    print('There is a dog on the image') 


