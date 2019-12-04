# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 21:38:42 2019

@author: johvicente
"""

# PURPOSE: Get probability score out of CNN
# Don't use flowfromdirectory for image import but numpy and convert
# values as:
#
# test_set = test_set.astype('float32')
# training_set = training_set.astype('float32')
#
# Use result = papernet.predict_proba(image_file, verbose = 1) for prediction
# Try sigmoid and softmax as final activation



import os 
import numpy as np 
from keras.preprocessing import image
 
PATH = os.getcwd()
 
train_path = PATH+'/data/train/'
train_batch = os.listdir(train_path)
x_train = []
 
# if data are in form of images
for sample in train_data:
	img_path = train_path+sample
	x = image.load_img(img_path)
	# preprocessing if required
	x_train.append(x)
 
test_path = PATH+'/data/test/'
test_batch = os.listdir(test_path)
x_test = []
 
for sample in test_data:
	img_path = test_path+sample
	x = image.load_img(img_path)
	# preprocessing if required
	x_test.append(x)
	
# finally converting list into numpy array
x_train = np.array(x_train)
x_test = np.array(x_test)



