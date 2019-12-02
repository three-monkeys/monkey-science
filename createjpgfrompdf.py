# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 20:21:03 2019

@author: johvicente
"""

from pdf2image import convert_from_path
from os import listdir
from os.path import isfile, join

mypath = 'C:/Users/Johvicente/Documents/DocumentForgeryProject/training_set/docs/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))] 


list_pdf = []
for pdf in onlyfiles:
	file = mypath + pdf
	pages = convert_from_path(file, 500)
	list_pdf.append(pages)

    
i = 0
for page in list_pdf:
    finalpath = 'C:/Users/Johvicente/Documents/DocumentForgeryProject/training_set/docs/images/' + str(i) + '.jpg'
    page[0].save(finalpath, 'JPEG')
    i = i + 1
    
