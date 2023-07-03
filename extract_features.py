# import the necessary packages
import numpy as np
import cv2
import imutils
import glob
import numpy as np
import csv
from skimage import feature
import os

#import module
from simple_module import Descriptor
from vgg_module import VGG_16


index_path = "index.csv"  
output = open(index_path, "w")
folder_paths = ['dataset/black_dress', 
                  #'dataset/black_pants', 
                  #'dataset/black_shirt', 
                  #'dataset/black_shoes', 
                  #'dataset/black_shorts', 
                  'dataset/blue_dress', 
                  #'dataset/blue_pants', 
                  #'dataset/blue_shirt', 
                  #'dataset/blue_shoes', 
                  #'dataset/blue_shorts', 
                  'dataset/brown_pants', 
                  #'dataset/brown_shoes', 
                  #'dataset/green_pants', 
                  #'dataset/green_shirt', 
                  #'dataset/green_shoes', 
                  #'dataset/green_shorts',
                  #'dataset/red_dress',
                  #'dataset/red_pants',
                  #'dataset/red_shoes',
                  #'dataset/white_dress',
                  #'dataset/white_pants',
                  #'dataset/green_shorts',
                  #'dataset/white_shorts',
                  ]
cd = Descriptor((8, 12, 3))
vgg = VGG_16()
folder = 0
for folder_path in folder_paths:
    print("Folder",folder)
    folder +=1
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            #model
            features = vgg.extract_features(image_path)
            #features = cd.Historam_extract(image)
            features = [str(f) for f in features] 
            output.write("%s,%s\n" % (f"{folder_path}/" + filename, ",".join(features)))

output.close()  
