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



types = ['histogram','hog','vgg']
cd = Descriptor((8, 12, 3))
vgg = VGG_16()

def Extract_feature(dataset_path, feature_path,feature_names):
    try: 
        os.makedirs(feature_path, exist_ok=True)
        # Create the empty CSV files
        for feature in feature_names:
            file_path = os.path.join(feature_path, f"{feature}.csv")
            with open(file_path, "w") as output:
                
                for folder_path in os.listdir(dataset_path):
                    print("Folder",folder_path)
                    folder_path = dataset_path + "/" + folder_path
                    for filename in os.listdir(folder_path):
                        print(filename)
                        if filename.endswith('.jpg'):
                            image_path = os.path.join(folder_path, filename)
                            image = cv2.imread(image_path)
                            #model
                            if feature == 'vgg':
                                features = vgg.extract_features(image_path)
                            elif feature == 'histogram':
                                features = cd.Historam_extract(image)
                            else:
                                features = cd.Hog_extract(image)
                            features = [str(f) for f in features] 
                            output.write("%s,%s\n" % (f"{folder_path}/" + filename, ",".join(features)))

            output.close()  

    except OSError as error: 
        print(error)  

#Run this 
#Extract_feature(dataset_path="dataset", feature_path="Features",feature_names=types)