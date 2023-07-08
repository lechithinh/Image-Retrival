# import the necessary packages
import numpy as np
import cv2
import imutils
import glob
import numpy as np
import csv
from skimage import feature
import os


class Descriptor:
    def __init__(self, bins):
        self.bins = bins

    def Historam_extract(self, image):
        # convert the image to the HSV color space and initialize
        # the features used to quantify the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []
        # grab the dimensions and compute the center of the image
        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))

        # divide the image into four rectangles/segments (top-left,
        # top-right, bottom-right, bottom-left)
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
                    (0, cX, cY, h)]
        # construct an elliptical mask representing the center of the
        # image
        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        ellipMask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
        # loop over the segments
        for (startX, endX, startY, endY) in segments:
            # construct a mask for each corner of the image, subtracting
            # the elliptical center from it
            cornerMask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)
            # extract a color histogram from the image, then update the
            # feature vector
            hist = self.histogram(image, cornerMask)
            features.extend(hist)
        # extract a color histogram from the elliptical region and
        # update the feature vector
        hist = self.histogram(image, ellipMask)
        features.extend(hist)
        # return the feature vector
        return features

    def histogram(self, image, mask):
        # extract a 3D color histogram from the masked region of the
        # image, using the supplied number of bins per channel
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
                            [0, 180, 0, 256, 0, 256])
        # normalize the histogram if we are using OpenCV 2.4
        if imutils.is_cv2():
            hist = cv2.normalize(hist).flatten()
        # otherwise handle for OpenCV 3+
        else:
            hist = cv2.normalize(hist, hist).flatten()
        # return the histogram
        return hist

    def Hog_extract(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128, 256))
        (hog, hog_image) = feature.hog(img, orientations=9, pixels_per_cell=(
            8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
        return hog.flatten()
      
    



def main():
    cd = Descriptor((8, 12, 3))
    dataset_path = "images"
    index_path = "index.csv"

    output = open(index_path, "w")
    folder_paths = ['dataset/black_dress',
                    # 'dataset/black_pants',  
                    # 'dataset/black_shirt',
                    'dataset/black_shoes',
                    # 'dataset/black_shorts',
                    'dataset/blue_dress',
                    'dataset/blue_pants',
                    # 'dataset/blue_shirt',
                    # 'dataset/blue_shoes',
                    # 'dataset/blue_shorts',
                    # 'dataset/brown_pants',
                    # 'dataset/brown_shoes',
                    # # 'dataset/green_pants',
                    # 'dataset/green_shirt',
                    # 'dataset/green_shoes',
                    # 'dataset/green_shorts',
                    # 'dataset/red_dress',
                    # 'dataset/red_pants',
                    # 'dataset/red_shoes',
                    # 'dataset/white_dress',
                    # 'dataset/white_pants',
                    # 'dataset/green_shorts',
                    # 'dataset/white_shorts',
                    ]
    labels = []

    folder = 0
    for folder_path in folder_paths:
        label = folder_path.split('/')[-1]
        labels.append(label)
        folder += 1
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg'):
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)
                features = cd.Historam_extract(image)
                features = [str(f) for f in features]
                output.write("%s,%s\n" %
                             (f"{folder_path}/" + filename, ",".join(features)))
    output.close()


if __name__ == "__main__":
    main()
