# import the necessary packages
import numpy as np
import cv2
import imutils
import glob
import numpy as np
import csv
from skimage import feature

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
        ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
        # loop over the segments
        for (startX, endX, startY, endY) in segments:
          # construct a mask for each corner of the image, subtracting
          # the elliptical center from it
          cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
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
  
    def Hog_extract(self,image):
        img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        img =cv2.resize(img, (128, 256))
        (hog, hog_image) = feature.hog(img, orientations=9,pixels_per_cell=(8, 8), cells_per_block=(2, 2),block_norm='L2-Hys', visualize=True)
        return hog.flatten()
      
#this function is used to extract the data
def main():
  cd = Descriptor((8, 12, 3))
  dataset_path = "images"  
  index_path = "index.csv"  


  output = open(index_path, "w")

  for imagePath in glob.glob(dataset_path + "/*.jpg"): # check other ways to read this

      imageID = imagePath[imagePath.rfind("/") + 1:]
      image = cv2.imread(imagePath)
      #features = cd.Historam_extract(image)
      features = cd.Hog_extract(image)
      print(features)
      features = [str(f) for f in features] 
      output.write("%s,%s\n" % (imageID, ",".join(features)))
    
  output.close()  

if __name__ == "__main__":
  main()