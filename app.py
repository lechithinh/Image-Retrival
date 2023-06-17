# import the necessary packages
import numpy as np
import cv2
import imutils
import glob


class ColorDescriptor:
    def __init__(self, bins):
      # store the number of bins for the 3D histogram
      self.bins = bins
    def describe(self, image):
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
  
  

cd = ColorDescriptor((8, 12, 3))

# Set the arguments (you can modify these as per your requirements)
dataset_path = "/content/Dataset"  # Replace with the actual path to your image dataset
index_path = "/content/index.csv"  # Replace with the desired path for the index file

# Open the output index file for writing
output = open(index_path, "w")

for imagePath in glob.glob(dataset_path + "/*.jpg"): #

    imageID = imagePath[imagePath.rfind("/") + 1:]
    image = cv2.imread(imagePath)
    # Describe the image
    features = cd.describe(image)
    print(features)

    features = [str(f) for f in features] # 1.jpg,
    output.write("%s,%s\n" % (imageID, ",".join(features)))

output.close()

# import the necessary packages
import numpy as np
import csv
class Searcher:
    def __init__(self, indexPath):
      self.indexPath = indexPath
    def chi2_distance(self, histA, histB, eps = 1e-10):
      d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
        for (a, b) in zip(histA, histB)])
      return d
    def search(self, queryFeatures, limit = 1):
      results = {}
      with open(self.indexPath) as f:
        reader = csv.reader(f)
        for row in reader:
          features = [float(x) for x in row[1:]] #1.jpg: 0.91
          d = self.chi2_distance(features, queryFeatures)
          results[row[0]] = d
        f.close()
        print("before", results)
      results = sorted([(v, k) for (k, v) in results.items()])
      print("after sort", results)
      return results[:limit]



# load the query image and describe it
query = cv2.imread("/content/Dataset/1.jpg")
features = cd.describe(query)
# perform the search
searcher = Searcher("/content/index.csv")
results = searcher.search(features)

cv2.imshow("Query Image", query)
# loop over the results
for (score, resultID) in results:
	# load the result image and display it
	result = cv2.imread("/content/Dataset" + "/" + resultID)
	cv2.imshow(resultID, result)
	cv2.waitKey(0)
