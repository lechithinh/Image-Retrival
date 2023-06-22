import numpy as np
import cv2
import glob
import numpy as np
import csv
from Extract_module import Descriptor
from scipy.spatial.distance import euclidean
from math import sqrt

class Searcher:
    def __init__(self, indexPath, limit_image =  1):
      self.indexPath = indexPath
      self.limit_image= limit_image
    def chi2_distance(self, histA, histB, eps = 1e-10):
      d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
        for (a, b) in zip(histA, histB)])
      return d
    def chi2_distance_search(self, queryFeatures, ):
      results = {}
      with open(self.indexPath) as f:
        reader = csv.reader(f)
        for row in reader:
          features = [float(x) for x in row[1:]] #1.jpg: 0.91
          d = self.chi2_distance(features, queryFeatures)
          results[row[0]] = d
        f.close()
      results = sorted([(v, k) for (k, v) in results.items()])
      return results[:self.limit_image]
    
    #add other distance methods here
    def hellinger(self,p,q):
      _SQRT2 = np.sqrt(2)
      return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) **2)) / _SQRT2
    
    def hellinger_distance_search(self, queryFeatures):
      results = {}
      with open(self.indexPath) as f:
        reader = csv.reader(f)
        for row in reader:
          features = [float(x) for x in row[1:]] #1.jpg: 0.91
          d = self.hellinger(features, queryFeatures)
          results[row[0]] = d
        f.close()
        print("before", results)
      results = sorted([(v, k) for (k, v) in results.items()])
      print("after sort", results)
      return results[:self.limit_image]
    

    def cosine(self,vec_a,vec_b):
      dot = sum(a*b for a, b in zip(vec_a, vec_b))
      norm_a = sum(a*a for a in vec_a) ** 0.5
      norm_b = sum(b*b for b in vec_b) ** 0.5

      cos_sim = dot / (norm_a*norm_b)
      return cos_sim
    
    def euclidean(self,vec1,vec2):
      pre_square_sum = 0
      for idx,_ in enumerate(vec1):
        pre_square_sum += (vec1[idx] - vec2[idx]) ** 2
    
      return sqrt(pre_square_sum)
    
def main():
  cd = Descriptor((8, 12, 3))



  query = cv2.imread(r"images\1.png")
  features = cd.describe(query)
  searcher = Searcher("index.csv")
  results = searcher.search(features)

  cv2.imshow("Query Image", query)
  for (score, resultID) in results:
    result = cv2.imread(resultID)
    cv2.imshow(resultID, result)
    cv2.waitKey(0)



