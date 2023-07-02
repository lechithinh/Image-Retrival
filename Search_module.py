import numpy as np
import cv2
import glob
import numpy as np
import csv
from Extract_module import Descriptor
from scipy.spatial.distance import euclidean
from math import sqrt
from timer import Timer


class Searcher:
    def __init__(self, indexPath, limit_image=1):
        self.indexPath = indexPath
        self.limit_image = limit_image

    def chi2_distance(self, histA, histB, eps=1e-10):
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                          for (a, b) in zip(histA, histB)])
        return d
    # add other distance methods here

    def hellinger(self, p, q):
        _SQRT2 = np.sqrt(2)
        return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2

    def cosine(self, vec_a, vec_b):
        dot = sum(a*b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a*a for a in vec_a) ** 0.5
        norm_b = sum(b*b for b in vec_b) ** 0.5

        cos_sim = dot / (norm_a*norm_b)
        return cos_sim

    def euclidean(self, vec1, vec2):
        pre_square_sum = 0
        length = min(len(vec1), len(vec2))

        for index in range(length):
            pre_square_sum += (vec1[index] - vec2[index])**2
        return pre_square_sum
        # for idx,_ in enumerate(vec1):
        #   pre_square_sum += (vec1[idx] - vec2[idx]) ** 2

        # return sqrt(pre_square_sum)

    def Search(self, metric, queryFeatures):
        results = {}
        with open(self.indexPath) as f:
            reader = csv.reader(f)
            for row in reader:
                features = [float(x) for x in row[1:]]
                if metric == "Chi2":
                    d = self.chi2_distance(features, queryFeatures)
                elif metric == "Hellinger":
                    d = self.hellinger(features, queryFeatures)
                elif metric == "Cosine":
                    d = self.cosine(features, queryFeatures)
                elif metric == "Euclidean":
                    d = self.cosine(features, queryFeatures)
                results[row[0]] = d
            f.close()
        if metric in ["Chi2", "Hellinger", "Euclidean"]:
            results = sorted([(v, k) for (k, v) in results.items()])
        else:
            results = [(v, k) for (k, v) in results.items()]
        
        if self.limit_image != None:
          return results[:self.limit_image]
        else:
          return results

def main():
    t = Timer()
    t.tic()
    cd = Descriptor((8, 12, 3))
    query = cv2.imread(
        "dataset/black_shirt/0a4ed000d20e1f1ccc7effd23fa659dbda0e38a3.jpg")
    features = cd.Historam_extract(query)
    searcher = Searcher("index.csv", limit_image=10)

    results = searcher.Search("Cosine", features)
    timeResult = t.toc()
    t.clear()
    print(timeResult)
    cv2.imshow("Query Image", query)
    for (score, resultID) in results:
        result = cv2.imread(resultID)
        cv2.imshow(resultID, result)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
