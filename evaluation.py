import numpy as np
import os
import cv2
from Search_module import Searcher
from simple_module import Descriptor

def calculate_precision(retrieval, labels):
  relevant = 0
  for value in labels:
    if value == retrieval:
      relevant += 1
  return relevant / len(labels)
def calculate_recall(retrieval, labels, total_relevants):

  relevant = 0
  for value in labels:
    if value == retrieval:
      relevant += 1
  return relevant / total_relevants
  
def f1_score(precision, recall):
  return 2*(precision * recall) / (precision + recall)

def calculate_AP(retrieval, labels):
  #retrievals is a label of each retried image -> set equal to label of input image
  retrievals = [retrieval for x in range(len(labels))]
  precisions = []
  r_k = []
  pos = 0
  num = 0
  for retrieval, label in zip(retrievals, labels):
    num += 1
    if retrieval == label:
      r_k.append(1)
      pos += 1
    else:
      r_k.append(0)
    precisions.append(pos/num)
  precision_r = 0
  RD = 0
  for i in range(len(precisions)):
    if r_k[i] == 1:
      RD += 1
    precision_r += precisions[i]*r_k[i]

  return 1/RD * precision_r

def mean_average_precision(ap):
  return np.mean(ap)

def main():
  cd = Descriptor((8, 12, 3))
  query1 = cv2.imread("dataset/black_dress/433a36e22273b3c314b57aa72c42270fbef8bf53.jpg")
  query2 = cv2.imread("dataset/blue_dress/9cd3a4c79350968817a6146c5ba6bfec5c780b8a.jpg")
  query3 = cv2.imread("dataset/blue_pants/0aa249793bcde3fd4ab53ed17062beb2f24782c4.jpg")
  features = cd.Historam_extract(query2)
  searcher = Searcher("index.csv", limit_image=40)
  label_results = []

  results = searcher.Search("Chi2", features)
  
  label_results = []
  # print(results)
  for item in results:
    label_result = item[1][8:].split('/')[0]
    label_results.append(label_result)
  print(label_results)
  ground_truth = 'black_dress'
  
  ap = calculate_AP(ground_truth, label_results)
  print(mean_average_precision(ap))

  print('----precision------')
  precision = calculate_precision(ground_truth, label_results)
  print(precision)
  print('recall')
  recall = calculate_recall(ground_truth, label_results, 50)
  print(recall)
  print('f1-score', f1_score(precision, recall))
  


if __name__ == "__main__":
    main()
