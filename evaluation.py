import numpy as np
import os
import cv2
from storage.Search_module import Searcher
from storage.simple_module import Descriptor
from PIL import Image
from retrieve import get_image_list, retrieve_image
import base64


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
    # retrievals is a label of each retried image -> set equal to label of input image
    retrievals = [retrieval for x in range(len(labels))]
    precisions = []
    r_k = [] #r_k[] is array of relavant image if rue r_k = 1 else r_k = 0
    pos = 0 # pos is number of precision in k-th
    num = 0 #image thu k-th
    for retrieval, label in zip(retrievals, labels):
        num += 1
        if retrieval == label: #if retrival image == label
            r_k.append(1)
            pos += 1
        else:
            r_k.append(0)
        precisions.append(pos/num) #calculate precsion thu k
    precision_r = 0
    RD = 0 # RD is number relevant image

    #calculate AP in query thu i
    for i in range(len(precisions)):
        if r_k[i] == 1:
            RD += 1
        precision_r += precisions[i]*r_k[i]
    #if relevant image  = 0 -> AP = 0
    if RD == 0:
        return 0
    else:
      return 1/RD * precision_r


def mean_average_precision(ap):
    return np.mean(ap)


def main():
    image_root = 'dataset'

    ap = []
    mAP = []
    count = 0
    for filename in os.listdir(image_root):
        uploaded_file = image_root + '/' + filename
        label = filename.split(' ')[0]

        query_image = Image.open(uploaded_file)
        option = 'VGG16'
        limit_image = 30
        search_model = 'Faiss'
    # search image

        retriev = retrieve_image(query_image, option, limit_image)

    # get the dataset to display
        image_list = get_image_list(image_root)

        result_images = []
        link_images = []
        for u in range(len(retriev)):
            count += 1
            image = Image.open(image_list[retriev[u]])
            result_images.append(image)

            link_images.append((str(image_list[u]).split('\\')[1]).split(' ')[0])
        #calculate ap in query i-th
        ap.append(calculate_AP(label, link_images))
        if count == 30:
            count = 0
            print(ap)
            ap = []
            mAP = mean_average_precision(np.array(ap))
            print(mAP)
            print(link_images)
            break


if __name__ == "__main__":
    main()
