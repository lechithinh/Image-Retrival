import numpy as np
import os
import cv2
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

#Fix lõi: AP cuối nó khong append vô 
#
def calculate_AP(class_query, class_list):  # class name, class_list
    precisions = []
    # If the class in class list is the same as the retrival = 1, otherwise = 0 (dài bằng labels)
    relenvance = []
    pos = 0  # tử thứ k
    num = 0  # mẫu thứ k
    for class_name in class_list:
        num += 1  # mẫu sẽ tăng theo mỗi lần lặp
        if class_query == class_name:  # if retrival image giống label
            relenvance.append(1)
            pos += 1
        else:
            relenvance.append(0)
        precisions.append(float(pos/num))  # calculate precsion thứ K

    precision_r = 0
    n_of_correct_class = sum(relenvance)

    # calculate AP in query thu i
    for i in range(len(precisions)):
        precision_r += precisions[i]*relenvance[i]
    # if relevant image  = 0 -> AP = 0
    if n_of_correct_class == 0:
        return 0
    else:
        AP = 1/n_of_correct_class * precision_r
        return AP


def mean_average_precision(ap):
    return np.mean(ap)


def Evaluate(image_root='dataset', option='VGG16', limit_image=30):

    AP_list = []  # AP cho mỗi class
    mAP_list = []  # mAP của mỗi class -> tính mean của mAP này
    previous_label = os.listdir(image_root)[0].split(' ')[0]
    # back_dress
    image_count = 0
    for filename in os.listdir(image_root):
        image_count += 1
        uploaded_file = image_root + '/' + filename
        class_name = filename.split(' ')[0]  # black dress
        if previous_label == class_name:
            previous_label = class_name
        else:
            # in ra 30 AP tấm hình trước đó
            # print(AP_list)
            # nếu mà qua class mới thì tính mAP của class trước đó
            mAP = mean_average_precision(np.array(AP_list))
            mAP_list.append(mAP)
            # reset mảng AP cho class tiếp theo
            # print(AP_list)
            AP_list = []
            previous_label = class_name
            # print(mAP_list)
        # faiss / if else chọn search khác
        query_image = Image.open(uploaded_file)
        retriev = retrieve_image(query_image, option, limit_image)
        # 30 phần tử
        image_list = get_image_list(image_root)
        # danh sách ảnh => toàn bộ ảnh
        class_list = []
        # 30 black dress, black pant
        for u in range(len(retriev)):
            image_link = (str(image_list[retriev[u]]).split(
                '\\')[1]).split(' ')[0]
            class_list.append(image_link)
        # calculate ap in query i-th

        # black dress -> 30 class khác
        AP = calculate_AP(class_name, class_list)
        AP_list.append(AP)
        if len(os.listdir(image_root)) == image_count:
            mAP = mean_average_precision(np.array(AP_list))
            mAP_list.append(mAP)
        # print(class_list)
    return mAP_list, np.mean(mAP_list)


if __name__ == "__main__":
    options = ['VGG16', 'RGBHistogram', 'LBP']
    mAP_list, mAP_final = Evaluate(option=options[1])

    # tính cho cả thằng eluc + cosion
