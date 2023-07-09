import numpy as np
import os
import cv2
import pandas as pd
from PIL import Image
from retrieve import get_image_list, retrieve_image, Search
import base64
from indexer import Indexing_feature

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



def Evaluate(image_root = 'dataset', option = 'VGG16', search = 'Faiss', limit_image = 30):
    image_root = 'dataset'
    FEATURE_PATH = 'feature'
    if os.path.exists(FEATURE_PATH + "/" + option + ".index.bin") == False:
        Indexing_feature(image_root,option)
        
    AP_list = []  # AP cho mỗi class
    mAP_list = []  # mAP của mỗi class -> tính mean của mAP này
    previous_label = os.listdir(image_root)[0].split(' ')[0]
    # back_dress
    image_count = 0
    label = []
    for filename in os.listdir(image_root):
        image_count += 1
        print(image_count)
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
            label.append(previous_label)
            previous_label = class_name
            # print(mAP_list)
        # faiss / if else chọn search khác
        # danh sách ảnh => toàn bộ ảnh
        class_list = []
        query_image = Image.open(uploaded_file)
        if search == 'Faiss':
            retriev = retrieve_image(query_image, option, limit_image)
            # 30 phần tử

            image_list = get_image_list(image_root)
            
            # 30 black dress, black pant
            for u in range(len(retriev)):
                image_link = (str(image_list[retriev[u]]).split(
                    '\\')[1]).split(' ')[0]
                class_list.append(image_link)
        else:
            retriev = Search(query_image,option,search,limit_image)
            for item in retriev:
                image_link = (item[1].split('/')[1]).split(' ')[0]
                class_list.append(image_link)
        # calculate ap in query i-th

        # black dress -> 30 class khác
        AP = calculate_AP(class_name, class_list)
        AP_list.append(AP)
        if len(os.listdir(image_root)) == image_count:
            label.append(previous_label)
            mAP = mean_average_precision(np.array(AP_list))
            mAP_list.append(mAP)
        # print(class_list)

    MMAP = np.mean(mAP_list)
    return mAP_list, MMAP,label

def get_label():
    label = set()
    image_root = 'dataset'
    for filename in os.listdir(image_root):
        label.add(filename.split(' ')[0])
    return label

def store_mmap(option,search,MMAP,mAP_list,label,limit_image=30):
    RESULT_PATH = "result"
    CSV_PATH = os.path.join(RESULT_PATH,"MMAP.csv")
    if os.path.exists(CSV_PATH):
        idx = len(pd.read_csv(CSV_PATH)) + 1
        with open (CSV_PATH,'a') as file:
            output = f"{idx},{option},{search},{limit_image},{MMAP}"
            file.write('\n'+output)
    else:
        idx = 1
        with open (CSV_PATH,'w') as file:
            file.write("ID,Method,Search,Limit Image,MMAP")
            output = f"{idx},{option},{search},{limit_image},{MMAP}"
            file.write('\n'+output)
    store_map(idx,mAP_list,label)

def store_map(idx,mAP_list,label):
    RESULT_PATH = "result"
    CSV_PATH = os.path.join(RESULT_PATH,"mAP.csv")
    if os.path.exists(CSV_PATH):
        with open (CSV_PATH,'a') as file:
            for i in range(0,len(mAP_list)):
                output = f"{idx},{label[i]},{mAP_list[i]}"
                file.write('\n'+output)

    else:
        with open (CSV_PATH,'w') as file:
            file.write("Id_MMAP,Class,mAP")
            for i in range(0,len(mAP_list)):
                output = f"{idx},{label[i]},{mAP_list[i]},"
                file.write('\n'+output)

def store_infor():
    options = ['VGG16', 'RGBHistogram', 'LBP'] 
    searchs = ['Faiss','Cosine','Euclidean']

    limit_image = 30
    for option in options:
        for search in searchs:
            mAP_list, mAP_final,label = Evaluate(option=option, search=search)
            store_mmap(option,search,limit_image,mAP_final,mAP_list,label)

if __name__ == "__main__":
    options = ['VGG16', 'RGBHistogram', 'LBP']
    mAP_list, mAP_final,label = Evaluate(option=options[2], search='Faiss')
    print(mAP_list)
    store_mmap(options[2],'Faiss',mAP_final,mAP_list,label)

    # tính cho cả thằng eluc + cosion
