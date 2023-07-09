#new libraries
import time
import torch
import faiss
import pathlib
import os
import csv
from PIL import Image
from helpers import paginator
from streamlit_cropper import st_cropper

from descriptor import MyVGG16, RGBHistogram, LBP
from dataloader import get_transformation




FEATURE_PATH = 'feature'
def cosine(vec_a, vec_b):
        dot = sum(a*b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a*a for a in vec_a) ** 0.5
        norm_b = sum(b*b for b in vec_b) ** 0.5

        cos_sim = dot / (norm_a*norm_b)
        return 1 - cos_sim

def euclidean(vec1, vec2):
    pre_square_sum = 0
    length = min(len(vec1), len(vec2))

    for index in range(length):
        pre_square_sum += (vec1[index] - vec2[index])**2
    return pre_square_sum

def get_image_list(image_root = 'dataset'):
    image_root = pathlib.Path(image_root)
    image_list = list()
    for image_path in image_root.iterdir():
        if image_path.exists():
            image_list.append(image_path)
    image_list = sorted(image_list, key = lambda x: x.name)
    return image_list


def retrieve_image(query_image, feature_extractor, limit_image = 10):
    device_option = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_option)
    
    if feature_extractor == 'VGG16':
        extractor = MyVGG16(device)
    elif feature_extractor == 'RGBHistogram':
        extractor = RGBHistogram(device)
    elif feature_extractor == 'LBP':
        extractor = LBP(device)

    transform = get_transformation()

    query_image = query_image.convert('RGB')
    image_tensor = transform(query_image)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    feat = extractor.extract_features(image_tensor)

    indexer = faiss.read_index(FEATURE_PATH + '/' + feature_extractor + '.index.bin')

    _, indices = indexer.search(feat, k=limit_image)

    return indices[0]

def Search(query_image, feature_extractor, metric, limit_image = 10):
    device_option = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_option)
    
    if feature_extractor == 'VGG16':
        extractor = MyVGG16(device)
    elif feature_extractor == 'RGBHistogram':
        extractor = RGBHistogram(device)
    elif feature_extractor == 'LBP':
        extractor = LBP(device)
    transform = get_transformation()

    query_image = query_image.convert('RGB')
    image_tensor = transform(query_image)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    queryFeatures = extractor.extract_features(image_tensor)
    
    results = {}
    csv_path = os.path.join("feature/",f"{feature_extractor}.csv")
    with open(csv_path) as f:
        reader = csv.reader(f)

        for row in reader:
            features = [float(x) for x in row[1:]]
            if metric == "Cosine":
                d = cosine(features, queryFeatures[0])
            elif metric == "Euclidean":
                d = euclidean(features, queryFeatures[0])
            results[row[0]] = d
        f.close() 
    if metric in ["Euclidean", "Cosine"]:
        results = sorted([(v, k) for (k, v) in results.items()])
    else:
        results = [(v, k) for (k, v) in results.items()]
    
    if limit_image != None:
        return results[:limit_image]
    else:
        return results

def main():  
    uploaded_file = 'dataset' + '/' + 'white_pants (2).jpg'
    query_image = Image.open(uploaded_file)
    print(Search(query_image,'RGBHistogram','Euclidean'))
    
if __name__ == "__main__":
    main()