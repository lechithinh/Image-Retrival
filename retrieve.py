#new libraries
import time
import torch
import faiss
import pathlib
from PIL import Image
from helpers import paginator
from streamlit_cropper import st_cropper

from descriptor import MyVGG16, MyResnet50, RGBHistogram, LBP
from dataloader import get_transformation




FEATURE_PATH = 'feature'

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
    elif feature_extractor == 'Resnet50':
        extractor = MyResnet50(device)
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