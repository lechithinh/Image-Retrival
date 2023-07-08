import time
import faiss
import torch
from torch.utils.data import DataLoader, SequentialSampler
from descriptor import MyResnet50, MyVGG16, RGBHistogram, LBP
from helpers import get_faiss_indexer
from dataloader import MyDataLoader
import os


def Indexing_feature(image_path, feature_descriptor,batch_size = 64):
    FEATURE_PATH = 'feature'
    try: 
        os.makedirs(FEATURE_PATH, exist_ok=True)
    except:
        print(KeyError)
    
    print('Start indexing .......')
    device_option = "cuda" if torch.cuda.is_available() else "cpu"
    start = time.time()
    device = torch.device(device_option)
 
    # Load module feature extraction 
    if feature_descriptor == 'Resnet50':
        descriptor = MyResnet50(device)
    elif feature_descriptor == 'VGG16':
        descriptor = MyVGG16(device)
    elif feature_descriptor == 'RGBHistogram':
        descriptor = RGBHistogram(device)
    elif feature_descriptor == 'LBP':
        descriptor = LBP(device)
    else:
        print("No matching model found")
        return

    dataset = MyDataLoader(image_path)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,batch_size=batch_size,sampler=sampler)

    indexer = get_faiss_indexer(descriptor.shape)
    image_number = 1
    for images, image_paths in dataloader:
        print("Image Number", image_number)
        image_number += 1
        images = images.to(device)
        features = descriptor.extract_features(images) #64 đặc trưng => xử lý tên file, vector...
        # print(features.shape)
        #store to csv 
        indexer.add(features) #store faisss
    
    # Save features
    feature_file = FEATURE_PATH + '/' + feature_descriptor + '.index.bin'
    try: 
        if os.path.exists(feature_file):
            os.remove(feature_file)
            print("File exists - removed")
        faiss.write_index(indexer, feature_file)
    except:
        print(KeyError)
    
    end = time.time()
    print('Finish in ' + str(end - start) + ' seconds')

if __name__ == '__main__':
    image_root = 'dataset'
    feature_descriptor = ['Resnet50', 'VGG16', 'RGBHistogram', 'LBP']
    Indexing_feature(image_path=image_root, feature_descriptor = feature_descriptor[2])