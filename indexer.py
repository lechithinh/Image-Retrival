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
    DATASET_PATH = 'dataset'
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
    csv_path = os.path.join(FEATURE_PATH, f"{feature_descriptor}.csv")
    with open(csv_path,"w") as output:
        for images, image_paths in dataloader:
            print("Image Number", image_number)
            image_number += 1
            images = images.to(device)
            features = descriptor.extract_features(images) #64 đặc trưng => xử lý tên file, vector...
            indexer.add(features) #store faisss
            #store to csv 
            for i in range(0,features.shape[0]):
                feature = [str(f) for f in features[i]] 
                output.write("%s,%s\n" % (f"{DATASET_PATH}/" + image_paths[i].split("\\")[1], ",".join(feature)))
    
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