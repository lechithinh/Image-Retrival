import math
import os
from PIL import Image
import numpy as np

from tensorflow.keras.preprocessing import image
# from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import  Model



class VGG_16:
    def __init__(self) -> None:
        self.vgg16_model = VGG16(weights="imagenet")
        self.model = Model(inputs=self.vgg16_model.inputs, outputs = self.vgg16_model.get_layer("fc1").output)
    # Ham tien xu ly, chuyen doi hinh anh thanh tensor
    def image_preprocess(self,img):
        img = img.resize((224,224))
        img = img.convert("RGB")
        x = image.img_to_array(img)
        x = np.expand_dims(img, axis=0)
        x = preprocess_input(x)
        return x

    def extract_features(self,image_path):
        print("Xu ly : ", image_path)
        img = Image.open(image_path)
        img_tensor = self.image_preprocess(image_path)

        # Trich dac trung
        vector = self.model.predict(img_tensor)[0]
        # Chuan hoa vector = chia chia L2 norm (tu google search)
        vector = vector / np.linalg.norm(vector)
        return vector
    
    def preprocess(self,img):
        x = np.expand_dims(img, axis=0)
        x = preprocess_input(x)
        return x

    def Vgg_extract(self,image):
        img_tensor = self.preprocess(image)

        # Trich dac trung
        vector = self.model.predict(img_tensor)[0]
        # Chuan hoa vector = chia chia L2 norm (tu google search)
        vector = vector / np.linalg.norm(vector)
        return vector
    