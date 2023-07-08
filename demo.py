import time
import torch
import faiss
import pathlib
from PIL import Image
from helpers import paginator
import streamlit as st
from streamlit_cropper import st_cropper

from feature_module import MyVGG16, MyResnet50, RGBHistogram, LBP
from dataloader import get_transformation

st.set_page_config(layout="wide")

device = torch.device('cpu')
image_root = 'dataset/black_dress'
feature_root = 'feature'


def get_image_list(image_root):
    image_root = pathlib.Path(image_root)
    image_list = list()
    for image_path in image_root.iterdir():
        if image_path.exists():
            image_list.append(image_path)
    image_list = sorted(image_list, key = lambda x: x.name)
    return image_list

def retrieve_image(img, feature_extractor):
    if (feature_extractor == 'VGG16'):
        extractor = MyVGG16(device)
    elif (feature_extractor == 'Resnet50'):
        extractor = MyResnet50(device)
    elif (feature_extractor == 'RGBHistogram'):
        extractor = RGBHistogram(device)
    elif (feature_extractor == 'LBP'):
        extractor = LBP(device)

    transform = get_transformation()

    img = img.convert('RGB')
    image_tensor = transform(img)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    feat = extractor.extract_features(image_tensor)

    indexer = faiss.read_index(feature_root + '/' + feature_extractor + '.index.bin')

    _, indices = indexer.search(feat, k=20)

    return indices[0]

def main():
    st.title('CONTENT-BASED IMAGE RETRIEVAL')
    
    col1, col2 = st.columns(2)

    with col1:
        st.header('QUERY')

        st.subheader('Choose feature extractor')
        option = st.selectbox('.', ( 'Resnet50', 'VGG16', 'RGBHistogram', 'LBP'))

        st.subheader('Upload image')
        img_file = st.file_uploader(label='.', type=['png', 'jpg'])

        if img_file:
            img = Image.open(img_file)
            # Get a cropped image from the frontend
            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0004')
            
            # Manipulate cropped image at will
            st.write("Preview")
            _ = cropped_img.thumbnail((150,150))
            st.image(cropped_img)

    with col2:
        st.header('RESULT')
        if img_file:
            st.markdown('**Retrieving .......**')
            start = time.time()

            retriev = retrieve_image(cropped_img, option)
            image_list = get_image_list(image_root)
            print(image_list)
            print(retriev)

            end = time.time()
            st.markdown('**Finish in ' + str(end - start) + ' seconds**')

            result_images = []
            len(retriev)
            for u in range(len(retriev)):
                image = Image.open(image_list[retriev[u]])
                
                result_images.append(image)
            image_iterator = paginator("Select the total page", result_images)
            indices_on_page, images_on_page = map(list, zip(*image_iterator))
            st.image(images_on_page, width=200, caption=indices_on_page)


            

if __name__ == '__main__':
    main()