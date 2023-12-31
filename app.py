import streamlit as st
from streamlit_option_menu import option_menu
import requests
from io import BytesIO
from helpers import initialize_result
import pandas as pd
import math


# for search module
import cv2
import numpy as np
#new libraries
import time
import torch
import faiss
import pathlib
from PIL import Image
from helpers import paginator
from streamlit_cropper import st_cropper
from retrieve import get_image_list, retrieve_image
from indexer import Indexing_feature
from st_aggrid import AgGrid
import altair as alt
from retrieve import Search
device = torch.device('cpu')


def Webapp():
    with st.sidebar:
        selected = option_menu(f"Main Menu", ["Overview",'Extract Features', 'Search System', "Camera"],
                               icons=['kanban-fill', 'grid-1x2-fill', 'search-heart', 'camera-fill'],
                               menu_icon="cast",
                               default_index=0,
                               styles={
            "container": {"padding": "0!important", "background-color": "#f1f2f6"},
        })

    if selected == "Overview":
        st.title('Image Retrival About clothes dataset')
        st.markdown('''
          # ABOUT US \n 
           The website provides a range of impressive features to image retrieval on Clothes Dataset.\n
            
            Here are our fantastic features:
            - **Image Retrieval with upload file**
            - **Image Retrieval with URL address**

        
            Our current version is just the beginning, and we are continually working to improve and expand our offerings. 
            In the next versions, we plan to introduce even more advanced features. We welcome your collaboration and feedback as we strive to create a well-structured and effective platform.
            ''')
    elif selected == "Extract Features":
         with st.form("Extract Features"):
                with st.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        dataset_path = st.text_input("Enter the path of dataset", value="dataset")
                    with col2:
                        feature_descriptor = st.selectbox(
                                'Select the feature options',
                                ['VGG16', 'RGBHistogram', 'LBP'],)
                    extract_button = st.form_submit_button(
                        "Extract", type="primary")
                    
                    if extract_button:
                        try:
                            completed_time = Indexing_feature(dataset_path, feature_descriptor)
                            st.write('Finished in ' + completed_time + ' seconds')
                        except:
                            st.error('Please check the dataset path', icon="🚨")  
                        
                       
    
    elif selected == "Search System":
        st.subheader('Choose image to upload')
        # upload image
        uploaded_file = st.file_uploader('', type=['png', 'jpg'])
        # remove file name
        st.markdown('''
        <style>
            .uploadedFile {display: none}
        <style>''',
                    unsafe_allow_html=True)

        # configure
        image_mode = st.selectbox("Select image mode", ["Full Image", "Crop Image"])

        if uploaded_file is not None:
        
            #convert uploaded file to numpy array
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            uploaded_img = Image.open(uploaded_file)
            query_image = uploaded_img
            #resize to display the image
            _ = uploaded_img.thumbnail((500,500))
            
            if image_mode == "Full Image":
                with st.form("Full Image"):
                    with st.container():
                        col1, col2= st.columns(2)
                        with col1:
                            image_root = st.text_input("Enter the dataset path", value="dataset")
                            option = st.selectbox('Choose feature extraction', ('VGG16', 'RGBHistogram', 'LBP'))
                            search_model = st.selectbox('Choose the search model', ( 'Faiss',"Euclidean", "Cosine" ))
                            limit_image = st.slider(
                                'Select the limit image retrival', 0, 50, 10)
                            search_button = st.form_submit_button(
                            "Search", type="primary")
                        with col2:
                            st.image(uploaded_img)
                    
            elif image_mode == "Crop Image":
                with st.form("Crop Image"):
                    with st.container():
                        column1, column2 = st.columns(2)
                        with column1:
                            image_root = st.text_input("Enter the dataset path", value="dataset")
                            option = st.selectbox('Choose feature extraction', ('VGG16', 'RGBHistogram', 'LBP'))
                        with column2:
                            search_model = st.selectbox('Choose the search model', ('Faiss',"Euclidean", "Cosine" ))
                            limit_image = st.slider(
                                'Select the limit image retrival', 0, 50, 10)
                        search_button = st.form_submit_button(
                            "Search", type="primary")
       
                co1, co2 = st.columns(2)
                with co1:
                    cropped_img = st_cropper(uploaded_img, realtime_update=True, box_color='#FF0004', aspect_ratio=(1,1))
                    _ = cropped_img.thumbnail((500,500))
                    #cropped_arr = np.asarray(cropped_img, dtype=np.uint8)
                with co2:
                    st.image(cropped_img, use_column_width=True)
                    
                #query image is cropped
                query_image = cropped_img
                
                            
            try:
                image_list = get_image_list(image_root)  
                if uploaded_file is not None and search_button:
                    start = time.time()
                    #For faiss result
                    link_images = []
                    
                    result_images = []
                    
                    if search_model == "Faiss":
                        #search image
                        retriev = retrieve_image(query_image, option, limit_image)
                        #get the dataset to display
                        image_list = get_image_list(image_root)
                        for u in range(len(retriev)):
                            image = Image.open(image_list[retriev[u]])
                            result_images.append(image)
                            link_images.append(str(image_list[retriev[u]]))
                            print(link_images)
                                                        
                    else:
                        results = Search(query_image,option,search_model)
                        for (score, resultID) in results:
                            result_images.append(resultID)
                            print(score, resultID)
                    #time
                    end = time.time()
                    finished_time = str(math.floor(end - start))
                    #display result images
                    image_iterator = paginator("Select the total page", result_images)
                    indices_on_page, images_on_page = map(list, zip(*image_iterator))
                    st.image(images_on_page, width=230, caption=indices_on_page)
                    
                    st.markdown('**Finish in ' + finished_time + ' seconds**')
                    #show the chart
                    if search_model == "Faiss":
                        labels = initialize_result(link_images, search_model)
                    else:
                        labels = initialize_result(result_images, search_model)
                    
                    #Show the table
                    st.subheader('Statistics')
                    st.data_editor(pd.DataFrame(labels), use_container_width= True, hide_index=1)
                    
                    
                    #Show the bar chart
                    st.subheader('Visualization')
                    chart = (
                        alt.Chart(pd.DataFrame(labels))
                        .mark_bar()
                        .encode(
                            x='Labels',
                            y='Values',
                            color='Labels:N'
                        ).interactive()
                        
                    )
                    st.altair_chart(chart, use_container_width =True)
            except:
                st.error('Please check the dataset path', icon="🚨")      
            
            
    elif selected == "Camera":
        picture = st.camera_input("Take the picture to search")
        # remove file name
        st.markdown('''
        <style>
            .uploadedFile {display: none}
        <style>''',
                    unsafe_allow_html=True)

        # configure
        image_mode = st.selectbox("Select image mode", ["Full Image", "Crop Image"])

        if picture:
            #convert uploaded file to numpy array
            file_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            uploaded_img = Image.open(picture)
            query_image = uploaded_img
            #resize to display the image
            _ = uploaded_img.thumbnail((500,500))
            
            if image_mode == "Full Image":
                with st.form("Full Image"):
                    with st.container():
                        col1, col2= st.columns(2)
                        with col1:
                            image_root = st.text_input("Enter the dataset path", value="dataset")
                            option = st.selectbox('Choose feature extraction', ('VGG16', 'RGBHistogram', 'LBP'))
                            search_model = st.selectbox('Choose the search model', ( 'Faiss',"Euclidean", "Cosine" ))
                            limit_image = st.slider(
                                'Select the limit image retrival', 0, 50, 10)
                            search_button = st.form_submit_button(
                            "Search", type="primary")
                        with col2:
                            st.image(uploaded_img)
                    
            elif image_mode == "Crop Image":
                with st.form("Crop Image"):
                    with st.container():
                        column1, column2 = st.columns(2)
                        with column1:
                            image_root = st.text_input("Enter the dataset path", value="dataset")
                            option = st.selectbox('Choose feature extraction', ('VGG16', 'RGBHistogram', 'LBP'))
                        with column2:
                            search_model = st.selectbox('Choose the search model', ('Faiss',"Euclidean", "Cosine" ))
                            limit_image = st.slider(
                                'Select the limit image retrival', 0, 50, 10)
                        search_button = st.form_submit_button(
                            "Search", type="primary")
       
                co1, co2 = st.columns(2)
                with co1:
                    cropped_img = st_cropper(uploaded_img, realtime_update=True, box_color='#FF0004', aspect_ratio=(1,1))
                    _ = cropped_img.thumbnail((500,500))
                    #cropped_arr = np.asarray(cropped_img, dtype=np.uint8)
                with co2:
                    st.image(cropped_img, use_column_width=True)
                    
                #query image is cropped
                query_image = cropped_img
           
            #check the path is valid    
            try:
                image_list = get_image_list(image_root)
                                
                if picture is not None and search_button:
                    start = time.time()
                    #For faiss result
                    link_images = []
                    
                    result_images = []
                    
                    if search_model == "Faiss":
                        #search image
                        retriev = retrieve_image(query_image, option, limit_image)
                        #get the dataset to display
                    
                        image_list = get_image_list(image_root)
                        for u in range(len(retriev)):
                            image = Image.open(image_list[retriev[u]])
                            result_images.append(image)
                            link_images.append(str(image_list[retriev[u]]))
                            print(link_images)
                                                        
                    else:
                        results = Search(query_image,option,search_model)
                        for (score, resultID) in results:
                            result_images.append(resultID)
                            print(score, resultID)
                    #time
                    end = time.time()
                    finished_time = str(math.floor(end - start))
                    #display result images
                    image_iterator = paginator("Select the total page", result_images)
                    indices_on_page, images_on_page = map(list, zip(*image_iterator))
                    st.image(images_on_page, width=230, caption=indices_on_page)
                    
                    st.markdown('**Finish in ' + finished_time + ' seconds**')
                    #show the chart
                    if search_model == "Faiss":
                        labels = initialize_result(link_images, search_model)
                    else:
                        labels = initialize_result(result_images, search_model)
                    
                    #Show the table
                    st.subheader('Statistics')
                    st.data_editor(pd.DataFrame(labels), use_container_width= True, hide_index=1)
                    
                    #Show the bar chart
                    st.subheader('Visualization')
                    chart = (
                        alt.Chart(pd.DataFrame(labels))
                        .mark_bar()
                        .encode(
                            x='Labels',
                            y='Values',
                            color='Labels:N'
                        ).interactive()
                        
                    )
                    st.altair_chart(chart, use_container_width =True)
            except:
                st.error('Please check the dataset path', icon="🚨")  
