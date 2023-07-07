import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_cropper import st_cropper
import requests
from io import BytesIO
from PIL import Image
from helpers import initialize_result
import pandas as pd

# for search module
from simple_module import Descriptor
from Search_module import Searcher
import cv2
import numpy as np
from helpers import paginator
from vgg_module import VGG_16
from timer import Timer
from extract_features import Extract_feature





def Webapp():
    with st.sidebar:
        selected = option_menu(f"Main Menu", ["Overview",'Extract Features', 'Search System' ],
                               icons=['kanban-fill', 'grid-1x2-fill'],
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
                        dataset_path = st.text_input("Enter the path of dataset")
                    with col2:
                        feature_names = st.multiselect(
                                'Select the feature options',
                                ['histogram','hog','vgg'],
                                ['histogram', 'hog','vgg'])
                    extract_button = st.form_submit_button(
                        "Search", type="primary")
                    
                    if extract_button:
                        Extract_feature(dataset_path = dataset_path, feature_path = 'Test', feature_names= feature_names)
    
    elif selected == "Search System":
        st.subheader('Upload image')
        upload_search, url_search = st.tabs(["Uploaded Image", "Url image"])

        with upload_search:
            st.subheader('Choose image to upload')
            # upload image
            uploaded_file = st.file_uploader('')
            # remove file name
            st.markdown('''
            <style>
                .uploadedFile {display: none}
            <style>''',
                        unsafe_allow_html=True)

            # configure
            image_mode = st.selectbox("Select image mode", ["Full Image", "Crop Image"])
   
                  
            # Start timer
            timer = Timer()
            timer.tic()

            if uploaded_file is not None:
                results = []

                #convert uploaded file to numpy array
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1)

                #query image is full size
                query_image = opencv_image.copy()
                uploaded_img = Image.open(uploaded_file)
                
                #resize image 
                _ = uploaded_img.thumbnail((500,500))
                
                if image_mode == "Full Image":
                    with st.form("Full Image"):
                        with st.container():
                            col1, col2= st.columns(2)
                            with col1:
                                option = st.selectbox('Choose feature extraction', ('VGG16', 'Histogram', 'HOG'))
                                search_models = st.selectbox(
                                    'Select search metrics',
                                    ('Chi2', 'Euclidean', 'Hellinger', 'Cosine'))
                                limit_image = st.slider(
                                    'Select the limit image retrival', 0, 50, 1)
                                search_button = st.form_submit_button(
                                "Search", type="primary")
                            with col2:
                                st.image(uploaded_img)
                        
                elif image_mode == "Crop Image":
                    with st.form("Crop Image"):
                        with st.container():
                            column1, column2, column3 = st.columns(3)
                            with column1:
                                option = st.selectbox('Choose feature extraction', ('VGG16', 'Histogram', 'HOG'))
                            with column2:
                                search_models = st.selectbox(
                                    'Select search metrics',
                                    ('Chi2', 'Euclidean', 'Hellinger', 'Cosine'))
                            with column3:    
                                limit_image = st.slider(
                                    'Select the limit image retrival', 0, 50, 1)
                            search_button = st.form_submit_button(
                                "Search", type="primary")
                    co1, co2 = st.columns(2)
                    with co1:
                        cropped_img = st_cropper(uploaded_img, realtime_update=True, box_color='#FF0004', aspect_ratio=(1,1))
                        _ = cropped_img.thumbnail((500,500))
                    cropped_arr = np.asarray(cropped_img, dtype=np.uint8)
                    with co2:
                        st.image(cropped_img, use_column_width=True)
                        
                    #query image is cropped
                    query_image = cropped_arr
                    
                if option == 'Histogram':
                    cd = Descriptor((8, 12, 3))
                    searcher = Searcher(indexPath="Features/histogram.csv", limit_image=limit_image) 
                elif option == 'HOG':
                    cd = Descriptor((8, 12, 3))
                    searcher = Searcher(indexPath="Features/hog.csv", limit_image=limit_image) 
                elif option == 'VGG16':
                    searcher = Searcher(indexPath="Features/vgg.csv", limit_image=limit_image) 
                
     
                if uploaded_file is not None and search_button:
                    #Extract the query feature
                    if option == 'Histogram':
                        features = cd.Historam_extract(query_image)
                    elif option == 'VGG16': #cannot except crop image
                        model = VGG_16()
                        features = model.Vgg_extract(opencv_image)
                    elif option == 'HOG':
                        features = cd.Hog_extract(query_image)

                    #Start search function
                    results = searcher.Search(search_models, features)

                        
                    #show the time - Chỉ cái này lại cho đẹp nha
                    Durationtime = timer.toc()
                    timer.clear()

                    # show a list of images
                    result_images = []
                    for (score, resultID) in results:
                        result_images.append(resultID)
                        print(score, resultID)
                    st.markdown(
                        f"""
                            <div style="color: red; font-size: 24px; font-weight: bold;">Durationtime: {Durationtime}</div>
                            """,
                        unsafe_allow_html=True
                    )
                    
                    image_iterator = paginator("Select the total page", result_images)
                    indices_on_page, images_on_page = map(list, zip(*image_iterator))
                    st.image(images_on_page, width=200, caption=indices_on_page)

                    #show the chart
                    labels = dict()
                    labels = initialize_result(results)
                    st.bar_chart(pd.DataFrame(labels.values(), labels.keys()))

        with url_search:
            url = st.text_input('URL address', placeholder="Enter url address")
            with st.form("Retrival information"):
                with st.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        # the max = len (index.csv - row)
                        option = st.selectbox('Choose feature extraction', ('VGG16', 'Histogram', 'HOG'))
                    with col2:
                        search_models = st.selectbox(
                            'Select search metrics',
                            ('Chi2', 'Euclidean', 'Hellinger', 'Cosine'))
                    search_button = st.form_submit_button(
                        "Search", type="primary")
                    with col3:
                        limit_image = st.slider(
                            'Select the limit image retrival', 0, 50, 1)
            if url == '':
                st.markdown(
                    f"""
                            <div style="color: red; font-size: 24px; font-weight: bold;">Input URL to retrie similar image</div>
                            """, unsafe_allow_html=True)
            elif url != '':
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
                img_bytes = BytesIO()
                img.save(img_bytes, format="JPEG")
                img_bytes.seek(0)

                timer = Timer()
                timer.tic()
                if option == 'Histogram':
                    cd = Descriptor((8, 12, 3))
                    searcher = Searcher(indexPath="index.csv",limit_image=limit_image) #edit indexPath
                elif option == 'HOG':
                    cd = Descriptor((8, 12, 3))
                    searcher = Searcher(indexPath="index.csv",limit_image=limit_image) #edit indexPath
                elif option == 'VGG16':
                    searcher = Searcher(indexPath="index.csv", limit_image=limit_image) #edit indexPath

                if url is not None:
                    results = []
                    file_bytes = np.asarray(
                        bytearray(img_bytes.read()), dtype=np.uint8)
                    opencv_image = cv2.imdecode(file_bytes, 1)
                    # st.image(opencv_image, channels="BGR", width=250)
                    upload_img = Image.open(img_bytes)
                    
                    #crop image
                    cropped_img = st_cropper(upload_img, realtime_update=True, box_color='#FF0004')
                    
                    st.write("Preview")
                    _ = cropped_img.thumbnail((150,150))

                    cropped_arr = np.asarray(cropped_img, dtype=np.uint8)
                
                    st.image(cropped_img)


                    if url is not None and search_button:
                        # search image
                        if option == 'Histogram':
                            features = cd.Historam_extract(cropped_arr)
                        elif option == 'HOG':
                            features = cd.Hog_extract(cropped_arr)
                        elif option == 'VGG16':
                            pass

                        if search_models == "Chi2":
                            results = searcher.Search('Chi2', features)
                        elif search_models == "Euclidean":
                            results = searcher.Search('Euclidean', features)
                            # call the code
                        elif search_models == 'Hellinger':
                            results = searcher.Search('Hellinger', features)
                        elif search_models == "Cosine":
                            results = searcher.Search('Cosine', features)
                        Durationtime = timer.toc()
                        timer.clear()

                        # show a list of images
                        result_images = []
                        for (score, resultID) in results:
                            result_images.append(resultID)
                            print(score, resultID)
                        st.markdown(
                            f"""
                                <div style="color: red; font-size: 24px; font-weight: bold;">Durationtime: {Durationtime}</div>
                                """,
                            unsafe_allow_html=True
                        )
                        image_iterator = paginator(
                            "Select the total page", result_images)
                        indices_on_page, images_on_page = map(
                            list, zip(*image_iterator))
                        st.image(images_on_page, width=200,
                                 caption=indices_on_page)
                        labels = dict()
                    
                        labels = initialize_result(results)
                        
                        st.bar_chart(pd.DataFrame(labels.values(), labels.keys()))
