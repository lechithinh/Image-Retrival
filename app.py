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


# model = VGG_16()


def Webapp():
    with st.sidebar:
        selected = option_menu(f"Main Menu", ["Overview", 'Search System'],
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

            with st.form("Room inforamtion"):
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

            # models
            timer = Timer()
            timer.tic()
            #if option is histogram -> query in histogram database 
            #if option is hog -> query in hog database
            if option == 'Histogram':
                cd = Descriptor((8, 12, 3))
                searcher = Searcher(indexPath="index.csv", limit_image=limit_image) #edit indexPath = "histogram.csv"
            elif option == 'HOG':
                cd = Descriptor((8, 12, 3))
                searcher = Searcher(indexPath="index.csv", limit_image=limit_image) #edit indexPath = 'hog.csv'
            elif option == 'VGG16':
                searcher = Searcher(indexPath="index.csv", limit_image=limit_image) #edit indexPath = 'vgg16.csv'

            if uploaded_file is not None:
                results = []


                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                # file_bytes = np.asarray(bytearray(cropped_img.tobytes()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1)
                size_o_image = opencv_image.shape
    
                #visualize image in order to crop
                uploaded_img = Image.open(uploaded_file)
                #crop image
                cropped_img = st_cropper(uploaded_img, realtime_update=True, box_color='#FF0004')
                
                st.write("Preview")
                _ = cropped_img.thumbnail((150,150))

                cropped_arr = np.asarray(cropped_img, dtype=np.uint8)
            
                st.image(cropped_img)




                image = opencv_image.copy()
                # Resize the image
                resized_img = cv2.resize(image, (224, 224))
                # Convert the image to RGB format
                rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

                if uploaded_file is not None and search_button:
                    # search image
                    if option == 'Histogram':
                        features = cd.Historam_extract(cropped_arr)
                    elif option == 'VGG16':
                        pass
                        # features = model.Vgg_extract(cropped_img)
                    elif option == 'HOG':
                        features = cd.Hog_extract(cropped_arr)


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
                    print(results)
                    image_iterator = paginator("Select the total page", result_images)
                    indices_on_page, images_on_page = map(list, zip(*image_iterator))
                    st.image(images_on_page, width=200, caption=indices_on_page)

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
