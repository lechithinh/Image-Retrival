import streamlit as st
from streamlit_option_menu import option_menu

#for search module
from Extract_module import Descriptor
from Search_module import Searcher
import cv2
import numpy as np
from helpers import paginator




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
        st.write("Overview")
    elif selected == "Search System":
        upload_search, url_search = st.tabs(["Uploaded Image", "Url image"])
        with upload_search:
            #upload image
            uploaded_file = st.file_uploader('')
            
            #remove file name 
            st.markdown('''
            <style>
                .uploadedFile {display: none}
            <style>''',
            unsafe_allow_html=True)
            
            #configure
           

            with st.form("Room inforamtion"):                                 
                with st.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        limit_image = st.slider('Select the limit image retrival', 0, 50, 1)  #the max = len (index.csv - row)  
                    with col2:
                        search_models = st.selectbox(
                            'Select search metrics',
                                ('Chi2', 'Euclidean', 'Hellinger', 'Cosine'))
                    search_button = st.form_submit_button("Search", type="primary")
            
            #models
            cd = Descriptor((8, 12, 3))
            searcher = Searcher(indexPath="index.csv", limit_image = limit_image)
            
            if uploaded_file is not None:
                results = []
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1)
                st.image(opencv_image, channels="BGR", width=250)
                
                if uploaded_file is not None and search_button: 
                    #search image
                    features = cd.Historam_extract(opencv_image)
                    
                    if search_models == "Chi2":
                        results = searcher.Search('Chi2', features)
                    elif search_models == "Euclidean":
                        results = searcher.Search('Euclidean', features)
                        #call the code
                    elif search_models == 'Hellinger':
                        results = searcher.Search('Hellinger', features)
                    elif search_models == "Cosine":
                        results = searcher.Search('Cosine', features)

                        
                    
                    #show a list of images
                    result_images = []
                    for (score, resultID) in results:
                        result_images.append(resultID)
                        print(score, resultID)
                    
                    image_iterator = paginator("Select the total page", result_images)
                    indices_on_page, images_on_page = map(list, zip(*image_iterator))
                    st.image(images_on_page, width=200, caption=indices_on_page)
                        

                    
        with url_search:
            pass
            