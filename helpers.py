import streamlit as st
import numpy as np
import faiss
from PIL import Image

def get_faiss_indexer(shape):

    indexer = faiss.IndexFlatL2(shape) # features.shape[1]

    return indexer

def paginator(label, items, items_per_page=50, on_sidebar=True):
    # Figure out where to display the paginator
    if on_sidebar:
        location = st.sidebar.empty()
    else:
        location = st.empty()

    # Display a pagination selectbox in the specified location.
    items = list(items)
    n_pages = len(items)
    n_pages = (len(items) - 1) // items_per_page + 1
    page_format_func = lambda i: "Page %s" % i
    page_number = location.selectbox(label, range(n_pages), format_func=page_format_func)

    # Iterate over the items in the page to let the user display them.
    min_index = page_number * items_per_page
    max_index = min_index + items_per_page
    import itertools
    return itertools.islice(enumerate(items), min_index, max_index)

def initialize_result(results, search_model):
    labels = {'Labels': [],
              'Values': []}
    
    for i in range(len(results)):
        if search_model == "Faiss":
            label = (results[i].split('\\')[-1]).split(' ')[0]
        else:
            label = (results[i].split('/')[-1]).split(' ')[0]
        if label not in labels['Labels']:
            labels['Labels'].append(label)
            labels['Values'].append(1)
        else:
            indx = list(labels['Labels']).index(label)
            labels['Values'][indx] += 1
    return labels


def LoginPageInfor():
    st.markdown(f'''
            <h2 style='text-align: center;  color: black;'>Image Retrieval System </h2>
            ''', unsafe_allow_html=True)
    st.write(f"Introducing our game-changing **image retrieval system**. With advanced algorithms and machine learning, it revolutionizes image searches, delivering precise results effortlessly. Say goodbye to endless scrolling and hello to a new era of image discovery.")
    col1, col2, col3 = st.columns(3)
    with col1: 
        image1 = Image.open('assets/1.png')
        st.image(image1, width=300, caption="Image Search")
    with col2:
        image3 = Image.open('assets/2.png')
        st.image(image3, width=300, caption="Visualization")
    with col3: 
        image3 = Image.open('assets/3.png')
        st.image(image3, width=300, caption="Camera Search")
      

