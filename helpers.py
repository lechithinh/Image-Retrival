import streamlit as st
import numpy as np

def paginator(label, items, items_per_page=10, on_sidebar=True):
    # Figure out where to display the paginator
    if on_sidebar:
        location = st.sidebar.empty()
    else:
        location = st.empty()

    # Display a pagination selectbox in the specified location.
    # items = list(items)
    # n_pages = len(items)
    # n_pages = (len(items) - 1) // items_per_page + 1
    # page_format_func = lambda i: "Page %s" % i
    # page_number = location.selectbox(label, range(n_pages), format_func=page_format_func)

    # # Iterate over the items in the page to let the user display them.
    # min_index = page_number * items_per_page
    # max_index = min_index + items_per_page
    # import itertools
    # return itertools.islice(enumerate(items), min_index, max_index)