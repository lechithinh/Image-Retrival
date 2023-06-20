import streamlit as st
import streamlit_authenticator as stauth
from app import Webapp
import time

names = ['Thinh Le', 'Thien', "Tony Tri"]
usernames = ['lechithinh', 'huynhcongthien', 'nguyenminhtri']
passwords = ['123', '456', '789']
#configure streamlit width
#st.set_page_config(layout="wide")

hashed_passwords = stauth.Hasher(passwords).generate()
credentials = {"usernames":{}}
for user, name, pw in zip(usernames, names, hashed_passwords):
    user_dict = {"name":name,"password":pw}
    credentials["usernames"].update({user:user_dict})

#Authenticate 
authenticator = stauth.Authenticate(credentials, "HMS", "auth", cookie_expiry_days=0)

#Login Panel
with st.sidebar: 
    name, authentication_status, username = authenticator.login('Login', 'main')



if st.session_state['authentication_status']: #Login successfully
    time.sleep(1)
    Webapp()
    with st.sidebar:
        st.divider()
        authenticator.logout('Logout', 'sidebar')

elif st.session_state['authentication_status'] == False: 
    #Our demo image
    st.image('http://placekitten.com/300/250')
    st.sidebar.error('Username/password is incorrect')
elif st.session_state['authentication_status'] == None:
    #Our demo image
    st.image('http://placekitten.com/300/250')
    st.sidebar.warning('Please enter your username and password')