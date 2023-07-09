import streamlit as st
import streamlit_authenticator as stauth
from app import Webapp
import time
from helpers import LoginPageInfor
import json

# Load data from the JSON file
with open('credentials.json', 'r') as file:
    loaded_data = json.load(file)

# Access the loaded data
names = loaded_data['names']
usernames = loaded_data['usernames']
passwords = loaded_data['passwords']


hashed_passwords = stauth.Hasher(passwords).generate()
credentials = {"usernames":{}}
for user, name, pw in zip(usernames, names, hashed_passwords):
    user_dict = {"name":name,"password":pw}
    credentials["usernames"].update({user:user_dict})

#Authenticate 
authenticator = stauth.Authenticate(credentials, "IRS", "auth", cookie_expiry_days=0)

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
    st.sidebar.error('Username/password is incorrect')
    LoginPageInfor()
elif st.session_state['authentication_status'] == None:
    st.sidebar.warning('Please enter your username and password')
    LoginPageInfor()