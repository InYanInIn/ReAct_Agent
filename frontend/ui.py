import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/debug/"

st.title("Re-Act Code Debugger")

code_text = st.text_area("Enter your code snippet:", "def add(a, b):\n    return a + b")
error_text = st.text_area("Enter the error message you received:")

if st.button("Debug Code"):
    response = requests.post(API_URL, json={"code": code_text, "error_description": error_text})
    if response.status_code == 200:
        result = response.json()["result"]
        st.subheader("Debugged Code")
        st.code(result)
    else:
        st.error("Error: could not get translations from API")
