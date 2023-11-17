import streamlit as st
import pandas

st.title("Hello World")
st.write("hello Here's our first attempt at using data to create a table:")
st.write(pandas.DataFrame({
    'first': [1, 2, 3, 4],
    'second': [10, 20, 30, 40]}))

