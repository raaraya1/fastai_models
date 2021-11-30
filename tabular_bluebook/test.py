import matplotlib.pyplot as plt
import streamlit as st

st.write('hola')
im = plt.figure(figsize=(10, 6))
st.pyplot(fig=im)
st.write(im)
