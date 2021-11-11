import streamlit as st
from fastai.vision.all import *
import pathlib
from PIL import Image
import numpy as np
import requests
import urllib.request
from request_from_drive import *
import os
from custom_streamlit import custom
#from fastbook import *
# https://drive.google.com/file/d/1J7Bdg5cj_2huVQedjLEgXR1KJrYarwjb/view?usp=sharing

class class_pet_st():
    def __init__(self):
        pass

    def model(self):
        #custom()
        # para cargar el modelo
        #pathlib.PosixPath = pathlib.WindowsPath
        plt = platform.system()
        if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath
        temp = pathlib.PosixPath

        path = Path.cwd()
        file_id = '1J7Bdg5cj_2huVQedjLEgXR1KJrYarwjb'
        destination = 'pet_class.plk'
        download_file_from_google_drive(file_id, destination)
        #st.write(str(path))
        #st.write(str(os.listdir()))

        path = Path(str(path) + '/pet_class.plk')
        learn = load_learner(path)

        # Haciendo la prediccion
        archivo = st.file_uploader('Colaca la imagen de una raza de gato o perro')
        if archivo:
            st.image(archivo, width=128)
            img = PILImage.create(archivo)
            prediccion = learn.predict(img)
            #st.write(str(prediccion))
            prob = int(np.round(torch.max(prediccion[2])*100, 0))
            st.write(f'''
            Se predice que es la raza **{prediccion[0]}** con una probablidad del **{prob} % **''')
