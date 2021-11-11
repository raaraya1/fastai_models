import streamlit as st
#import fastbook
from fastai.vision.all import *
from fastai.tabular.all import *
import pathlib
from PIL import Image
import numpy as np
import requests
import urllib.request
from request_from_drive import *
import os
from custom_streamlit import custom
#https://drive.google.com/file/d/1dPFRDG0UaXkjGcGxwDG3e_CVOOjY0gZ7/view?usp=sharing

class rating_movies_st():
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
        file_id = '1dPFRDG0UaXkjGcGxwDG3e_CVOOjY0gZ7'
        destination = 'm_rating_movies.plk'
        download_file_from_google_drive(file_id, destination)
        #st.write(str(path))
        #st.write(str(os.listdir()))

        path = Path(str(path) + '/m_rating_movies.plk')
        learn = load_learner(path)

        learn.show_results()


        # Haciendo la prediccion
        archivo = st.file_uploader('Colaca la imagen de un gato o perro')
        if archivo:
            st.image(archivo, width=128)
            img = PILImage.create(archivo)
            prediccion = learn.predict(img)
            prob = np.round(prediccion[2][1]*100, 0)
            if prediccion[0] == 'True':
                st.write(f'Se predice que es un gato con una probabilidad del {prob}%')
            else:
                st.write(f'Se predice que es un perro con una probabilidad del {100-prob}%')
