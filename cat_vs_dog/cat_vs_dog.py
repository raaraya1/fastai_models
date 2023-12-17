import streamlit as st
#import fastbook
from fastai.vision.all import *
import pathlib
from PIL import Image
import numpy as np
import requests
import urllib.request
from request_from_drive import *
import os
from custom_streamlit import custom
from functools import partial
import pickle
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
#https://drive.google.com/file/d/1_pyJdn4pIIp5UVU1poidRh0oiE5iTGNq/view?usp=sharing
#https://drive.google.com/file/d/1_pyJdn4pIIp5UVU1poidRh0oiE5iTGNq/view?usp=sharing
#1_pyJdn4pIIp5UVU1poidRh0oiE5iTGNq

def is_cat(x):
    return x[0].isupper()

class cat_vs_dog_st():
    def __init__(self):
        pass

    def model(self):


        st.write('''
        ## **CAT** vs **DOG**

        Este es el modelo introductorio del curso. En este nos enseñan, de una
        manera corta y sencilla, la manera de elaborar un clasificador que logre
        distinguir entre la imagen de un gato o un perro.

        ### Datos

        La base de datos utilizada se extrajo del siguiente enlace: http://www.robots.ox.ac.uk/~vgg/data/pets/
        - 7349 imágenes en total

        ### Modelo

        Para la elaboración del modelo se utilizó uno ya pre-entrenado con otro set
        de imágenes. Este se puede descargar en el siguiente enlace https://download.pytorch.org/models/resnet34-b627a593.pth

        Así, una de las primeras particularidades que nos enseñan en el curso es
        que podemos hacer uso de modelos ya pre-entrenados y ajustarlos a nuestros
        objetivos.

        ### DEMO
        ''')
        # para cargar el modelo
        plt = platform.system()
        if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath
        temp = pathlib.PosixPath

        path = Path.cwd()
        file_id = '1_pyJdn4pIIp5UVU1poidRh0oiE5iTGNq'
        destination = 'm_cat_vs_dog.plk'
        download_file_from_google_drive(file_id, destination)
        #st.write(str(path))
        #st.write(str(os.listdir()))

        path = Path(str(path) + '/m_cat_vs_dog.plk')
        learn = load_learner(path)

        # DEMO

        # Haciendo la prediccion
        st.write('''
        #### Coloca la imagen de un **gato** o un **perro**.
        ''')
        archivo = st.file_uploader('')
        col1, col2, col3 = st.columns(3)
        if archivo:
            col2.image(archivo, width=128)
            img = PILImage.create(archivo)
            prediccion = learn.predict(img)
            prob = np.round(prediccion[2][1]*100, 0)
            if prediccion[0] == 'True':
                st.write(f'Se predice que es un **gato** con una probabilidad del {prob}%')
            else:
                st.write(f'Se predice que es un **perro** con una probabilidad del {100-prob}%')
