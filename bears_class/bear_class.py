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
#https://drive.google.com/file/d/15dDQA8SJhdW7LIhS09Pzjo1QitpSQ88M/view?usp=sharing


class bear_class_st():
    def __init__(self):
        pass

    def model(self):

        st.write('''
        ## Clasificador de Osos

        Este modelo, al igual que con el modelo de clasificación de gatos y perros,
        utiliza un modelo de red convolucional pre-entrenado. Así, lo que se hace es
        cambiar la salida del modelo adaptándolo a esta nueva tarea de **diferenciar
        entre imágenes de Osos Grizzlies, Osos Negros y Osos Teddy**.

        En particular, uno de los objetivos propuestos para la segunda clase del curso
        era el de elaborar un clasificador propio, construyendo, primeramente, un
        banco de imágenes para luego reentrenar el modelo.

        En consecuencia, te comento que si buscas una manera alternativa para generar
        este banco de imágenes, puedes como lo hice yo en mis notas del curso (con
        las bibliotecas de **BeautifulSoup o MechanicalSoup**).

        ### Datos

        Las imágenes fueron extraídas mediante web scraping con el motor de
        búsqueda de Bing y la utilización de la biblioteca BeatifulSoup.

        - **Grizzly bear**: 150 - 170 imágenes
        - **Black bear**: 150 - 160 imágenes
        - **Teddy bear**: 140 - 140 imágenes

        ### Modelo

        **resnet18**: Red neuronal convolucional pre-entrenada extraída desde:
        https://download.pytorch.org/models/resnet18-f37072fd.pth

        ### DEMO
        ''')
        #custom()
        # para cargar el modelo
        #pathlib.PosixPath = pathlib.WindowsPath
        plt = platform.system()
        if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath
        temp = pathlib.PosixPath

        path = Path.cwd()
        file_id = '15dDQA8SJhdW7LIhS09Pzjo1QitpSQ88M'
        destination = 'm_bear_class.plk'
        download_file_from_google_drive(file_id, destination)

        path = Path(str(path) + '/m_bear_class.plk')
        learn = load_learner(path)

        # Haciendo la prediccion
        st.write('**Coloca la imagen de un oso (Grizzly, Black o Teddy)**')
        archivo = st.file_uploader('')
        col1, col2, col3 = st.columns(3)
        if archivo:
            col2.image(archivo, width=128)
            img = PILImage.create(archivo)
            prediccion = learn.predict(img)
            prob = int(np.round(torch.max(prediccion[2])*100, 0))
            #st.write(str(prediccion))
            st.write(f'''
            Se predice que la clase es **{prediccion[0]}** con una probabilidad del **{prob}%**
            ''')
