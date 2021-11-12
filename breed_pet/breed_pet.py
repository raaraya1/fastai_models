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

class breed_pet_st():
    def __init__(self):
        pass

    def model(self):
        st.write('''
        ## Clasificador de Razas

        El banco de imágenes utilizado para elaborar el clasificador de gatos y
        perros, contemplaba dentro de sus objetivos, hacer la distinción entre
        razas de gatos y razas de perros. Así, para este modelo se utiliza el
        mismo banco de imágenes, pero haciendo la distinción entre razas de mascotas.


        ### Datos

        La base de datos utilizada se extrajo del siguiente enlace: http://www.robots.ox.ac.uk/~vgg/data/pets/
        - 7349 imágenes en total

        ### Modelo

        Para la elaboración del modelo se utilizó uno ya pre-entrenado con otro set
        de imágenes. Este se puede descargar en el siguiente enlace https://download.pytorch.org/models/resnet34-b627a593.pth

        ### DEMO

        ''')

        #custom()
        # para cargar el modelo
        #pathlib.PosixPath = pathlib.WindowsPath
        plt = platform.system()
        if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath
        temp = pathlib.PosixPath

        path = Path.cwd()
        file_id = '1J7Bdg5cj_2huVQedjLEgXR1KJrYarwjb'
        destination = 'breed_class.plk'
        download_file_from_google_drive(file_id, destination)
        #st.write(str(path))
        #st.write(str(os.listdir()))

        path = Path(str(path) + '/breed_class.plk')
        learn = load_learner(path)

        # Haciendo la prediccion
        st.write('**Coloca la imagen de un gato o perro**')
        archivo = st.file_uploader('')
        col1, col2, col3 = st.columns(3)
        if archivo:
            col2.image(archivo, width=128)
            img = PILImage.create(archivo)
            prediccion = learn.predict(img)
            #st.write(str(prediccion))
            prob = int(np.round(torch.max(prediccion[2])*100, 0))
            st.write(f'''
            Se predice que es la raza **{prediccion[0]}** con una probablidad del **{prob} % **''')
