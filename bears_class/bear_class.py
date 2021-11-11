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
        archivo = st.file_uploader('Colaca la imagen de un gato o perro')
        if archivo:
            st.image(archivo, width=128)
            img = PILImage.create(archivo)
            prediccion = learn.predict(img)
            prob = int(np.round(torch.max(prediccion[2])*100, 0))
            st.write(str(prediccion))
            st.write(f'''
            Se predice que la clase es **{prediccion[0]}** con una probabilidad del **{prob}%**
            ''')
