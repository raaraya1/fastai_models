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
#https://drive.google.com/file/d/1VKUk-t-1jswDlZDlak8nj7giqc9oD8r2/view?usp=sharing

def get_x(r):
  return path/'train'/r['fname']

def get_y(r):
    return r['labels'].split(' ')

def splitter(df):
    train = df.index[~df['is_valid']].tolist()
    valid = df.index[df['is_valid']].tolist()
    return train,valid

class multi_class_st():
    def __init__(self):
        pass

    def model(self):
        #custom()
        # para cargar el modelo
        plt = platform.system()
        if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath
        temp = pathlib.PosixPath

        path = Path.cwd()
        file_id = '1VKUk-t-1jswDlZDlak8nj7giqc9oD8r2'
        destination = 'multi_class.plk'
        download_file_from_google_drive(file_id, destination)
        #st.write(str(path))
        #st.write(str(os.listdir()))

        path = Path(str(path) + '/multi_class.plk')
        learn = load_learner(path)

        # Haciendo la prediccion
        archivo = st.file_uploader('Colaca la imagen de una raza de gato o perro')
        if archivo:
            st.image(archivo, width=128)
            img = PILImage.create(archivo)
            prediccion = learn.predict(img)
            prob = int(np.round(torch.max(prediccion[2])*100, 0))
            st.write(str(prediccion))
            st.write(f'''
            Se predice que es un {prediccion[0]} con una probabilidad del {prob}%
            ''')
