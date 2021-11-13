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

        st.write('''
        ## Multiclass Classification Model

        Este modelo, al igual que el resto de los modelos de clasificación de imágenes
        anteriormente presentados, hace uso de una red neuronal convolucional previamente entrenada
        para luego ser ajustada en sus últimas capas (incluyendo la capa de salida)
        a nuestros objetivos.

        **Nota:**
        Si tu interés es elaborar modelos de clasificación con más de dos categorías,
        entonces te recomiendo que prestes atención al concepto de **Binary Cross Entropy**,
        ya que esta es la función de perdida utilizada en este modelo.

        ## Datos

        El banco de imágenes es extraído del siguiente enlace: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/
        Así, las imágenes se encuentran ordenas bajo las siguientes categorías:

        - Aeroplanes
        - Bicycles
        - Birds
        - Boats
        - Bottles
        - Buses
        - Cars
        - Cats
        - Chairs
        - Cows
        - Dining tables
        - Dogs
        - Horses
        - Motorbikes
        - People
        - Potted plants
        - Sheep
        - Sofas
        - Trains
        - TV/Monitors

        ## Modelo

        **resnet18**: Red neuronal convolucional pre-entrenada extraída desde:
        https://download.pytorch.org/models/resnet18-f37072fd.pth

        ## DEMO


        ''')
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
        st.write('**Colaca la imagen de alguna de las categorias anteriores**')
        archivo = st.file_uploader('')
        col1, col2, col3 = st.columns(3)
        if archivo:
            col2.image(archivo, width=128)
            img = PILImage.create(archivo)
            prediccion = learn.predict(img)
            prob = int(np.round(torch.max(prediccion[2])*100, 0))
            #st.write(str(prediccion))
            st.write(f'''
            Se predice que es un **{prediccion[0]}** con una probabilidad del **{prob}%**
            ''')
