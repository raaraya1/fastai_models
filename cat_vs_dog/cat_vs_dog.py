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
import gdown
#https://drive.google.com/file/d/1_pyJdn4pIIp5UVU1poidRh0oiE5iTGNq/view?usp=sharing
#1_pyJdn4pIIp5UVU1poidRh0oiE5iTGNq

def load_learner_(fname, cpu=True, pickle_module=pickle):
    "Load a `Learner` object in `fname`, by default putting it on the `cpu`"
    distrib_barrier()
    map_loc = 'cpu' if cpu else default_device()
    try: res = torch.load(fname, map_location=map_loc, encoding='latin1')
    except AttributeError as e: 
        e.args = [f"Custom classes or functions exported with your `Learner` not available in namespace.\Re-declare/import before loading:\n\t{e.args[0]}"]
        raise
    if cpu: 
        res.dls.cpu()
        if hasattr(res, 'channels_last'): res = res.to_contiguous(to_fp32=True)
        elif hasattr(res, 'mixed_precision'): res = res.to_fp32()
        elif hasattr(res, 'non_native_mixed_precision'): res = res.to_non_native_fp32()
    return res


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
        #download_file_from_google_drive(file_id, destination)
        #st.write(str(path))
        #st.write(str(os.listdir()))

        path = Path(str(path) + '/m_cat_vs_dog.plk')
        url = "https://drive.google.com/u/0/uc?id=1_pyJdn4pIIp5UVU1poidRh0oiE5iTGNq"
        gdown.download(url, path, quiet=False)
        
        learn = load_learner_(path)

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
