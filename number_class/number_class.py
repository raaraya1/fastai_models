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
#https://drive.google.com/file/d/13q3ak0bJFjg53KNIxis99YSWjCcig1di/view?usp=sharing
#recordar añadir streamlit-drawable-canvas==0.8.0 a requeriments
from streamlit_drawable_canvas import st_canvas
from matplotlib import cm


class number_class_st():
    def __init__(self):
        pass

    def model(self):

        st.write('''
        ## Clasificador de dígitos

        Similar al resto de los modelos de visión ya presentados, este modelo busca
        clasificar imágenes de números entre el 0 al 9. Para esto también se utiliza
        un modelo pre-entrenado de red convolucional, ajustando las ultimas capas
        a los dígitos.

        **Nota**:
        A diferencia del modelo trabajado en el curso (clasificador de 3 y 7), este
        fue elaborado con un set de datos más completo del banco de imágenes (MNIST).

        ### Datos

        Los datos fueron extraídos del siguiente enlace: http://yann.lecun.com/exdb/mnist/
        Estos se acceden desde **URLs.MNIST**

        ### Modelo

        **resnet18**: Red neuronal convolucional pre-entrenada extraída desde:
        https://download.pytorch.org/models/resnet18-f37072fd.pth

        ### DEMO


        ''')

        #custom()
        # para cargar el modelo
        plt = platform.system()
        if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath
        temp = pathlib.PosixPath

        path = Path.cwd()
        file_id = '13q3ak0bJFjg53KNIxis99YSWjCcig1di'
        destination = 'mnist_model.plk'
        download_file_from_google_drive(file_id, destination)
        #st.write(str(path))
        #st.write(str(os.listdir()))

        path = Path(str(path) + '/mnist_model.plk')
        learn = load_learner(path)

        # para hacer el dibujo del numero
        st.write('**Dibuja un numero del 0 al 9**')
        stroke_width = 7
        stroke_color = "rgb(255, 255, 255)"
        bg_color = "rgb(0, 0, 0)"

        col1, col2, col3 = st.columns(3)
        with col2:
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                update_streamlit='True',
                height=150,
                width=150,
                drawing_mode="freedraw",
            )

        if canvas_result.image_data is not None:
            img = canvas_result.image_data
            #st.image(img.size)

            im = Image.fromarray(img.astype("uint8"), mode="RGBA")
            im = im.convert('RGB').resize(size=(32, 32))
            #st.write(type(im))
            prediccion = col2.button('Predicción')
            if prediccion:
                im = PILImage(im)
                prediccion = learn.predict(im)
                prob = int(np.round(torch.max(prediccion[2])*100, 0))
                st.write(f'''
                Se predice que el numero es el **{prediccion[0]}** con una probabilidad del **{prob}%**
                ''')
                prediccion = False
