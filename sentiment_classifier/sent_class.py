import streamlit as st
#import fastbook
#from fastai.vision.all import *
from fastai.text.all import *
import pathlib
import numpy as np
import googletrans
from googletrans import Translator
import requests
import urllib.request
from request_from_drive import *
from custom_streamlit import custom
#notas (necesario instalar estas versiones)
#pip install spacy==2.2.4
#pip install googletrans==4.0.0-rc1

def sentiment_classifier():

    st.write('''
    ## Clasificador de Sentimientos

    Tanto al inicio como al final del curso, se nos presenta este modelo de
    clasificación, cuya principal característica es que está construido a partir
    de un modelo pre-entrenado del lenguaje (NLP). Así, este modelo es capaz de
    distinguir si un comentario es de carácter positivo o negativo.

    **Notas: **
    - El modelo fue entrenado en **ingles**, con lo cual, si bien en la DEMO le integre
    un traductor, este no queda excepto de errores de traducción.
    - Al probar este clasificador, también pude notar que, al ser este construido
    a base de **comentarios de películas**, le es difícil acertar sobre el sentimiento
    del comentario en otro contexto.


    ### Datos

    La base de datos utilizada corresponde a comentarios de películas extraídas
    del siguiente enlace: https://ai.stanford.edu/~amaas/data/sentiment/

    ### Modelo

    El modelo utilizado corresponde a una red neuronal recurrente con una LSTM
    (Long Short Term Memory) integrada. Asimismo, la estructura de la red neuronal,
    es la que viene por defecto de la clase AWD_LSTM de Fastai.

    ### DEMO
    ''')

    # directorio de la carpeta
    path = Path.cwd()
    file_id = '1YRnsXQwl2scl6H2f4xJlnBogckBGZwhe'
    destination = 'm_sent_class.pkl'
    download_file_from_google_drive(file_id, destination)

    # correcciones de ruta
    #pathlib.PosixPath = pathlib.WindowsPath
    plt = platform.system()
    if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath
    temp = pathlib.PosixPath


    # cargando el modelo
    learn = load_learner(str(path) + "/m_sent_class.pkl")

    #st.write(str(learn.model))

    # Llamar al traductor
    traductor = Translator()

    # Haciendo la prediccion
    col1, col2, col3 = st.columns(3)
    col2.write('**Escribe un comentario**')
    resena = st.text_input('')
    if resena:
        resena_eng = traductor.translate(resena).text
        st.write('**Traducción**')
        st.write(str(resena_eng))
        prediccion = learn.predict(resena_eng)
        prob = int(np.round(prediccion[2][1]*100, 0))
        #st.write(str(prediccion))
        st.write('**Prediccion**')
        if prediccion[0] == 'pos':
            st.write(f'Se predice que el comentario es **positivo** con una probabilidad del {prob}%')
        else:
            st.write(f'Se predice que el comentario es **negativo** con una probabilidad del {100-prob}%')

#sentiment_classifier()
