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
#https://drive.google.com/file/d/19-NkPM8S39UGs6g3_k1uxCKals-8gDAN/view?usp=sharing

class tabular_adult_st():
    def __init__(self):
        pass

    def model(self):


        st.write('''
        ## Predicción de Ingresos

        Este es de los primeros modelos tabulares introducidos en el curso. Así,
        el objetivo con este es, para un conjunto de datos en específicos, predecir
        si el ingreso de una persona se encontrara **por sobre o bajo los 50k.**

        **Nota: **
        Antes de empezar la construcción de este tipo de modelos, es
        necesario trabajar los datos:
        - **Limpiar Datos (valores vacíos, erróneos, outliers)**
        - **Distinguir entre variables categóricas y variables continuas**
        - **Transformación de datos (escalar, por ejemplo) **
        - **Descartar variables que puedan ser redundantes (ojo con la correlación)**


        ### Datos

        Los datos son extraídos del siguiente enlace https://archive.ics.uci.edu/ml/datasets/Adult

        **Todas las variables: **
        - age
        - workclass
        - **fnlwgt** (cantidad de personas que comparten todas estas variables)
        - education-num
        - marital-status
        - occupation
        - relationship
        - race
        - sex
        - capital-gain
        - capital-loss
        - hours-per-week
        - native-country

        ### Modelo

        El modelo utilizado es el que viene por defecto (**model_dir='models'**).
        En particular este tiene la siguiente secuencia:
        - 1) Entrada
          - variables categorías -> Embedding + Dropout
          - variables continuas ->  BatchNorm1d
        - 2) Capa1 (200 neuronas + ReLU)
        - 3) Capa2 (100 neuronas + ReLU)
        - 4) Salida (2)

        ### DEMO
        ''')

        # para cargar el modelo
        plt = platform.system()
        if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath
        temp = pathlib.PosixPath

        path = Path.cwd()
        file_id = '19-NkPM8S39UGs6g3_k1uxCKals-8gDAN'
        destination = 'm_tabular_adult.plk'
        download_file_from_google_drive(file_id, destination)
        #st.write(str(path))
        #st.write(str(os.listdir()))

        path = Path(str(path) + '/m_tabular_adult.plk')

        # DEMO
        col1, col2, col3 = st.columns(3)
        col2.write('**Coloquemos los datos de la persona**')
        st.write('**variables categóricas**')
        col1, col2, col3 = st.columns(3)
        workclass = str(col1.selectbox('workclass', options=['',
                                                 'Federal-gov',
                                                 'Local-gov',
                                                 'Never-worked',
                                                 'Private',
                                                 'Self-emp-inc',
                                                 'Self-emp-not-inc',
                                                 'State-gov',
                                                 'Without-pay']))
        education = str(col2.selectbox('education', options=['10th',
                                                         '11th',
                                                         '12th',
                                                         '1st-4th',
                                                         '5th-6th',
                                                         '7th-8th',
                                                         '9th',
                                                         'Assoc-acdm',
                                                         'Assoc-voc',
                                                         'Bachelors',
                                                         'Doctorate',
                                                         'HS-grad',
                                                         'Masters',
                                                         'Preschool',
                                                         'Prof-school',
                                                         'Some-college']))
        marital_status = str(col3.selectbox('marital status', options=['Divorced',
                                                                     'Married-AF-spouse',
                                                                     'Married-civ-spouse',
                                                                     'Married-spouse-absent',
                                                                     'Never-married',
                                                                     'Separated',
                                                                     'Widowed']))
        occupation = str(col1.selectbox('occupation', options=['',
                                                             'Adm-clerical',
                                                             'Armed-Forces',
                                                             'Craft-repair',
                                                             'Exec-managerial',
                                                             'Farming-fishing',
                                                             'Handlers-cleaners',
                                                             'Machine-op-inspct',
                                                             'Other-service',
                                                             'Priv-house-serv',
                                                             'Prof-specialty',
                                                             'Protective-serv',
                                                             'Sales',
                                                             'Tech-support',
                                                             'Transport-moving',]))
        relationship = str(col2.selectbox('relationship', options=['Husband',
                                                                 'Not-in-family',
                                                                 'Other-relative',
                                                                 'Own-child',
                                                                 'Unmarried',
                                                                 'Wife']))
        race = str(col3.selectbox('race', options=['Amer-Indian-Eskimo',
                                                'Asian-Pac-Islander',
                                                'Black',
                                                'Other',
                                                'White']))

        st.write('**variables continuas**')
        age = int(st.slider('age', min_value=17, value=38, max_value=90))
        fnlwgt = int(st.slider('fnlwgt', min_value=12285, value=101320, max_value=1484705))
        education_num = float(st.slider('education num', min_value=1, value=10, max_value=16))

        test_data = {
            'age': [age],
            'workclass': [workclass],
            'fnlwgt': [fnlwgt],
            'education': [education],
            'education-num': [education_num],
            'marital-status': [marital_status],
            'occupation': [occupation],
            'relationship': [relationship],
            'race': [race],
        }

        # Haciendo la prediccion
        st.write('**Predicción**')
        learn = load_learner(path)
        #st.write(learn.model)
        input1 = pd.DataFrame(test_data)
        input1 = input1.iloc[0]
        prediccion = learn.predict(input1)
        #st.write(str(prediccion))
        if prediccion[1].item() == 0:
            prob = int(np.round(prediccion[2][0].item()*100, 0))
            #st.write(str(prob))
            st.write(f'Se predice que la persona tiene un ingreso **< 50k** con **{prob}%** de probabilidad')
        elif prediccion[1].item() == 1:
            prob = int(np.round(prediccion[2][1].item()*100, 0))

            st.write(f'Se predice que la persona tiene un ingreso **>= 50k** con **{prob}%** de probabilidad')
