import streamlit as st
from cat_vs_dog.cat_vs_dog import *
from sentiment_classifier.sent_class import *
from breed_pet.breed_pet import *
from bears_class.bear_class import *
from multi_class.multi_class import *
from number_class.number_class import *
from recomendation_system.recomendation_system import *
from tabular_adult.tabular_adult import *
from tabular_bluebook.tabular_bluebook import *
import re
import os
from custom_streamlit import custom
import matplotlib.pyplot as plt

def fastai_models():
    # Personalizar pagina
    #custom()

    st.title('Modelos del curso de FastAI')

    st.write('''
    ## **Contexto**
    Hola y bienvenidos a esta aplicación web que sirve como una **DEMO** sobre
    algunos de los modelos presentados en el curso de Fastai. Así, mi intención
    con esta es mostrarte, de una manera entretenida, lo que podrías llegar a hacer,
    luego de haber completado el primer curso con esta biblioteca.

    En consecuencia, a la izquierda te dejo algunos de los modelos trabajados
    en el curso y aquí abajo algunos enlaces que te pueden ser de interés.

    ### **Enlaces Utiles**

    - Enlaces al **curso de Fastai**: https://course.fast.ai/
    - Enlace a las **notas** del curso que fui tomando (adaptadas para Google Collab): https://github.com/raaraya1/Personal-Proyects/tree/main/Cursos/Fastai
    - Enlace para la construcción de esta **DEMO**: https://github.com/raaraya1/fastai_models
    #
    #
    ''')

    st.sidebar.write('**Modelos**')
    model_name = st.sidebar.selectbox('Seleccionar Modelo',
                                     ['cat_vs_dog',
                                     'sentiment_classifier',
                                     'bears_class',
                                     'breed_pet',
                                     'multi_class',
                                     'number_class',
                                     'recomendation_system',
                                     'tabular_adult',
                                     'tabular_bluebook'])

    if model_name == 'cat_vs_dog':
        cat_vs_dog_st().model()

    elif model_name == 'sentiment_classifier':
        sentiment_classifier()

    elif model_name == 'breed_pet':
        breed_pet_st().model()

    elif model_name == 'bears_class':
        bear_class_st().model()

    elif model_name == 'multi_class':
        multi_class_st().model()

    elif model_name == 'number_class':
        number_class_st().model()

    elif model_name == 'recomendation_system':
        recomendation_system_st().model()

    elif model_name == 'tabular_adult':
        tabular_adult_st().model()

    elif model_name == 'tabular_bluebook':
        tabular_bluebook_st().model()


if __name__ == '__main__':
    fastai_models()
