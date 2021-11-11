import streamlit as st
from cat_vs_dog.cat_vs_dog import *
from sentiment_classifier.sent_class import *
from class_pet.class_pet import *
from bears_class.bear_class import *
from multi_class.multi_class import *
from number_class.number_class import *
from recomendation_system.recomendation_system import *
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
                                     'class_pet',
                                     'multi_class',
                                     'number_class',
                                     'recomendation_system'])

    if model_name == 'cat_vs_dog':
        cat_vs_dog_st().model()

    elif model_name == 'sentiment_classifier':
        sentiment_classifier()

    elif model_name == 'class_pet':
        class_pet_st().model()

    elif model_name == 'bears_class':
        bear_class_st().model()

    elif model_name == 'multi_class':
        multi_class_st().model()

    elif model_name == 'number_class':
        number_class_st().model()

    elif model_name == 'recomendation_system':
        recomendation_system_st().model()


# agregar google analytics
anlytcs_code = """<script async src="https://www.googletagmanager.com/gtag/js?id=UA-210353274-2"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'UA-210353274-2');
</script>"""


# Fetch the path of the index.html file
path_ind = os.path.dirname(st.__file__)+'/static/index.html'

# Open the file
with open(path_ind, 'r') as index_file:
    data=index_file.read()

    # Check whether there is GA script
    if len(re.findall('UA-', data))==0:

        # Insert Script for Google Analytics
        with open(path_ind, 'w') as index_file_f:

            # The Google Analytics script should be pasted in the header of the HTML file
            newdata=re.sub('<head>','<head>'+anlytcs_code,data)

            index_file_f.write(newdata)


if __name__ == '__main__':
    fastai_models()
