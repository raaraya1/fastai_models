import streamlit as st
from cat_vs_dog.cat_vs_dog import *
from sentiment_classifier.sent_class import *
import re
import os
from custom_streamlit import custom


def fastai_models():
    # Personalizar pagina
    custom()

    st.title('Modelos del curso de FastAI')

    st.write('''
    ## **Contexto**
    ### En construccion...
    ''')

    st.sidebar.write('**Modelos**')
    model_name = st.sidebar.selectbox('Seleccionar Modelo',
                                     ['cat_vs_dog',
                                     'sentiment_classifier'])

    if model_name == 'cat_vs_dog':
        cat_vs_dog_st().model()

    elif model_name == 'sentiment_classifier':
        sentiment_classifier()

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
