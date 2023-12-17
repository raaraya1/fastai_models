import streamlit as st
import fastbook
from fastbook import *
from fastai.tabular.all import *
import pathlib
from PIL import Image
import numpy as np
import requests
import urllib.request
from request_from_drive import *
import os
import pickle
#from custom_streamlit import custom
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from fastai.tabular.all import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_graphviz
from dtreeviz.trees import *
from IPython.display import Image, display_svg, SVG
import streamlit.components.v1 as components
import base64
import matplotlib.pyplot as plt
import fastai
from functools import partial
import pickle as pc
pickle.load = pc.load
pickle.Unpickler = pc.Unpickler
#https://drive.google.com/file/d/1h2VoVGOE2GrpMdj0c_J26xfFLLFBKWpP/view?usp=sharing


# ----------------- Funciones modificadas -------------------------------
# Esta funcion es de fastai (solo modifique el final para obtener la figura)
def cluster_columns_fig(df, figsize=(10,6), font_size=12):
    corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
    corr_condensed = hc.distance.squareform(1-corr)
    z = hc.linkage(corr_condensed, method='average')
    fig1 = plt.figure(figsize=figsize)
    hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=font_size)
    return st.pyplot(fig=fig1)

# ------para cargar el modelo (haciendo la conversion GPU a CPU)-------
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def load_pickle_cpu(fn):
    "Load a pickle file from a file name or opened file"
    with open_file(fn, 'rb') as f: return CPU_Unpickler(f).load()
# --------------------------------------------------------------------


class tabular_bluebook_st():
    def __init__(self):
        pass

    def model(self):


        st.write('''
        ## Elaboración de un modelo tabular

        Anteriormente, en el curso ya habíamos elaborado un pequeño modelo tabular (tabular_adult),
        aprendiendo así a cómo tratar variables categóricas y continuas, transformación
        de datos y la estructura de red neuronal que proporciona fastai para abordar estos
        problemas. Sin embargo, con el modelo anterior, aun nos faltaba dar respuesta a
        **¿Como seleccionamos las variables más relevantes para armar el modelo de red neuronal? **

        En particular, la manera en que se aborda esta pregunta en la clase es digna de imitar,
        puesto que se apoya en la confección de otros modelos (decision tree y random forest) para
        observar la interacción entre variables independientes y las mismas con la variable dependiente (precio de venta)

        Así, en esta no solo te presento el resultado de la clase, sino que también
        te presento el análisis y desarrollo realizado.

        **Nota: **
        Una de las cosas que más rescato de esta clase es que también sirve de introducción
        a la biblioteca **sklearn**, dado que los modelos de **decision tree** y **random forest**
        son los pertenecientes a esta biblioteca.


        ### Datos

        En esta oportunidad los datos fueron extraídos desde Kaggle a través del
        siguiente enlace https://www.kaggle.com/c/bluebook-for-bulldozers/data

        Asimismo, en el curso se nos comentan una serie de pasos previos que son
        necesarios de completar antes de poder tener acceso a estos datos.

        ### Modelo

        En esta sección pretendo cubrir, no solo el modelo
        de **red neuronal**, sino que también los de **decision tree** y el de **random forest**, al menos en un nivel básico.


        ### DEMO

        ''')


        # para cargar los datos
        plt = platform.system()
        if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath
        temp = pathlib.PosixPath

        path = Path.cwd()
        file_id = '1h2VoVGOE2GrpMdj0c_J26xfFLLFBKWpP'
        destination = 'to.pkl'
        download_file_from_google_drive(file_id, destination)
        #st.write(str(path))
        #st.write(str(os.listdir()))

        path = Path(str(path) + '/to.pkl')
        archivo = open(path, 'rb')
        to = pickle.load(archivo)

        xs,y = to.train.xs.copy(),to.train.y.copy()
        valid_xs,valid_y = to.valid.xs.copy(),to.valid.y.copy()

        # Funcioes para estimar el rendimiento de los Modelos
        def r_mse(pred,y):
          return round(math.sqrt(((pred-y)**2).mean()), 6)

        def m_rmse(m, xs, y):
          return r_mse(m.predict(xs), y)

        # Error en los datos (años muy antiguos)
        xs.loc[xs['YearMade']<1900, 'YearMade'] = 1950
        valid_xs.loc[valid_xs['YearMade']<1900, 'YearMade'] = 1950

        # Opciones
        choice = st.sidebar.radio('Modelos', options=['decision tree',
                                            'random forest',
                                            'neuronal network'])

        if choice == 'decision tree':
            # Arbol de decision

            st.write('''
            ### Decision Tree

            Un árbol de decisión es un algoritmo que sirve para clasificar y segmentar,
            utilizando alguna métrica (MSE, para este caso) sobre el set de datos de entrenamiento

            En resumidas palabras para construir este algoritmo debemos:
            - Primero seleccionar una columna (se comienza con la primera columna)
            - Luego selecciono un valor de esa columna (se comienza con el más bajo)
            - Luego separo los datos en dos grupos (menores y mayores a tal valor)
            - Genero una predicción para ambos grupos (predicción = media (variable dependiente del grupo))
            - Recorro todos los posibles valores de la columna (segmentando en dos grupos) y voy guardando el valor que genera la menor diferencia entre los valores predichos y la variable dependiente.
            - Luego hago este proceso para cada columna.
            - Rescato la columna y su valor en específico donde se obtuvo la menor diferencia entre los valores predichos y la variable dependiente.
            - Genero dos set de datos nuevos (menores y mayores al valor de la columna escogida), asegurándome de quitar la columna utilizada.
            - Vuelvo a ejecutar el mismo procedimiento para cada nuevo set de datos originados, hasta cumplirse alguna condición (número max de nodos, por ejemplo)

            Así, a continuación, podrás observar cómo se genera este árbol, utilizando
            como condición de termino un numero máximo de nodos al final.

            ''')
            nodos_final = st.sidebar.slider('Cantidad de nodos al final del árbol', min_value=2, max_value=10, value=4)
            m = DecisionTreeRegressor(max_leaf_nodes=nodos_final)
            m.fit(xs, y)

            # mostrar resultados
            tree_graph = export_graphviz(m, out_file=None, feature_names=xs.columns, filled=True,
                special_characters=True, rotate=True, precision=2)

            st.graphviz_chart(tree_graph)

            col1, col2, col3 = st.columns(3)
            error_dt = m_rmse(m, valid_xs, valid_y)
            col2.metric(label='Validation Error', value=error_dt)

        elif choice == 'random forest':

            st.write('''
            ### Random Forest

            Sin entrar en mayor detalle sobre el funcionamiento de este algoritmo,
            este lo podríamos llegar a describir como un conjunto de árboles de decisión,
            todos estos elaborados de una manera distinta. Así, al momento de generar una
            predicción sobre un set de datos, es que cada árbol emite un juicio y de esta
            manera el promedio de las decisiones de los árboles pasa a ser la predicción
            del algoritmo de **Random Forest**.
            ''')

            st.sidebar.write('**Random Forest**')
            cant_tree = int(st.sidebar.number_input('Cantidad de árboles (estimadores)', value=40))
            max_sample = int(st.sidebar.number_input('Tamaño máximo de muestras', value=200_000))
            min_samples_leaf = int(st.sidebar.number_input('Cantidad mínima de datos en nodos finales', value=5))
            max_features = st.sidebar.slider('Cantidad máxima de variables (en fracción)', min_value=0.1, max_value=1.0, value=0.5)

            def rf(xs, y, n_estimators=cant_tree, max_samples=max_sample, max_features=max_features, min_samples_leaf=min_samples_leaf, **kwargs):
                return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators, max_samples=max_samples, max_features=max_features, min_samples_leaf=min_samples_leaf, oob_score=True).fit(xs, y)

            m3 = rf(xs, y)

            st.write('''
            #### Feature Importance

            Para observar la relevancia de las variables en el modelo podemos hacer uso del
            atributo `feature_importances_`. Este en particular:
            - Recorre todos los arboles
            - Luego recorre todas las separaciones (categorías)
            - Y para cada una de estas separaciones se mide cuanto mejora el modelo
            - De esta manera para cada categoría se registra la suma de estas mejoras
            - Por último, se normaliza todo, para que la suma de las mejoras en las categorías dé 1

            Así, el siguiente grafico muestra las 30 variables más relevantes en orden
            ascendente.

            **Nota: **
            Esta función nos resultará sumamente útil a la hora de construir el modelo
            de redes neuronales, dado que fijando como criterio **descartar** aquellas variables
            con un nivel de **importancia bajo** nos ayudará a construir un modelo más sencillo,
            asegurando a su vez que este tenga un buen desempeño. De esta manera, y aplicando como
            criterio descartar aquellas variables con un nivel menor a 0.005 se pasó de trabajar
            con un modelo de **66 variables** a trabajar con uno de **21 variables**.

            ''')

            def rf_feat_importance(m, df):
                return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                                   ).sort_values('imp', ascending=False)

            fi = rf_feat_importance(m3, xs)
            def plot_fi(fi):
                return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
            fig1 = plot_fi(fi[:30]).get_figure()
            st.pyplot(fig=fig1)

            # se remueven las categorias menos relevantes
            to_keep = fi[fi.imp>0.005].cols
            xs_imp = xs[to_keep].copy()
            valid_xs_imp = valid_xs[to_keep].copy()

            # se vuelve a cargar el modelo
            #m4 = rf(xs_imp, y)

            st.write('''
            #### Removing Redundant Features

            Lo siguiente que nos puede ser útil es **remover aquellas categorías** que se
            encuentren **fuertemente correlacionadas**. Por ejemplo, para otros tipos de datos,
            podríamos encontrarnos que la edad de las personas y sus fechas de nacimiento
            se encuentren fuertemente relacionadas. Para este caso, será fácil intuir
            que descartar una de estas variables no impactará de manera importante en el
            desempeño predictivo del modelo.

            ''')

            cluster_columns_fig(xs_imp)

            st.write('''
            Así, lo que se prosigue hacer es **dejar solo una de las variables que tengan los
            'caminos' más cortos**.

            ''')

            col1, col2, col3 = st.columns(3)
            error_rf = m_rmse(m3, valid_xs, valid_y)
            col2.metric(label='Validation Error', value=error_rf)

        elif choice == 'neuronal network':

            st.write('''
            ### Tabular Learner

            Del análisis de los modelos, principalmente del **modelo de Random Forest**, es
            que se seleccionan las variables que son más relevantes para
            entrenar al modelo de red neuronal.

            De esta manera, y sin entrar en mayor detalle sobre que representa cada
            variable, es que de las **66 columnas** que se contaban inicialmente en los datos, se
            deciden utilizar solo **15 de estas**.

            Así, las categorías seleccionadas son:

            - YearMade
            - ProductSize
            - Coupler_System
            - fiProductClassDesc
            - ModelID
            - fiSecondaryDesc
            - Hydraulics_Flow
            - ProductGroup
            - fiModelDesc
            - Enclosure
            - Hydraulics
            - Drive_System
            - Tire_Size
            - Pad_Type
            - saleElapsed

            Asimismo, la estructura de la red neuronal tiene como entrada una matriz
            de embedding, para luego introducirse en una primera capa con 500 neuronas
            (ReLu), luego por una segunda capa con 250 neuronas (ReLU) y, por último
            el resultado se hace pasar por una función sigmoide con un rango de 8 a 12 para
            ajustarse a los valores por los cuales se mueve el precio de venta.

            **Nota: ** Hay que recordar que a la columna de precio de venta (variable dependiente)
            se le aplicó una **función logarítmica**. En consecuencia, si luego queremos realizar
            inferencias con el modelo entrenado es que necesitamos aplicar
            la función exponencial para obtener el valor real en la predicción del precio de venta.
            ''')



            # descargar el modelo
            # https://drive.google.com/file/d/1evFXrmOeFH572FHeI1N6pAiuIN41wcto/view?usp=sharing
            path = Path.cwd()
            file_id = '1evFXrmOeFH572FHeI1N6pAiuIN41wcto'
            destination = 'to_nn_v3.pkl'
            download_file_from_google_drive(file_id, destination)

            path = Path(str(path) + '/to_nn_v3.pkl')

            learn = load_pickle_cpu(path)
            def r_mse(pred,y):
              return round(math.sqrt(((pred-y)**2).mean()), 6)

            def m_rmse(m, xs, y):
              return r_mse(m.predict(xs), y)

            preds, targs = learn.get_preds()
            error_nn = r_mse(preds, targs)

            col1, col2, col3 = st.columns(3)
            col2.metric(label='Validation Error', value=error_nn)

            st.write('''
            ### Predicción
            ''')

            df = learn.dls.train.xs
            YearMade = set(df['YearMade'])
            ProductSize = set(df['ProductSize'])
            Coupler_System = set(df['Coupler_System'])
            fiProductClassDesc = set(df['fiProductClassDesc'])
            ModelID = set(df['ModelID'])
            fiSecondaryDesc = set(df['fiSecondaryDesc'])
            Hydraulics_Flow = set(df['Hydraulics_Flow'])
            ProductGroup = set(df['ProductGroup'])
            fiModelDesc = set(df['fiModelDesc'])
            Enclosure = set(df['Enclosure'])
            Hydraulics = set(df['Hydraulics'])
            Drive_System = set(df['Drive_System'])
            Tire_Size = set(df['Tire_Size'])
            Pad_Type = set(df['Pad_Type'])
            saleElapsed = set(df['saleElapsed'])

            col1, col2, col3, col4, col5 = st.columns(5)
            set_YearMade = col1.selectbox('YearMade', options=YearMade)
            set_ProductSize = col1.selectbox('ProductSize', options=ProductSize)
            set_Coupler_System = col1.selectbox('Coupler_System', options=Coupler_System)
            set_fiProductClassDesc = col2.selectbox('fiProductClassDesc', options=fiProductClassDesc)
            set_ModelID = col2.selectbox('ModelID', options=ModelID)
            set_fiSecondaryDesc = col2.selectbox('fiSecondaryDesc', options=fiSecondaryDesc)
            set_Hydraulics_Flow = col3.selectbox('Hydraulics_Flow', options=Hydraulics_Flow)
            set_ProductGroup = col3.selectbox('ProductGroup', options=ProductGroup)
            set_fiModelDesc = col3.selectbox('fiModelDesc', options=fiModelDesc)
            set_Enclosure = col4.selectbox('Enclosure', options=Enclosure)
            set_Hydraulics = col4.selectbox('Hydraulics', options=Hydraulics)
            set_Drive_System = col4.selectbox('Drive_System', options=Drive_System)
            set_Tire_Size = col5.selectbox('Tire_Size', options=Tire_Size)
            set_Pad_Type = col5.selectbox('Pad_Type', options=Pad_Type)
            set_saleElapsed = col5.selectbox('saleElapsed', options=saleElapsed)

            test_data = {
                'YearMade': [set_YearMade],
                'ProductSize': [set_ProductSize],
                'Coupler_System': [set_Coupler_System],
                'fiProductClassDesc': [set_fiProductClassDesc],
                'ModelID': [set_ModelID],
                'fiSecondaryDesc': [set_fiSecondaryDesc],
                'Hydraulics_Flow': [set_Hydraulics_Flow],
                'ProductGroup': [set_ProductGroup],
                'fiModelDesc': [set_fiModelDesc],
                'Enclosure':[set_Enclosure],
                'Hydraulics':[set_Hydraulics],
                'Drive_System':[set_Drive_System],
                'Tire_Size':[set_Tire_Size],
                'Pad_Type':[set_Pad_Type],
                'saleElapsed':[set_saleElapsed]
                }
            input = pd.DataFrame(test_data).iloc[0]

            prediccion = learn.predict(input)
            sale_price = int(np.round(np.e**(prediccion[1].item()), 0))
            st.write(f'Se predice un precio de venta de **{sale_price}**')
















        # DEMO
        #learn = load_learner(path)
