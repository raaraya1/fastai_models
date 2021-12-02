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
from fastai.collab import *
from fastai.tabular.all import *
import matplotlib
from matplotlib.pyplot import subplots, figure
import plotly.express as px
#https://drive.google.com/file/d/1PvtDw7m9EdzPQXcGWgGOEA1wi-wqAAVI/view?usp=sharing


class recomendation_system_st():
    def __init__(self):
        pass

    def model(self):

        st.write('''
        ## Sistema de recomendaciones
        Este modelo tiene por objetivo predecir la calificación que pondría un
        usuario sobre algún ítem. De esta manera, para trabajar con este modelo
        se requiere que los datos contengan:
        - 1. **Los Ítems** (Ej: Películas)
        - 2. **Los Usuarios**
        - 3. **Las Calificaciones** (Ej: Puntuaciones de 0 a 5)

        Para el caso analizado en el curso se plantea elaborar un modelo capaz de
        'rellenar' las calificaciones faltantes de los usuarios que aún no han visto
        alguna película. Así, se espera que el modelo sea capaz de
        ordenar esta información desorganizada bajo una cierta cantidad de categorías
        o factores. En el curso, a estas categorías le son llamadas **factores latentes**,
        así al conjunto de todos estos factores latentes (Películas + Usuarios) forman
        la **Matriz de Embeddings.**

        ''')
        path_movies = untar_data(URLs.ML_100k)
        movies = pd.read_csv(path_movies/'u.item',  delimiter='|', encoding='latin-1',
            usecols=(0,1), names=('movie','title'), header=None)
        ratings = pd.read_csv(path_movies/'u.data', delimiter='\t', header=None,
                  names=['user','movie','rating','timestamp'])
        ratings = ratings.merge(movies)
        #custom()
        # para cargar el modelo
        plt = platform.system()
        if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath
        temp = pathlib.PosixPath

        path = Path.cwd()
        file_id = '1PvtDw7m9EdzPQXcGWgGOEA1wi-wqAAVI'
        destination = 'ranking_collab.plk'
        download_file_from_google_drive(file_id, destination)
        #st.write(str(path))
        #st.write(str(os.listdir()))

        path = Path(str(path) + '/ranking_collab.plk')
        learn = load_learner(path)

        # datos
        st.write('''
        ### Datos
        Los datos fueron extraídos de los datasets que tiene almacenos fastai.
        Este se accede desde **URLs.ML_100k**

        Aquí se presenta una muestra de cómo se ordenan los datos.


        ''')
        df_ratings = pd.DataFrame(ratings)
        st.dataframe(df_ratings)


        st.write('''
        ### Modelo

        En esta ocasión, el modelo utilizado no es el de una red neuronal, sino uno
        que trabaja sobre el cruce entre los fatores latentes de los usuarios con el de
        las películas para así generar las puntuaciones. Este modelo se
        introduce en el curso con la clase **DotProductBias**.

        **Nota: **
        El modelo generado trabaja sobre 50 factores latentes tanto para los usuarios
        como para las películas. (también podemos usar la función **get_emb_sz** para
        calcular el número de factores a utilizar)

        ### DEMO

        Quizás, más interesante que observar las predicciones sobre las puntuaciones
        de los usuarios, puede ser observar cómo los títulos de las películas son ordenados en función de los factores latentes.

        El siguiente grafico muestra, de manera tridimensional, como estos títulos
        son agrupados. Así, aquellos títulos que se encuentren más cercanos corresponden
        a aquellas películas que comparten más factores latentes en común.

        ''')

        num_movies = 1000
        movies_show = int(st.sidebar.number_input('Cantidad de películas para mostrar', value=20, min_value=1, max_value=1000))
        g = ratings.groupby('title')['rating'].count() #agrupar por rating
        top_movies = g.sort_values(ascending=False).index.values[:num_movies] # ordenar por rating descendente
        top_idxs = ([learn.dls.classes['title'].o2i[m] for m in top_movies]) # buscar el indice de las peliculas anteriores
        movie_w = learn.model.i_weight.weight[top_idxs].cpu().detach() # buscamos los factores latentes en las peliculas
        movie_pca = movie_w.pca(3) # a cada factor lo dividimos en 3 dim
        fac0,fac1,fac2 = movie_pca.t()
        idxs = list(range(movies_show))
        X = fac0[idxs]
        Z = fac1[idxs]
        Y = fac2[idxs]
        movies = top_movies[idxs]

        lista = np.transpose([[i.item() for i in X], [i.item() for i in Y], [i.item() for i in Z]])
        df = pd.DataFrame(lista)
        df.columns = ['x', 'y', 'z']
        df['movie'] = movies

        fig = px.scatter_3d(df, x='x', y='y', z='z', color='movie')
        st.plotly_chart(fig, use_container_width=True)

        # Recomendacion en base a una pelicula
        st.write('''

        ### Recomendaciones
        De esta manera, lo siguiente que podríamos hacer es generar la recomendación de una película
        en base a un título que demos como input.

        ''')

        cant_recom = int(st.sidebar.number_input('Cantidad de películas para recomendar',
                                                value=1,
                                                min_value=1,
                                                max_value=movies_show-2))
        movie_input = st.selectbox('', options=movies)
        movie_factors = learn.model.i_weight.weight
        idx = learn.dls.classes['title'].o2i[movie_input]
        distances = nn.CosineSimilarity(dim=1)(movie_factors, movie_factors[idx][None])
        idx = distances.argsort(descending=True)[1:cant_recom+1]
        movie_recomend = learn.dls.classes['title'][idx]
        movie_recomend = pd.DataFrame([i for i in movie_recomend], columns=['Películas Recomendadas'])

        st.write('**Se recomiendan ver las siguientes películas:**')
        col1, col2, col3 = st.columns(3)
        col2.table(movie_recomend)
