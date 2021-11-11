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
from fastbook import *
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
        # datos
        ''')
        df_ratings = pd.DataFrame(ratings)
        st.dataframe(df_ratings)


        st.write('''
        # Resultado
        ''')

        num_movies = 1000
        movies_show = 20
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
