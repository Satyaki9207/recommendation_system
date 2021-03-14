import pandas as pd 
import streamlit as st
from reco_helper import reco

movies=pd.read_table('ml-1m/movies.dat',sep='::',header=None,names=['movie_id','title','genre'],engine='python')
st.title('Movie Recommendation Engine')
option=st.sidebar.selectbox('Choose your favorite movie',movies['title'].unique())
st.write('Since you like ',option)
st.write('You may also like \n')
st.write(reco(option))