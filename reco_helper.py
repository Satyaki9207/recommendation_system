from reco_functions import print_recommendations
from fuzzywuzzy import process
import pandas as pd

# user friendly movie finder function that finds the closest title to what the user has typed
def movie_finder(title):
    '''Input: Expects the title of a movie
       
       Output: Returns the closest matching movie title from the list of movies in the database
    '''
    df=pd.read_table('ml-1m/movies.dat',sep='::',header=None,names=['movie_id','title','genre'],engine='python')
    all_titles=df['title'].tolist()
    closest_match=process.extractOne(title,all_titles)[0]
    movie_index_map1=dict(zip(df['title'],df['movie_id']))
    movie_index=movie_index_map1[closest_match]
    return closest_match,movie_index

def reco(title):
    '''Input: Expects the name of a movie
    
       Output: top 10 recommendations based on the given title
    '''
    id1=movie_finder(title)[1]
    return print_recommendations(id1)