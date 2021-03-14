import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

## Read files
def read_files():
    movies=pd.read_table('ml-1m/movies.dat',sep='::',header=None,names=['movie_id','title','genre'],engine='python')
    users=pd.read_table('ml-1m/users.dat',sep='::',header=None,names=['user_id','gender','age','occupation','zipcode'],engine='python')
    ratings=pd.read_table('ml-1m/ratings.dat',sep='::',header=None,names=['user_id','movie_id','rating','timestamp'],engine='python')
    return movies,users,ratings

# create user-item matrix
def create_X(df):
    '''Creates the user-item matrix from ratings dataframe
    
    Args: Pandas Dataframe
    
    Outputs: Sparse user-item matrix
    user mapper: dictionary to map user ids to user indices
    user_inv_mapper : dictionary to map user indices to user ids
    
    movie_mapper: dictionary to map movie_id to movie indices
    movie_inverse_mapper: dictionary to map movie indices to movie ids 
    '''
    N=df['user_id'].nunique()
    M=df['movie_id'].nunique()
    
    user_mapper=dict(zip(np.unique(df['user_id']),list(range(N))))
    movie_mapper=dict(zip(np.unique(df['movie_id']),list(range(M))))
    
    user_inv_mapper=dict(zip(list(range(N)),np.unique(df['user_id'])))
    movie_inv_mapper=dict(zip(list(range(M)),np.unique(df['movie_id'])))
    
    user_index=[user_mapper[i] for i in df['user_id']]
    movie_index=[movie_mapper[i] for i in df['movie_id']]
    
    X=csr_matrix((df['rating'],(movie_index,user_index)),shape=(M,N))
    
    return X,user_mapper,movie_mapper,user_inv_mapper,movie_inv_mapper  


# Finding similar movies
def find_similar_movies(movie_id,movie_map,movie_inv_map,X,k=10,metric='cosine',show_distance=False):
    '''Finds K nearest neighbors for a given movie id
    
    Args
    movie_id : id of the movie of interest
    X : User-item matrix
    k : number of similar movies to find
    metric : metric to measure similarity
    
    Output
    Returns a list of k similar movie ids
    '''
    neighbor_ids=[]
    movie_ind=movie_map[movie_id]
    movie_vec=X[movie_ind]
    
    KNN=NearestNeighbors(n_neighbors=k+1,algorithm='brute',metric=metric)
    KNN.fit(X)
    
    if isinstance(movie_vec,(np.ndarray)):
        movie_vec=movie_vec.reshape(-1,1)
    neighbor=KNN.kneighbors(movie_vec,return_distance=show_distance)
    for i in range(0,k+1):
        n=neighbor.item(i)
        neighbor_ids.append(movie_inv_map[n])
    neighbor_ids.pop(0)
    return neighbor_ids  


# Function to suggest movie recommendations
def print_recommendations(movie_id=1):
    '''Expects a movie title to make recommendations'''
    movies,users,ratings=read_files()
    X,user_map,movie_map,user_inv_map,movie_inv_map=create_X(ratings)
    sparsity=100*X.count_nonzero()/(X.shape[0]*X.shape[1])
    movie_titles=dict(zip(movies['movie_id'],movies['title']))

    similar_ids=find_similar_movies(movie_id,movie_map,movie_inv_map,X,metric='cosine')
    current_title=movie_titles[movie_id]

    print(f"Because you watched {current_title}")
    print("You may like")
    print("\n")
    return [movie_titles[i] for i in similar_ids]
    

