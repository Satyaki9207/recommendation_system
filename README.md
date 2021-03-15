### Recommendation system based on collaborative filtering

The app can be found [here](https://movie-recommender1.herokuapp.com/)  

The dataset used for building the system can be found [here](https://grouplens.org/datasets/movielens/1m/)
  
### Project Background
Recommendation engines are one of the most popular applications of unsupervised machine learning models. Here I have implemented a model based on user-item collaborative filtering. Collaborative filtering works on the principle of homophilly. That is similar users like similar items and the system can predict movies to an individual based on patterns observed from a larger dataset

### Dependencies
+ python 3.6 or higher
+ numpy
+ pandas
+ sklearn
+ fuzzywuzzy

### Files in the repository
+ ml-1m : folder containg 3 data files for movies, ratings and users
+ my_recommender.ipynb : jupyter notebook containing the code development workflow and EDA
+ reco_functions.py : file containing modularized code for generating the user item matrices and generating recommendations 
+ reco_helper.py : file containing helper functions for identifying movie names and generating recommendations
+ app.py : file containg the code for the deployed web application

### Results summary
Here I have used K nearest neighbors algorithm and cosine similarity as the metric to determine the most similar movies given a user input. Other distance metrics like euclidean distance,manhattan distance can also be used with the KNN algorithm. The function identifies rows in the user item matrix that are closest to the given user input to make recommendations
