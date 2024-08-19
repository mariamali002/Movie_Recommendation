import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
# Load your movie data and model here
movies = pickle.load(open('movies.pkl','rb'))  # Example CSV file with movie data
# Assuming you have a cosine similarity matrix
cosine_sim = pickle.load(open('similarity.pkl','rb'))  # Example: Precomputed cosine similarity matrix

def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = movies.index[movies['title'] == title].tolist()[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return movies['title'].iloc[movie_indices]

st.title('Movie Recommendation System')
st.write('Enter a movie title to get recommendations:')

title = st.text_input('Movie Title')

if title:
    recommendations = get_recommendations(title)
    st.write('Recommendations:')
    for movie in recommendations:
        st.write(movie)
