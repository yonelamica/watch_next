import numpy as np 
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from cosmos import cosmos_sim
from sklearn.feature_extraction.text import CountVectorizer

import spacy
# Load the English model
nlp = spacy.load('en_core_web_sm')

movies = pd.read_csv('movies.txt', sep='|', header=None)
movies.columns = ['title', 'description']

def get_similarity(description):
    # Tokenize and preprocess the description using spaCy
    description_doc = nlp(description)
    tokenized_description = ' '.join([token.lemma_ for token in description_doc])
    
    # Vectorize the description
    vectorizer = CountVectorizer()
    description_matrix = vectorizer.fit_transform([tokenized_description] + list(movies['description']))
    
    # Calculate cosine similarity 
    cosine_sim = cosine_similarity(description_matrix, description_matrix)
    
    # Get the index of the input movie
    movie_index = list(movies[movies['description'] == description].index)[0]
    
    # Get the index of subsequent movies
    subsequent_movie_indexes = cosmos_sim[movie_index].argsort().flatten()[::-1]
    subsequent_movies = movies.iloc[subsequent_movie_indexes].title
    
    # Return the title of the most similar movie
    return subsequent_movies.iloc[1]

description = 'Will he save their world or destroy it? When the Hulk becomes too dangerous for the Earth, the Illuminati trick Hulk into a shuttle and launch him into space to a planet where the Hulk can live in peace. Unfortunately, Hulk land on the planet Sakaar where he is sold into slavery and trained as a gladiator.' 
most_similar = get_similarity(description)
print(most_similar)


