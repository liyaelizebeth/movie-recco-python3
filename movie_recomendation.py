import numpy as np
import pandas as pd
#read CSV FILE
df = pd.read_csv("movies.csv")
print (df.head())

print(df.columns)
#select features
features = ['keywords','cast','genres','director']

for feature in features:
    df[feature] = df[feature].fillna('')
    
#create a column in DF which combines all the selected features
def combine_features(row):
   try:  
    return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']
   except:
        print ("Error:",row)

df["combined_features"] = df.apply(combine_features,axis=1)
print("combined features:",df["combined_features"].head())

#create count matrix for new combined column
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

#compute cosine similarity based on count_matrix
from sklearn.metrics.pairwise import cosine_similarity
similarity_movies = cosine_similarity(count_matrix)
movie_user_likes = "Avatar"

#get index of movie from title
def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]

movie_index = get_index_from_title(movie_user_likes)

similar_movies = list(enumerate(similarity_movies[movie_index]))

#get list of similar movies in descending order
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)

        break

    