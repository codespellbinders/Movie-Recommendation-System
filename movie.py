import pandas as pd
import numpy as np

# Load data from the csv files
movies_df = pd.read_csv("movies.csv")
ratings_df = pd.read_csv("ratings.csv")

# Merge the two dataframes on movieId
merged_df = pd.merge(movies_df, ratings_df, on='movieId')

# Create a pivot table for better readability and easier analysis
pivot_table = pd.pivot_table(merged_df, index='userId', columns='title', values='rating')

# Calculate the mean of each movie's ratings
mean = pivot_table.mean(axis=0)

# Normalize the data by subtracting the mean of each movie's ratings from its ratings
movie_features = pivot_table.sub(mean, axis=1)

# Fill the missing values with 0
movie_features.fillna(0, inplace=True)

# Calculate the similarity matrix using cosine similarity
dot_product = np.dot(movie_features.T, movie_features)
norms = np.sqrt(np.outer(np.diag(dot_product), np.diag(dot_product)))
similarity_matrix = dot_product / (norms * norms.T + 1e-10)

# Convert the similarity matrix to a dataframe for better readability and easier analysis
similarity_df = pd.DataFrame(similarity_matrix, index=movie_features.columns, columns=movie_features.columns)


# Define a function to recommend similar movies
def get_similar_movies(movie_name, num_similar=1000):
    similar_scores = similarity_df[movie_name].sort_values(ascending=False)
    similar_movies = similar_scores.index[1:num_similar + 1]
    return similar_movies


# Test the function by recommending movies similar to "Toy Story (1995)"
print(get_similar_movies("Toy Story (1995)", 1000))
