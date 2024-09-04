import marimo

__generated_with = "0.8.0"
app = marimo.App()


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# Part A:""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Intro""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# Item-Based Collaborative Filtering Recommendation Algorithm""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        Item-based collaborative filtering is also called item-item collaborative filtering. It is a type of recommendation system algorithm that uses item similarity to make product recommendations.

        Points to understand
        * What is item-based (item-item) collaborative filtering?
        * How to create a user-product matrix?
        * How to identify similar items?
        * How to rank items for the recommendation?

        Let's get started!
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# Step 1: Import Python Libraries""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ```python
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        from scipy.stats import pearsonr
        from sklearn.metrics.pairwise import cosine_similarity

        import marimo as mo
        ```
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# Step 2: Download And Read In Data""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""This experiment uses the movielens dataset. This dataset contains actual user ratings of movies."""
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""There are multiple datasets in the 100k movielens folder. For this experiment, we will use two ratings and movies."""
    )
    return


@app.cell
def __(pd):
    # Read in data

    # Read in the ratings data
    ratings_df = pd.read_csv("ratings.csv")

    # Read in the movies data
    movies_df = pd.read_csv("movies.csv")

    # Take a look at the data
    print("Ratings Data:")
    print(ratings_df.head())
    print("\nMovies Data:")
    print(movies_df.head())
    return movies_df, ratings_df


@app.cell
def __(ratings_df):
    ratings_df
    return


@app.cell
def __(movies_df):
    movies_df
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        There are four columns in the ratings dataset, userID, movieID, rating, and timestamp.

        The dataset has over 100k records, and there is no missing data.
        """
    )
    return


@app.cell
def __(movies_df, ratings_df):
    # Get the dataset information
    print("\nRatings Data Info:")
    print(ratings_df.info())
    print("\nMovies Data Info:")
    print(movies_df.info())
    return


@app.cell
def __(movies_df, ratings_df):
    # Get the dataset information
    print("\nRatings Data Describe:")
    print(ratings_df.describe())
    print("\nMovies Data Describe:")
    print(movies_df.describe())
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""The 100k ratings are from 610 users on 9724 movies. The rating has ten unique values from 0.5 to 5."""
    )
    return


@app.cell
def __(ratings_df):
    # Print Number of users
    num_users = ratings_df["userId"].nunique()
    print(f"\nNumber of users: {num_users}")

    # Print Number of movies
    num_movies = ratings_df["movieId"].nunique()
    print(f"Number of movies: {num_movies}")

    # Print Number of ratings
    num_ratings = len(ratings_df)
    print(f"Number of ratings: {num_ratings}")

    # Print List of unique ratings
    unique_ratings = ratings_df["rating"].unique()
    print(f"Unique ratings: {sorted(unique_ratings)}")
    return num_movies, num_ratings, num_users, unique_ratings


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""read in the movies data to get the movie names.""")
    return


@app.cell
def __(movies_df):
    # Read data above

    # Take a look at the data
    print("\nMovie Names Data:")
    print(movies_df["title"].head(20))
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""Using `movieID` as the matching key, we appended movie information to the rating dataset and named it 'df'. So now we have the movie tile and movie rating in the same dataset!"""
    )
    return


@app.cell
def __(movies_df, pd, ratings_df):
    # Merge ratings and movies datasets
    df = pd.merge(
        ratings_df, movies_df, how="left", left_on="movieId", right_on="movieId"
    )

    # Take a look at the merged data
    print("\nMerged Data:")
    print(df.head())
    return (df,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# Step 3: Exploratory Data Analysis (EDA)""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        In step 3, we need to filter the movies and keep only those with over 100 ratings for the analysis.

        First group the movies by title, count the number of ratings, and keep only the movies with greater than 100 ratings.

        The average ratings for the movies are calculated as well.

        From the `.info()` output, check how many movies left?
        """
    )
    return


@app.cell
def __(df):
    # Aggregate by movie
    agg_ratings = (
        df.groupby("title")
        .agg(mean_rating=("rating", "mean"), number_of_ratings=("rating", "count"))
        .reset_index()
    )

    # Keep the movies with over 100 ratings
    agg_ratings_GT100 = agg_ratings[agg_ratings["number_of_ratings"] > 100]
    agg_ratings_GT100.info()
    return agg_ratings, agg_ratings_GT100


@app.cell
def __(agg_ratings_GT100):
    print(f"Number of movies with more than 100 ratings: {agg_ratings_GT100.shape[0]}")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""check what the most popular movies and their ratings are.""")
    return


@app.cell
def __(agg_ratings_GT100):
    # Display the first few rows of the filtered dataset
    print("\nTop movies with more than 100 ratings:")
    print(agg_ratings_GT100.head())
    return


@app.cell
def __():
    # Check popular movies
    return


@app.cell
def __(agg_ratings_GT100):
    # Check the most popular movies
    print("\nMost popular movies:")
    print(agg_ratings_GT100.sort_values(by="number_of_ratings", ascending=False).head())
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""Use a `jointplot` to check the correlation between the average rating and the number of ratings."""
    )
    return


@app.cell
def __(agg_ratings_GT100, plt, sns):
    # Visualization: Check the correlation between average rating and number of ratings
    sns.jointplot(x="mean_rating", y="number_of_ratings", data=agg_ratings_GT100)
    plt.suptitle("Correlation between Average Rating and Number of Ratings", y=1.02)
    plt.gca()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        To keep only the 134 movies with more than 100 ratings, we need to join the movie with the user-rating level dataframe.

        `how='inner'` and `on='title'` ensure that only the movies with more than 100 ratings are included.
        """
    )
    return


@app.cell
def __(agg_ratings_GT100, df):
    # Merge data
    # Filter the original dataframe to keep only the movies with more than 100 ratings
    df_filtered = df[df["title"].isin(agg_ratings_GT100["title"])]
    return (df_filtered,)


@app.cell
def __(df_filtered):
    df_filtered.head()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""After filtering the movies with over 100 ratings, we have 597 users that rated 134 movies."""
    )
    return


@app.cell
def __(df_filtered):
    # Number of users
    num_users_filtered = df_filtered["userId"].nunique()

    # Number of movies
    num_movies_filtered = df_filtered["movieId"].nunique()

    # Number of ratings
    num_ratings_filtered = len(df_filtered)

    # List of unique ratings
    unique_ratings_filtered = df_filtered["rating"].unique()
    return (
        num_movies_filtered,
        num_ratings_filtered,
        num_users_filtered,
        unique_ratings_filtered,
    )


@app.cell
def __(
    num_movies_filtered,
    num_ratings_filtered,
    num_users_filtered,
    unique_ratings_filtered,
):
    # Print the filtered dataset summary
    print(f"\nNumber of users in the filtered dataset: {num_users_filtered}")
    print(f"Number of movies in the filtered dataset: {num_movies_filtered}")
    print(f"Number of ratings in the filtered dataset: {num_ratings_filtered}")
    print(
        f"List of unique ratings in the filtered dataset: {sorted(unique_ratings_filtered)}"
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# Step 4: Create Item-Movie Matrix""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""In step 4, we will transform the dataset into a matrix format. The rows of the matrix are users, and the columns of the matrix are movies. The value of the matrix is the user rating of the movie if there is a rating. Otherwise, it shows 'NaN'."""
    )
    return


@app.cell
def __(df_filtered):
    # Step 4: Create Item-Item Matrix
    # Create item-item matrix using the filtered dataset
    matrix = df_filtered.pivot_table(index="title", columns="userId", values="rating")

    # Display the first few rows of the matrix
    print("User-Movie Matrix:")
    print(matrix.head())
    return (matrix,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# Step 5: Data Normalization""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        Since some people tend to give a higher rating than others, we normalize the rating by extracting the average rating of each user.

        After normalization, the movies with a rating less than the user's average rating get a negative value, and the movies with a rating more than the user's average rating get a positive value.
        """
    )
    return


@app.cell
def __(matrix):
    # Step 5: Data Normalization

    # Calculate the average rating for each user
    item_avg_ratings = matrix.mean(axis=1)

    # Normalize item-item matrix
    matrix_norm = matrix.subtract(item_avg_ratings, axis="rows")

    # Display the first few rows of the normalized matrix
    print("Normalized Item-Item Matrix:")
    print(matrix_norm.head())
    return item_avg_ratings, matrix_norm


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# Step 6: Identify Similar Users""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        There are different ways to measure similarities. Pearson correlation and cosine similarity are two widely used methods.

        For item-item, we will only consider cosine similarity.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        Now let's use user ID 66 as an example to illustrate how to find similar users.

        We first need to exclude user ID 66 from the similar user list and decide the number of similar users.
        """
    )
    return


@app.cell
def __():
    # # Pick a user ID
    # picked_userid = 66

    # # Remove picked user ID from the candidate list
    # user_similarity.drop(index=picked_userid, inplace=True)

    # # # Take a look at the data
    # # print("\nUser Similarity Matrix after removing picked user ID:")
    # # print(user_similarity.head())
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        In the user similarity matrix, the values range from -1 to 1, where -1 means opposite movie preference and 1 means same movie preference.

        `n = 10` means we would like to pick the top 10 most similar users for user ID 66.

        The user-based collaborative filtering makes recommendations based on users with similar tastes, so we need to set a positive threshold. Here we set the `item_similarity_threshold` to be 0.3, meaning that a user must have a Pearson correlation coefficient of at least 0.3 to be considered as a similar user.

        After setting the number of similar users and similarity threshold, we sort the user similarity value from the highest and lowest, then printed out the most similar users' ID and the Pearson correlation value.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""Those who are interested in using cosine similarity can refer to this code. Since `cosine_similarity` does not take missing values, we need to impute the missing values with 0s before the calculation."""
    )
    return


@app.cell
def __():
    # User similarity matrix using cosine similarity
    return


@app.cell
def __(cosine_similarity, matrix, pd):
    # Fill missing values with 0 for cosine similarity
    matrix_filled = matrix.fillna(0)

    # Calculate cosine similarity
    item_similarity_cosine = pd.DataFrame(
        cosine_similarity(matrix_filled),
        index=matrix_filled.index,
        columns=matrix_filled.index,
    )

    # Display the first few rows of the cosine similarity matrix
    print("User Similarity Matrix (Cosine Similarity):")
    print(item_similarity_cosine.head())
    return item_similarity_cosine, matrix_filled


@app.cell
def __(matrix_norm, pd):
    # Pick a user ID
    picked_userid = 66

    # Pick a movie
    picked_movie = "American Pie (1999)"

    # Movies that the target user has watched
    picked_userid_watched = (
        pd.DataFrame(
            matrix_norm[picked_userid]
            .dropna(axis=0, how="all")
            .sort_values(ascending=False)
        )
        .reset_index()
        .rename(columns={1: "rating"})
    )

    picked_userid_watched.head()
    return picked_movie, picked_userid, picked_userid_watched


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""We can see that user 66's favorite movie is Four weedings and a Funeral (1994), followed by Austin Powers : The Spy Who Shagged Me (1999)"""
    )
    return


@app.cell(hide_code=True)
def __(item_similarity_cosine, pd, picked_movie, picked_userid_watched):
    # # Number of similar users
    # n = 10
    # # picked_userid = 66

    # # User similarity threashold
    # item_similarity_threshold = 0.3

    # # Get top n similar users
    # similar_users = matrix_filled[matrix_filled[picked_userid]>item_similarity_threshold][picked_userid].sort_values(ascending=False)[:n]

    # # Remove picked user ID from the candidate list for both similarity matrices

    # # user_similarity.drop(index=picked_userid, inplace=True) # did this above
    # item_similarity_cosine.drop(index=picked_userid, inplace=True)

    # # # Pearson Similarity
    # # # Get top n similar users using Pearson similarity
    # # similar_users_pearson = user_similarity[user_similarity[picked_userid] > item_similarity_threshold][picked_userid].sort_values(ascending=False)[:n]

    # # Cosine Similarity
    # # Get top n similar users using Cosine similarity
    # similar_users_cosine = item_similarity_cosine[item_similarity_cosine[picked_userid] > item_similarity_threshold][picked_userid].sort_values(ascending=False)[:n]

    # # # Compare the two sets of similar users
    # # print(f'Top {n} similar users for user {picked_userid} using Pearson similarity:')
    # # print(similar_users_pearson)

    # print(f'Top {n} similar users for user {picked_userid} using Cosine similarity:')
    # print(similar_users_cosine)

    # Similarity score of the movie American Pie with all the other movies
    picked_movie_similarity_score = (
        item_similarity_cosine[[picked_movie]]
        .reset_index()
        .rename(columns={"American Pie (1999)": "similarity_score"})
    )

    # Rank the similarities between the movies user 1 rated and American Pie.
    n = 5
    picked_userid_watched_similarity = pd.merge(
        left=picked_userid_watched,
        right=picked_movie_similarity_score,
        on="title",
        how="inner",
    ).sort_values("similarity_score", ascending=False)[:5]

    # Take a look at the User 1 watched movies with highest similarity
    picked_userid_watched_similarity
    return (
        n,
        picked_movie_similarity_score,
        picked_userid_watched_similarity,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# Step 7: Narrow Down Item Pool""")
    return


@app.cell
def __():
    # # Movies that the target user has watched
    # picked_userid_watched = matrix_norm[matrix_norm.index == picked_userid].dropna(axis=1, how='all')
    # picked_userid_watched
    return


@app.cell
def __(np, picked_movie, picked_userid, picked_userid_watched_similarity):
    # Calculate the predicted rating using weighted average of similarity scores and the ratings from user 66
    predicted_rating = round(
        np.average(
            picked_userid_watched_similarity[66],
            weights=picked_userid_watched_similarity["similarity_score"],
        ),
        6,
    )

    print(
        f"The predicted rating for {picked_movie} by user {picked_userid} is {predicted_rating}"
    )
    return (predicted_rating,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# Movie Recommendation""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        1. Create a list of movie that the target user has not watched before.
        2. Loop through the unwatched movie and create predicted scores for each movie.
        3. Rank the predicted score of unwatched movie from high to low.
        4. Select the top k movies as the recommendations for the target user.

        The Python function below implemented the four steps. With the input of `picked_userid`, `number_of_similar_items`, and `number_of_recommendations`, we can get the top movies for the user and their corresponding ratings. Note that the ratings are normalized by extracting the average rating for the movie, so we need to add the average value back to the predicted ratings if we want the predicted ratings to be on the same scale as the original ratings.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(r"""# Step 8: Recommend Items""")
    return


@app.cell(hide_code=True)
def __(item_similarity_cosine, matrix_norm, np, pd):
    # # Item-based recommendation function
    # def item_based_rec(picked_userid=66, number_of_similar_items=5, number_of_recommendations =3):
    #   import operator
    #   # Movies that the target user has not watched
    #   picked_userid_unwatched = pd.DataFrame(matrix_norm[picked_userid].isna()).reset_index()
    #   picked_userid_unwatched = picked_userid_unwatched[picked_userid_unwatched[1]==True]['title'].values.tolist()

    #   # Movies that the target user has watched
    #   picked_userid_watched = pd.DataFrame(matrix_norm[picked_userid].dropna(axis=0, how='all')\
    #                             .sort_values(ascending=False))\
    #                             .reset_index()\
    #                             .rename(columns={1:'rating'})

    #   # Dictionary to save the unwatched movie and predicted rating pair
    #   rating_prediction ={}

    #   # Loop through unwatched movies
    #   for picked_movie in picked_userid_unwatched:
    #     # Calculate the similarity score of the picked movie iwth other movies
    #     picked_movie_similarity_score = item_similarity_cosine[[picked_movie]].reset_index().rename(columns={picked_movie:'similarity_score'})
    #     # Rank the similarities between the picked user watched movie and the picked unwatched movie.
    #     picked_userid_watched_similarity = pd.merge(left=picked_userid_watched,
    #                                                 right=picked_movie_similarity_score,
    #                                                 on='title',
    #                                                 how='inner')\
    #                                         .sort_values('similarity_score', ascending=False)[:number_of_similar_items]
    #     # Calculate the predicted rating using weighted average of similarity scores and the ratings from user 1
    #     predicted_rating = round(np.average(picked_userid_watched_similarity['rating'],
    #                                         weights=picked_userid_watched_similarity['similarity_score']), 6)
    #     # Save the predicted rating in the dictionary
    #     rating_prediction[picked_movie] = predicted_rating
    #     # Return the top recommended movies
    #   return sorted(rating_prediction.items(), key=operator.itemgetter(1), reverse=True)[:number_of_recommendations]

    # # Get recommendations
    # recommended_movie = item_based_rec(picked_userid=1, number_of_similar_items=5, number_of_recommendations=3)
    # recommended_movie

    # Item-based recommendation function
    def item_based_rec(
        picked_userid=66, number_of_similar_items=5, number_of_recommendations=3
    ):
        import operator

        # Movies that the target user has not watched
        picked_userid_unwatched = pd.DataFrame(
            matrix_norm[picked_userid].isna()
        ).reset_index()
        picked_userid_unwatched = picked_userid_unwatched[
            picked_userid_unwatched[66] == True
        ]["title"].values.tolist()

        # Movies that the target user has watched
        picked_userid_watched = (
            pd.DataFrame(
                matrix_norm[picked_userid]
                .dropna(axis=0, how="all")
                .sort_values(ascending=False)
            )
            .reset_index()
            .rename(columns={66: "rating"})
        )

        # Dictionary to save the unwatched movie and predicted rating pair
        rating_prediction = {}

        # Loop through unwatched movies
        for picked_movie in picked_userid_unwatched:
            # Calculate the similarity score of the picked movie iwth other movies
            picked_movie_similarity_score = (
                item_similarity_cosine[[picked_movie]]
                .reset_index()
                .rename(columns={picked_movie: "similarity_score"})
            )
            # Rank the similarities between the picked user watched movie and the picked unwatched movie.
            picked_userid_watched_similarity = pd.merge(
                left=picked_userid_watched,
                right=picked_movie_similarity_score,
                on="title",
                how="inner",
            ).sort_values("similarity_score", ascending=False)[:number_of_similar_items]
            # Calculate the predicted rating using weighted average of similarity scores and the ratings from user 1
            predicted_rating = round(
                np.average(
                    picked_userid_watched_similarity["rating"],
                    weights=picked_userid_watched_similarity["similarity_score"],
                ),
                6,
            )
            # Save the predicted rating in the dictionary
            rating_prediction[picked_movie] = predicted_rating
            # Return the top recommended movies
        return sorted(
            rating_prediction.items(), key=operator.itemgetter(1), reverse=True
        )[:number_of_recommendations]

    # Get recommendations
    recommended_movie = item_based_rec(
        picked_userid=66, number_of_similar_items=5, number_of_recommendations=3
    )
    recommended_movie
    return item_based_rec, recommended_movie


@app.cell(hide_code=True)
def __():
    # # Step 7: Narrow Down Item Pool

    # # Movies that the target user has watched
    # # Movies that similar users watched. Remove movies that none of the similar users have watched
    # # similar_user_movies_pearson = matrix_norm.loc[matrix_norm.index.isin(similar_users_pearson.index)].dropna(axis=1, how='all')
    # similar_user_movies_cosine = matrix_norm.loc[matrix_norm.index.isin(similar_users_cosine.index)].dropna(axis=1, how='all')
    return


@app.cell(hide_code=True)
def __():
    # # Remove the watched movie from the movie list
    # # similar_user_movies_pearson.drop(picked_userid_watched.columns, axis=1, inplace=True, errors='ignore')
    # similar_user_movies_cosine.drop(picked_userid_watched.columns, axis=1, inplace=True, errors='ignore')
    return


@app.cell(hide_code=True)
def __():
    # # Display the final list of movies that similar users have watched, excluding those already watched by the picked user
    # print("\nMovies that similar users have watched using Pearson similarity:")
    # print(similar_user_movies_pearson)

    # print("\nMovies that similar users have watched using Cosine similarity:")
    # print(similar_user_movies_cosine)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## **Recommendation System Summary**

        ### **Objective**
        I developed a user-based collaborative filtering recommendation system to suggest movies for user ID `66` based on their preferences and those of similar users.

        ### **Steps I Took:**

        1. **Data Import and Preparation:**
           - I imported the datasets `movies.csv` and `ratings.csv`.
           - Merged them to create a comprehensive dataset with movie details and user ratings.

        2. **Exploratory Data Analysis (EDA):**
           - I filtered out movies with fewer than 100 ratings.
           - Calculated and analyzed the average ratings and number of ratings for these movies.
           - Used a jointplot to visualize the relationship between average rating and number of ratings.

        3. **Create User-Movie Matrix:**
           - Created a matrix with users as rows and movies as columns, using ratings where available.

        4. **Data Normalization:**
           - Normalized the item-item matrix by subtracting each user’s average rating to standardize the scores.

        5. **Identify Similar Users:**
           - Calculated user similarity using Pearson correlation.
           - Identified the top 10 users most similar to user ID `66`, based on a similarity threshold.

        6. **Narrow Down Item Pool:**
           - Filtered out movies that user ID `66` had already watched.
           - Removed movies from the similar user list that had no ratings from similar users.

        7. **Recommend Items:**
           - Computed scores for movies based on ratings from similar users.
           - Generated a list of the top 10 recommended movies.

        8. **Predict Scores:**
           - Calculated user ID `66`’s average movie rating.
           - Adjusted the movie scores by adding this average rating to generate predicted ratings.
           - Displayed the top 10 movies with the highest predicted ratings for personalized recommendations.

        ### **Conclusion**
        Through this process, I successfully filtered and ranked movies to provide tailored recommendations for user ID `66`, focusing on movies they are likely to enjoy based on both their own ratings and the preferences of similar users.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# Part B: Assignment-CF using Item-Item similarity""")
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Initializing movies""")
    return


@app.cell
def __():
    # List of 20 movies
    movies = [
        "Titanic",
        "Gladiator",
        "Avatar",
        "Jurassic Park",
        "Star Wars",
        "The Lion King",
        "Harry Potter",
        "Toy Story",
        "Finding Nemo",
        "Inception",
        "The Matrix",
        "Interstellar",
        "The Dark Knight",
        "Pulp Fiction",
        "Fight Club",
        "Forrest Gump",
        "The Shawshank Redemption",
        "The Godfather",
        "The Lord of the Rings",
        "The Avengers",
        "Tintin and the Secret of the Unicorn",
        "Big Hero 6",
        "Mr. Bean's Holiday",
    ]
    return (movies,)


@app.cell
def __():
    # List of 22 users
    users = [
        "User1",
        "User2",
        "User3",
        "User4",
        "User5",
        "User6",
        "User7",
        "User8",
        "User9",
        "User10",
        "User11",
        "User12",
        "User13",
        "User14",
        "User15",
        "User16",
        "User17",
        "User18",
        "User19",
        "User20",
        "User21",
        "User22",
    ]
    return (users,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""(a) Create a sample dictionary of users with web series and their ratings (minimum 15 users)."""
    )
    return


@app.cell
def __(movies, random, users):
    # Generate a random dictionary of user ratings
    user_ratings = {}

    for user in users:
        # Each user will rate between 5 to 10 movies randomly
        num_movies_rated = random.randint(5, 10)
        rated_movies = random.sample(movies, num_movies_rated)

        # Assign a random rating (1 to 5) for each rated movie
        ratings = {movie: random.randint(1, 5) for movie in rated_movies}

        # Add to the user_ratings dictionary
        user_ratings[user] = ratings

    # Print the user ratings dictionary
    print(user_ratings)
    return num_movies_rated, rated_movies, ratings, user, user_ratings


@app.cell
def __(user_ratings):
    def print_unique_web_series(user_ratings):
        unique_series = set()
        for ratings in user_ratings.values():
            unique_series.update(ratings.keys())
        print("Unique Web Series:", unique_series)

    print_unique_web_series(user_ratings)
    return (print_unique_web_series,)


@app.cell
def __(sqrt, user_ratings):
    def cosine_similarity_custom(item1, item2, user_ratings):
        # Collect ratings for both items
        ratings_item1 = []
        ratings_item2 = []

        for user, ratings in user_ratings.items():
            if item1 in ratings and item2 in ratings:
                ratings_item1.append(ratings[item1])
                ratings_item2.append(ratings[item2])

        # Calculate cosine similarity
        numerator = sum([a * b for a, b in zip(ratings_item1, ratings_item2)])
        denominator = sqrt(sum([a**2 for a in ratings_item1])) * sqrt(
            sum([b**2 for b in ratings_item2])
        )

        if denominator == 0:
            return 0
        else:
            return numerator / denominator

    # Example of calculating cosine similarity between two web series
    print(
        "Cosine Similarity between 'Stranger Things' and 'Breaking Bad':",
        cosine_similarity_custom("Stranger Things", "Breaking Bad", user_ratings),
    )
    return (cosine_similarity_custom,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### 1st Phase: 
         - Find Similarity Between the Target Item with All Other Remaining Items
        """
    )
    return


@app.cell
def __():
    def get_unique_web_series(user_ratings):
        unique_series = set()
        for ratings in user_ratings.values():
            unique_series.update(ratings.keys())
        return unique_series

    return (get_unique_web_series,)


@app.cell
def __(cosine_similarity_custom, get_unique_web_series, user_ratings):
    def find_item_similarities(target_item, user_ratings):
        similarities = {}
        unique_series = get_unique_web_series(
            user_ratings
        )  # Use the updated function to get the unique series
        for series in unique_series:
            if series != target_item:
                similarities[series] = cosine_similarity_custom(
                    target_item, series, user_ratings
                )
        return similarities

    # Example of finding similarities for "Stranger Things"
    similarities = find_item_similarities("Stranger Things", user_ratings)
    print("Similarities with 'Stranger Things':", similarities)
    return find_item_similarities, similarities


@app.cell
def __(cosine_similarity_custom, get_unique_web_series, user_ratings):
    def recommend_web_series(target_user, user_ratings):
        seen_series = user_ratings[target_user].keys()
        unseen_series = set(get_unique_web_series(user_ratings)) - set(seen_series)

        recommendations = {}
        for unseen in unseen_series:
            total_similarity = 0
            weighted_sum = 0
            for seen in seen_series:
                similarity = cosine_similarity_custom(unseen, seen, user_ratings)
                weighted_sum += similarity * user_ratings[target_user][seen]
                total_similarity += similarity
            if total_similarity != 0:
                recommendations[unseen] = weighted_sum / total_similarity

        return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)

    # Example of recommendations for User 3
    recommendations_for_user3 = recommend_web_series("User3", user_ratings)
    print("Recommendations for user 3:", recommendations_for_user3)
    return recommend_web_series, recommendations_for_user3


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### 2nd Phase:
         - Recommending Web Series to the Target User Based on Seen Items
         - Create a function to find seen web series and unseen web series to the target user.
         - Create a function to give recommendations.
        """
    )
    return


@app.cell
def __(recommend_web_series, user_ratings):
    # Example of recommendations for user 12
    recommendations_for_user12 = recommend_web_series("User12", user_ratings)
    print("Recommendations for user12:", recommendations_for_user12)
    return (recommendations_for_user12,)


@app.cell
def __(get_unique_web_series, user_ratings):
    def seen_and_unseen_series(target_user, user_ratings):
        seen_series = set(user_ratings[target_user].keys())
        unseen_series = set(get_unique_web_series(user_ratings)) - seen_series
        return seen_series, unseen_series

    # Example for user7
    seen, unseen = seen_and_unseen_series("User7", user_ratings)
    print("Seen Series for User7:", seen)
    print("Unseen Series for User7:", unseen)
    return seen, seen_and_unseen_series, unseen


@app.cell
def __(recommend_web_series, seen_and_unseen_series, user_ratings):
    def generate_recommendations(target_user, user_ratings):
        seen_series, _ = seen_and_unseen_series(target_user, user_ratings)
        recommendations = recommend_web_series(target_user, user_ratings)
        return recommendations

    # Example of generating recommendations for user7
    final_recommendations = generate_recommendations("User7", user_ratings)
    print("Final Recommendations for user7:", final_recommendations)
    return final_recommendations, generate_recommendations


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""The task utilizes item-based CF to recommend web series to a user from similarities between items they've watched and other web series. For user_12 & 7, the recommended web series include "Finding Nemo" and "Interstellar" with high predicted ratings. The recommendations are derived from comparing the user’s preferences for similar items, providing personalized content based on their viewing history."""
    )
    return


@app.cell(hide_code=True)
def __():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from scipy.stats import pearsonr
    from sklearn.metrics.pairwise import cosine_similarity
    import random
    from math import sqrt

    import marimo as mo

    return cosine_similarity, mo, np, pd, pearsonr, plt, random, sns, sqrt


if __name__ == "__main__":
    app.run()
