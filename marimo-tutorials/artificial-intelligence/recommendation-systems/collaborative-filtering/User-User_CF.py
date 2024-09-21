import marimo

__generated_with = "0.7.14"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(r"""# Intro""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# User-Based Collaborative Filtering Recommendation Algorithm""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        User-based collaborative filtering makes recommendations based on user-product interactions in the past. The assumption behind the algorithm is that similar users like similar products.

        It is also called user-user collaborative filtering. It is a type of recommendation system algorithm that uses user similarity to make product recommendations.

        User-based collaborative filtering algorithm usually has the following steps:

        1. Find similar users based on interactions with common items.
        2. Identify the items rated high by similar users but have not been exposed to the active user of interest.
        3. Calculate the weighted average score for each item.
        4. Rank items based on the score and pick top n items to recommend.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(r"""# Step 1: Import Python Libraries""")
    return


@app.cell
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


@app.cell
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
        r"""Using 'movieID' as the matching key, we appended movie information to the rating dataset and named it 'df'. So now we have the movie tile and movie rating in the same dataset!"""
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


@app.cell
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


@app.cell
def __(mo):
    mo.md(r"""# Step 4: Create User-Movie Matrix""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""In step 4, we will transform the dataset into a matrix format. The rows of the matrix are users, and the columns of the matrix are movies. The value of the matrix is the user rating of the movie if there is a rating. Otherwise, it shows 'NaN'."""
    )
    return


@app.cell
def __(df_filtered):
    # Step 4: Create User-Item Matrix
    # Create user-item matrix using the filtered dataset
    matrix = df_filtered.pivot_table(index="userId", columns="title", values="rating")

    # Display the first few rows of the matrix
    print("User-Movie Matrix:")
    print(matrix.head())
    return (matrix,)


@app.cell
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
    user_avg_ratings = matrix.mean(axis=1)

    # Normalize user-item matrix
    matrix_norm = matrix.subtract(user_avg_ratings, axis="rows")

    # Display the first few rows of the normalized matrix
    print("Normalized User-Movie Matrix:")
    print(matrix_norm.head())
    return matrix_norm, user_avg_ratings


@app.cell
def __(mo):
    mo.md(r"""# Step 6: Identify Similar Users""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        There are different ways to measure similarities. Pearson correlation and cosine similarity are two widely used methods.

        In this tutorial, we will calculate the user similarity matrix using Pearson correlation.
        """
    )
    return


@app.cell
def __():
    # User similarity matrix using Pearson correlation
    return


@app.cell
def __(matrix_norm):
    # Step 6: Identify Similar Users

    # User similarity matrix using Pearson correlation
    user_similarity = matrix_norm.T.corr()

    # Display the first few rows of the user similarity matrix
    print("User Similarity Matrix (Pearson Correlation):")
    print(user_similarity.head())
    return (user_similarity,)


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
def __(user_similarity):
    # Pick a user ID
    picked_userid = 66

    # Remove picked user ID from the candidate list
    user_similarity.drop(index=picked_userid, inplace=True)

    # Take a look at the data
    print("\nUser Similarity Matrix after removing picked user ID:")
    print(user_similarity.head())
    return (picked_userid,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        In the user similarity matrix, the values range from -1 to 1, where -1 means opposite movie preference and 1 means same movie preference.

        `n = 10` means we would like to pick the top 10 most similar users for user ID 66.

        The user-based collaborative filtering makes recommendations based on users with similar tastes, so we need to set a positive threshold. Here we set the `user_similarity_threshold` to be 0.3, meaning that a user must have a Pearson correlation coefficient of at least 0.3 to be considered as a similar user.

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
    user_similarity_cosine = pd.DataFrame(
        cosine_similarity(matrix_filled),
        index=matrix_filled.index,
        columns=matrix_filled.index,
    )

    # Display the first few rows of the cosine similarity matrix
    print("User Similarity Matrix (Cosine Similarity):")
    print(user_similarity_cosine.head())
    return matrix_filled, user_similarity_cosine


@app.cell
def __(picked_userid, user_similarity, user_similarity_cosine):
    # Number of similar users
    n = 10

    # User similarity threashold
    user_similarity_threshold = 0.3

    # Get top n similar users
    similar_users = user_similarity[
        user_similarity[picked_userid] > user_similarity_threshold
    ][picked_userid].sort_values(ascending=False)[:n]

    # Remove picked user ID from the candidate list for both similarity matrices

    # user_similarity.drop(index=picked_userid, inplace=True) # did this above
    user_similarity_cosine.drop(index=picked_userid, inplace=True)

    # Pearson Similarity
    # Get top n similar users using Pearson similarity
    similar_users_pearson = user_similarity[
        user_similarity[picked_userid] > user_similarity_threshold
    ][picked_userid].sort_values(ascending=False)[:n]

    # Cosine Similarity
    # Get top n similar users using Cosine similarity
    similar_users_cosine = user_similarity_cosine[
        user_similarity_cosine[picked_userid] > user_similarity_threshold
    ][picked_userid].sort_values(ascending=False)[:n]

    # Compare the two sets of similar users
    print(f"Top {n} similar users for user {picked_userid} using Pearson similarity:")
    print(similar_users_pearson)

    print(f"Top {n} similar users for user {picked_userid} using Cosine similarity:")
    print(similar_users_cosine)
    return (
        n,
        similar_users,
        similar_users_cosine,
        similar_users_pearson,
        user_similarity_threshold,
    )


@app.cell
def __(mo):
    mo.md(r"""# Step 7: Narrow Down Item Pool""")
    return


@app.cell
def __(matrix_norm, picked_userid):
    # Movies that the target user has watched
    picked_userid_watched = matrix_norm[matrix_norm.index == picked_userid].dropna(
        axis=1, how="all"
    )
    picked_userid_watched
    return (picked_userid_watched,)


@app.cell
def __(matrix_norm, similar_users_cosine, similar_users_pearson):
    # Step 7: Narrow Down Item Pool

    # Movies that the target user has watched
    # Movies that similar users watched. Remove movies that none of the similar users have watched
    similar_user_movies_pearson = matrix_norm.loc[
        matrix_norm.index.isin(similar_users_pearson.index)
    ].dropna(axis=1, how="all")
    similar_user_movies_cosine = matrix_norm.loc[
        matrix_norm.index.isin(similar_users_cosine.index)
    ].dropna(axis=1, how="all")
    return similar_user_movies_cosine, similar_user_movies_pearson


@app.cell
def __(
    picked_userid_watched,
    similar_user_movies_cosine,
    similar_user_movies_pearson,
):
    # Remove the watched movie from the movie list
    similar_user_movies_pearson.drop(
        picked_userid_watched.columns, axis=1, inplace=True, errors="ignore"
    )
    similar_user_movies_cosine.drop(
        picked_userid_watched.columns, axis=1, inplace=True, errors="ignore"
    )
    return


@app.cell
def __(similar_user_movies_cosine, similar_user_movies_pearson):
    # Display the final list of movies that similar users have watched, excluding those already watched by the picked user
    print("\nMovies that similar users have watched using Pearson similarity:")
    print(similar_user_movies_pearson)

    print("\nMovies that similar users have watched using Cosine similarity:")
    print(similar_user_movies_cosine)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""To keep only the similar users' movies, we keep the user IDs in the top 10 similar user lists and remove the film with all missing values. All missing value for a movie means that none of the similar users have watched the movie."""
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""Next, we will drop the movies that user ID 66 watched from the similar user movie list. `errors='ignore'` drops columns if they exist without giving an error message."""
    )
    return


@app.cell
def __(mo):
    mo.md(r"""# Step 8: Recommend Items""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        In step 8, we will decide which movie to recommend to the target user. The recommended items are determined by the weighted average of user similarity score and movie rating. The movie ratings are weighted by the similarity scores, so the users with higher similarity get higher weights.

        This code loops through items and users to get the item score, rank the score from high to low and pick the top 10 movies to recommend to user ID 66.
        """
    )
    return


@app.cell
def __(pd):
    # Step 8: Recommend Items

    def recommend_items(similar_users, similar_user_movies):
        # A dictionary to store item scores
        item_score = {}

        # Loop through items (movies) in the similar_user_movies DataFrame
        for movie in similar_user_movies.columns:
            # Get the ratings for this movie from similar users
            movie_rating = similar_user_movies[movie]

            # Initialize variables to store the total weighted score and count of valid ratings
            total = 0
            count = 0

            # Loop through similar users
            for user in similar_users.index:
                # If the movie has a rating from this user
                if not pd.isna(movie_rating[user]):
                    # Calculate the weighted score: similarity score * movie rating
                    score = similar_users[user] * movie_rating[user]
                    # Add the weighted score to the total score for this movie
                    total += score
                    # Increment the count of valid ratings
                    count += 1

            # Calculate the average score for this movie
            if count > 0:
                item_score[movie] = total / count
            else:
                item_score[movie] = 0

        # Convert the item_score dictionary to a pandas DataFrame
        item_score_df = pd.DataFrame(
            item_score.items(), columns=["movie", "movie_score"]
        )

        # Sort the movies by their scores in descending order
        ranked_item_score = item_score_df.sort_values(by="movie_score", ascending=False)

        # Select the top m movies
        m = 10
        top_recommendations = ranked_item_score.head(m)

        return top_recommendations

    return (recommend_items,)


@app.cell
def __(
    recommend_items,
    similar_user_movies_cosine,
    similar_user_movies_pearson,
    similar_users_cosine,
    similar_users_pearson,
):
    top_recommendations_pearson = recommend_items(
        similar_users_pearson, similar_user_movies_pearson
    )
    top_recommendations_cosine = recommend_items(
        similar_users_cosine, similar_user_movies_cosine
    )

    print("\nTop 10 recommended movies using Pearson similarity:")
    print(top_recommendations_pearson)

    print("\nTop 10 recommended movies using Cosine similarity:")
    print(top_recommendations_cosine)
    return top_recommendations_cosine, top_recommendations_pearson


@app.cell
def __(mo):
    mo.md(r"""# Step 9: Predict Scores""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""If the goal is to choose the recommended items, having the rank of the items is enough. However, if the goal is to predict the user's rating, we need to add the user's average movie rating score back to the movie score."""
    )
    return


@app.cell
def __(matrix):
    # Step 9: Predict Scores

    def predict_scores(top_recommendations, picked_userid, m):
        # Calculate the average rating for the picked user
        avg_rating = matrix.loc[matrix.index == picked_userid].T.mean()[picked_userid]
        print(avg_rating)

        # Calculate the predicted rating
        # Add the average rating back to the movie scores to get predicted ratings
        top_recommendations["predicted_rating"] = (
            top_recommendations["movie_score"] + avg_rating
        )

        # Sort the movies by predicted ratings in descending order
        ranked_predicted_ratings = top_recommendations.sort_values(
            by="predicted_rating", ascending=False
        )

        # Select the top m movies with the highest predicted ratings
        top_predicted_recommendations = ranked_predicted_ratings.head(m)

        return top_predicted_recommendations

    return (predict_scores,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Explanation:

        1. **Define the Target User ID:**
           - We set `picked_userid = 66` to use the roll number as the user ID.

        2. **Calculate Average Rating:**
           - Compute the average rating given by the user with ID 66. This is done using `matrix[matrix.index == picked_userid].T.mean()[picked_userid]`.

        3. **Add Average Rating to Scores:**
           - Add the average rating to each movie's score to compute the predicted ratings. This adjustment provides an estimate of how the user would rate each movie.

        4. **Sort and Select Top Movies:**
           - Sort the movies based on predicted ratings and select the top `m` movies with the highest predicted ratings.

        5. **Display Results:**
           - Print the top 10 recommended movies along with their predicted ratings.

        ## Example Output:

        The output will display the average rating for the target user and the top 10 recommended movies with their predicted ratings. This provides a personalized list of movies that the user is likely to rate highly based on their average rating and the preferences of similar users.
        """
    )
    return


@app.cell
def __(
    picked_userid,
    predict_scores,
    top_recommendations_cosine,
    top_recommendations_pearson,
):
    # Example usage:
    m = 10
    top_predicted_recommendations_pearson = predict_scores(
        top_recommendations_pearson, picked_userid, m
    )
    top_predicted_recommendations_cosine = predict_scores(
        top_recommendations_cosine, picked_userid, m
    )

    print(
        "\nTop 10 recommended movies with predicted ratings using Pearson similarity:"
    )
    print(top_predicted_recommendations_pearson)

    print("\nTop 10 recommended movies with predicted ratings using Cosine similarity:")
    print(top_predicted_recommendations_cosine)
    return (
        m,
        top_predicted_recommendations_cosine,
        top_predicted_recommendations_pearson,
    )


@app.cell
def __(mo):
    mo.md(
        r"""The average movie rating for user 66 is 4.18, so we add 4.18 back to the movie score."""
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""We can see that the top 10 recommended movies all have predicted ratings greater than 4.345."""
    )
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
           - Normalized the user-item matrix by subtracting each user’s average rating to standardize the scores.

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
def __():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from scipy.stats import pearsonr
    from sklearn.metrics.pairwise import cosine_similarity

    import marimo as mo

    return cosine_similarity, mo, np, pd, pearsonr, plt, sns


if __name__ == "__main__":
    app.run()
