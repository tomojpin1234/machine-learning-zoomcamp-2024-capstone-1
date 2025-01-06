import ast
import pickle
import re
import string
from collections import Counter

import nltk
import numpy as np
import pandas as pd
import stopwordsiso as stopwords
from lightgbm import LGBMRegressor

from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from textblob import TextBlob

import warnings

# parameters
max_depth = -1
n_estimators = 180
learning_rate = 0.1
colsample_bytree = 0.6
min_child_samples = 20
num_leaves = 31
subsample = 1.0

output_file = "models/model_lgbmr.bin"
data_file = "data/processed_links.csv"
test_file = "data/df_test.csv"
features_file = "data/features.pkl"

warnings.filterwarnings("ignore")

# data preparation


# Extract the hour and classify the time of day
def classify_time_of_day(hour):
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    elif 18 <= hour < 24:
        return "evening"
    else:
        return "night"


# Categorize polarity into sentiment groups
def categorize_sentiment(polarity):
    if polarity < -0.3:
        return "Negative"
    elif polarity > 0.3:
        return "Positive"
    else:
        return "Neutral"


def tokenize_and_clean(text):
    # Convert to lowercase, remove punctuation, and split into words
    tokens = text.lower().translate(str.maketrans("", "", string.punctuation)).split()
    # Remove Polish stopwords
    tokens = [word for word in tokens if word not in polish_stop_words]
    return tokens


# Preprocess: Ensure valid Python list syntax
def clean_tags(tag_string):
    # Replace invalid single quotes with double quotes if needed
    if isinstance(tag_string, str):
        tag_string = re.sub(
            r"'", '"', tag_string
        )  # Replace single quotes with double quotes
    return tag_string


def get_features(use_categorical=10):
    numerical = [
        "comments_count",
        "hour_of_day",
        "title_char_length",
        "description_char_length",
        "title_word_count",
        "description_word_count",
        "title_unique_word_count",
        "description_unique_word_count",
        "title_avg_word_length",
        "description_avg_word_length",
        "title_sentiment_polarity",
        "title_sentiment_subjectivity",
        "description_sentiment_polarity",
        "description_sentiment_subjectivity",
    ]

    base_categorical = [
        "plus18",
        "day_of_week",
        "time_of_day",
    ]

    df = None

    df, title_contains_features, tfidf_top_features, tag_features = load_data(
        tfidf_top=use_categorical,
        tags_top=use_categorical,
        bottom_upvote=0,
        top_upvote=3000,
        most_common_words=use_categorical,
    )
    categorical = (
        base_categorical + title_contains_features + tfidf_top_features + tag_features
    )

    return df, categorical, numerical


def load_data(
    tfidf_top=10, tags_top=10, bottom_upvote=0, top_upvote=1000, most_common_words=10
):
    df = pd.read_csv(data_file)
    df = df.reset_index(drop=True)

    # Filter DataFrame
    df = df[(df["upvote_count"] >= bottom_upvote) & (df["upvote_count"] <= top_upvote)]

    df["creation_date"] = pd.to_datetime(df["creation_date"])

    del df["Unnamed: 0"]
    del df["downvotes"]
    del df["upvotes"]
    del df["info"]
    del df["link_id"]
    del df["author_user_id"]
    del df["status"]
    del df["can_vote"]
    del df["archived"]

    df["hour_of_day"] = df["creation_date"].dt.hour

    df["description"] = df["description"].fillna("")
    df["tags"] = df["tags"].fillna("['no_tags']")

    # Apply cleaning and safely evaluate as list
    df["tags_cleaned"] = df["tags"].apply(clean_tags)
    df["tags_list"] = df["tags_cleaned"].apply(ast.literal_eval)

    # Sort and join the tags
    df["tags_sorted_joined"] = df["tags_list"].apply(lambda x: "_".join(sorted(x)))

    # Extract the day of the week
    df["day_of_week"] = df["creation_date"].dt.day_name()
    # Extract the hour and classify the time of day
    df["time_of_day"] = df["creation_date"].dt.hour.apply(classify_time_of_day)

    df["title_char_length"] = df["title"].str.len()
    df["title_length_bins"] = pd.cut(df["title_char_length"], bins=range(0, 150, 30))
    df["description_char_length"] = df["description"].str.len()

    df["title_word_count"] = df["title"].str.split().apply(len)
    df["description_word_count"] = df["description"].str.split().apply(len)

    df["title_unique_word_count"] = df["title"].str.split().apply(lambda x: len(set(x)))
    df["description_unique_word_count"] = (
        df["description"].str.split().apply(lambda x: len(set(x)))
    )

    df["title_sentiment_polarity"] = df["title"].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )
    df["title_sentiment_subjectivity"] = df["title"].apply(
        lambda x: TextBlob(x).sentiment.subjectivity
    )

    df["description_sentiment_polarity"] = df["description"].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )
    df["description_sentiment_subjectivity"] = df["description"].apply(
        lambda x: TextBlob(x).sentiment.subjectivity
    )

    df["title_sentiment_group"] = df["title_sentiment_polarity"].apply(
        categorize_sentiment
    )
    df["description_sentiment_group"] = df["description_sentiment_polarity"].apply(
        categorize_sentiment
    )

    df["title_tokens"] = df["title"].apply(
        lambda x: tokenize_and_clean(x) if isinstance(x, str) else []
    )
    df["description_tokens"] = df["description"].apply(
        lambda x: tokenize_and_clean(x) if isinstance(x, str) else []
    )

    popular_threshold = 100
    popular_posts = df[df["upvote_count"] > popular_threshold]
    popular_posts.head().T

    popular_title_tokens = [
        token for tokens in popular_posts["title_tokens"] for token in tokens
    ]
    # popular_description_tokens = [token for tokens in popular_posts['description_tokens'] for token in tokens]

    popular_title_word_freq = Counter(popular_title_tokens)
    # popular_description_word_freq = Counter(popular_description_tokens)

    # Most common words
    popular_title_top_words = popular_title_word_freq.most_common(most_common_words)

    # Example words to investigate
    words_to_check = [word for word, count in popular_title_top_words]
    words_to_check

    title_contains_features = []

    for word in words_to_check:
        title_name = f"title_contains_{word}"
        df[title_name] = (
            df["title"].str.contains(word, case=False, na=False).astype(int)
        )
        title_contains_features.append(title_name)

    df["title_avg_word_length"] = df["title"].apply(
        lambda x: np.mean([len(word) for word in x.split()]) if pd.notnull(x) else 0
    )
    df["description_avg_word_length"] = df["description"].apply(
        lambda x: np.mean([len(word) for word in x.split()]) if pd.notnull(x) else 0
    )
    df["description_avg_word_length"] = df["description_avg_word_length"].fillna(0.0)

    tfidf = TfidfVectorizer(max_features=tfidf_top)  # Adjust max_features as needed
    tfidf_matrix = tfidf.fit_transform(df["title"].fillna(""))

    # Convert to DataFrame
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out()
    )
    tfidf_df.head().T

    correlations = tfidf_df.corrwith(df["upvote_count"])

    # Sort by absolute correlation
    correlations = correlations.abs().sort_values(ascending=False)
    tfidf_top_features = correlations.head(tfidf_top).index
    tfidf_top_df = tfidf_df[tfidf_top_features]
    tfidf_top_df

    # Concatenate the filtered TF-IDF DataFrame with the original DataFrame
    df = pd.concat(
        [df.reset_index(drop=True), tfidf_top_df.reset_index(drop=True)], axis=1
    )

    all_tags = df["tags_list"].explode()
    all_tags

    tag_counts = all_tags.value_counts()
    tag_counts
    top_tags = tag_counts.head(tags_top).index
    tag_features = []
    for tag in top_tags:
        tag_feature_name = f"tags_contains_{tag}"
        df[tag_feature_name] = (
            df["tags_sorted_joined"].str.contains(tag, case=False, na=False).astype(int)
        )
        tag_features.append(tag_feature_name)

    return df, title_contains_features, list(tfidf_top_features), tag_features


def split_data(df):
    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=1, shuffle=True
    )
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, random_state=1, shuffle=True
    )

    y_train = np.log1p(df_train.upvote_count.values)
    y_val = np.log1p(df_val.upvote_count.values)
    y_test = np.log1p(df_test.upvote_count.values)

    y_test_original = df_test.upvote_count.values

    del df_train["upvote_count"]
    del df_val["upvote_count"]
    del df_test["upvote_count"]

    return (
        df_full_train,
        df_train,
        y_train,
        df_val,
        y_val,
        df_test,
        y_test,
        y_test_original,
    )


# training


def train_lgbmr_with_scaler(
    df,
    y_train,
    categorical,
    numerical,
    n_estimators=100,
    max_depth=-1,
    learning_rate=0.1,
    colsample_bytree=1.0,
    min_child_samples=20,
    num_leaves=31,
    subsample=1.0,
):
    X_train_num = df[numerical].values
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_num)
    X_train_num

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_train_cat = ohe.fit_transform(df[categorical].values)
    X_train_cat

    X_train = np.column_stack([X_train_num, X_train_cat])
    model = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        colsample_bytree=colsample_bytree,
        min_child_samples=min_child_samples,
        num_leaves=num_leaves,
        subsample=subsample,
    )
    model.fit(X_train, y_train)
    return scaler, ohe, model


def predict_lgbmr_with_scaler(df, scaler, ohe, model, y_val, categorical, numerical):
    X_val_num = df[numerical].values
    X_val_num = scaler.transform(X_val_num)
    X_val_cat = ohe.transform(df[categorical].values)
    X_val = np.column_stack([X_val_num, X_val_cat])

    y_pred = model.predict(X_val)

    rmse = root_mean_squared_error(y_val, y_pred)
    return rmse


# validation

print("Preparing data")

nltk.download("stopwords")
polish_stop_words = set(stopwords.stopwords("pl"))

has_polish_lang = stopwords.has_lang("pl")
print("Has polish words: ", has_polish_lang)

df, categorical, numerical = get_features(use_categorical=100)
(
    df_full_train,
    df_train,
    y_train,
    df_val,
    y_val,
    df_test,
    y_test,
    y_test_original,
) = split_data(df)

print("Doing validation")

n_splits = 2
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
scores = []
iteration = 0
for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = np.log1p(df_train.upvote_count.values)
    y_val = np.log1p(df_val.upvote_count.values)

    del df_train["upvote_count"]
    del df_val["upvote_count"]

    scaler, ohe, model = train_lgbmr_with_scaler(
        df_train,
        y_train,
        categorical,
        numerical,
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        colsample_bytree=colsample_bytree,
        min_child_samples=min_child_samples,
        num_leaves=num_leaves,
        subsample=subsample,
    )
    rmse = predict_lgbmr_with_scaler(
        df_val, scaler, ohe, model, y_val, categorical, numerical
    )

    print(f"Fold: {iteration} rmse: {rmse}")
    iteration += 1
    scores.append(rmse)

print("Validation results:")
print("%.3f += %.3f" % (np.mean(scores), np.std(scores)))

# training the final model

print("Training the final model")

full_df_train = pd.concat([df_train, df_val], axis=0, ignore_index=True)
full_df_train = full_df_train.reset_index(drop=True)
full_y_train = np.concatenate((y_train, y_val), axis=0)

scaler, ohe, model = train_lgbmr_with_scaler(
    df_train,
    y_train,
    categorical,
    numerical,
    max_depth=max_depth,
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    colsample_bytree=colsample_bytree,
    min_child_samples=min_child_samples,
    num_leaves=num_leaves,
    subsample=subsample,
)
rmse = predict_lgbmr_with_scaler(
    df_val, scaler, ohe, model, y_val, categorical, numerical
)
print(f"RMSE={rmse}")

# Save the model
with open(output_file, "wb") as f_out:
    pickle.dump((scaler, ohe, model), f_out)
print(f"Model is saved to {output_file}")

# Export the test data to a file
df_test["upvote_count"] = y_test_original
df_test.to_csv(test_file, index=False)
print(f"Test dataframe is saved to {test_file}")


# Export features
with open(features_file, "wb") as f:
    pickle.dump({"categorical": categorical, "numerical": numerical}, f)

print(f"Features saved to {features_file}")
