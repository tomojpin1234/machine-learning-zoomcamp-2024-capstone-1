import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

# Parameters
output_file = "models/model_lgbmr.bin"
data_file = "data/processed_links.csv"

features_file = "data/features.pkl"

test_file = "data/df_test.csv"
test_posts = pd.read_csv(test_file)


app = Flask(__name__)
CORS(app)


# Global variables
scaler, ohe, model = None, None, None


def load_model_once(output_file):
    global scaler, ohe, model
    if scaler is None or ohe is None or model is None:
        with open(output_file, "rb") as f_in:
            scaler, ohe, model = load_model(output_file)


def get_features():
    with open(features_file, "rb") as f:
        features = pickle.load(f)

    categorical_loaded = features["categorical"]
    numerical_loaded = features["numerical"]

    print("Categorical:", len(categorical_loaded))
    print("Numerical:", len(numerical_loaded))

    return categorical_loaded, numerical_loaded


def load_model(output_file: str):
    with open(output_file, "rb") as f_in:
        scaler, ohe, model = pickle.load(f_in)
    return scaler, ohe, model


def predict_post_engagement(df):
    categorical, numerical = get_features()
    scaler, ohe, model = load_model(output_file)
    X_val_num = df[numerical].values
    X_val_num = scaler.transform(X_val_num)
    X_val_cat = ohe.transform(df[categorical].values)
    X_val = np.column_stack([X_val_num, X_val_cat])

    y_pred = model.predict(X_val)
    return y_pred


@app.route("/random-post", methods=["GET"])
def random_post():
    # Select a random row from the DataFrame
    random_row = test_posts.sample(n=1).iloc[0]
    random_post_data = random_row.to_dict()
    return jsonify(random_post_data)


@app.route("/predict", methods=["POST"])
def predict():
    if request.is_json:
        # Parse JSON data
        post = request.get_json()
    else:
        # Parse form data
        post = request.form.to_dict()

    print("predict")
    print("post: ", post)

    df_post = pd.DataFrame([post])

    y_pred = predict_post_engagement(df_post)
    upvote_predicted = np.expm1(y_pred[0])

    print(f"Predicted upvotes is {upvote_predicted}")
    result = {"post_upvotes": int(upvote_predicted)}
    return jsonify(result)


@app.route("/", methods=["GET"])
def form():
    return render_template("index.html")


if __name__ == "__main__":
    load_model_once(output_file)
    app.run(debug=True, host="0.0.0.0", port=8080)
