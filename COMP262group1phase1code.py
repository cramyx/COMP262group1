# %% [markdown]
# # PROJECT PHASE 1
# ## GROUP 1 (AMAZON FASHION)

# %%
import pandas as pd
import json
import matplotlib.pyplot as plt

# %% [markdown]
# ## Data Exploration

# %%
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 18:21:29 2025

@author: coles
"""

# DELIVERABLE 1)


# creating a function to load line-delimited json
def load_json_lines(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"error decoding json at line {len(data)}: {e}")

    return pd.DataFrame(data)


# loading the dataset (adjust the path based on where you store it locally on your end)
# file_path = "AMAZON_FASHION_5.json"
file_path = "./COMP262group1/AMAZON_FASHION_5.json"
df = load_json_lines(file_path)

# showing some basic dataset info
print(df.info())
print(df.head())

# basic statistics of the dataset (A)
print(f"total reviews: {len(df)}")
print(f"total unique products: {df['asin'].nunique()}")
print(f"total unique users: {df['reviewerID'].nunique()}")
print(f"average rating: {df['overall'].mean():.2f}")
print(df["overall"].value_counts(normalize=True) * 100)

# distribution of reviews per product (B & C)
reviews_per_product = df.groupby("asin")["reviewText"].count()
plt.hist(reviews_per_product, bins=50, edgecolor="black")
plt.xlabel("number of reviews per product")
plt.ylabel("frequency")
plt.title("distribution of reviews per product")
plt.show()

# distribution of reviews per user
reviews_per_user = df.groupby("reviewerID")["reviewText"].count()
plt.hist(reviews_per_user, bins=50, edgecolor="black")
plt.xlabel("number of reviews per User")
plt.ylabel("frequency")
plt.title("distribution of reviews per user")
plt.show()

# review length analysis
df["review_length"] = df["reviewText"].apply(
    lambda x: len(str(x).split()) if pd.notna(x) else 0
)
print(f"average review length (words): {df['review_length'].mean():.2f}")
print(f"max review length: {df['review_length'].max()}")
print(f"min review length: {df['review_length'].min()}")
plt.hist(df["review_length"], bins=50, edgecolor="black")
plt.xlabel("review length (words)")
plt.ylabel("frequency")
plt.title("distribution of review lengths")
plt.show()

# checking for duplicates and dropping duplicates (G) (this could be moved to the front
# of the script so that the duplicates are removed first before other analysis, up
# to you guys)
print(f"initial dataset size: {len(df)}")
duplicates = df[df.duplicated(subset=["reviewText", "reviewerID", "asin"], keep=False)]
print(f"number of duplicate reviews: {len(duplicates)}")
df = df.drop_duplicates(subset=["reviewText", "reviewerID", "asin"], keep="first")
# updated dataset size after removing duplicates
print(f"dataset size after removing duplicates: {len(df)}")


# %% [markdown]
# ## Text Pre-processing

# %%
# Salma

# Step 2a: Labeling the sentiment

def label_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"


df["sentiment"] = df["overall"].apply(label_sentiment)

print(df["sentiment"].value_counts())
df.head()

# %%
# Step 2b: Selecting columns for sentiment analysis
selected_columns = ["reviewText", "summary", "sentiment"]
df = df[selected_columns]  # can rename to df_selected if needed

# Justification:
# - 'reviewText': Main body of the review, containing the most valuable sentiment information.
# - 'summary': Short version of the review, often useful for reinforcement of sentiment.
# - 'sentiment': The labeled target variable for sentiment analysis. (only needed if labels will be used)
# 'asin' (Product ID) and 'reviewerID' (User ID) could be useful in some cases, to track sentiment trends per product or user. may not be necessary for a basic sentiment analysis.

# %%
# Step 2c: Checking for outliers in review length
review_lengths = df["reviewText"].apply(
    lambda x: len(str(x).split()) if pd.notna(x) else 0
)

# Boxplot of Review Lengths
plt.boxplot(review_lengths)
plt.title("Boxplot of Review Lengths")
plt.ylabel("Number of Words")
plt.show()

# Identifying extreme outliers
q1 = review_lengths.quantile(0.25)
q3 = review_lengths.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Reviews that are too short (length 0 or below)
too_short = df[review_lengths == 0]

# Reviews that are too long (greater than upper bound)
too_long = df[review_lengths > upper_bound]

# Printing samples of too short and too long reviews
print("Sample of reviews that are too short (length 0):")
print(too_short[["reviewText"]].head())

print("\nSample of reviews that are too long:")
print(too_long[["reviewText"]].head())

# Print number of reviews that are too short and too long
print(f"\nNumber of reviews that are too short (length 0): {len(too_short)}")
print(f"Number of reviews that are too long: {len(too_long)}")

# ## Pre-processing
import seaborn as sns
import html
import contractions
# Remove empty reviews

before_empty_removal = len(df)
df = df[df["reviewText"].notna() & (df["reviewText"] != "")]
print(f"\nNumber of rows before removing empty reviews: {before_empty_removal}")
print(f"Number of rows after removing empty reviews: {len(df)}")


# Pre-processing for VADER

print("\nSample of 'reviewText' before VADER preprocessing:")
print(df["reviewText"].head(3))  # Print first 3 rows
df["reviewText"] = df["reviewText"].str.lower()  # Lowercase
df["reviewText"] = df["reviewText"].apply(
    lambda x: html.unescape(x)
)  # Decode HTML entities
# Emojis and punctuation are retained by default; no additional steps needed
print("\nSample of 'reviewText' after VADER preprocessing:")
print(df["reviewText"].head(3))

# Pre-processing for TextBlob


print("\nSample of 'reviewText' before TextBlob preprocessing:")
print(df["reviewText"].head(3))  # Print first 3 rows
df["reviewText"] = df["reviewText"].str.lower()  # Lowercase
df["reviewText"] = df["reviewText"].str.replace(
    r"[^\w\s]", "", regex=True
)  # Remove special characters
df["reviewText"] = df["reviewText"].apply(
    lambda x: contractions.fix(x)
)  # Expand contractions
print("\nSample of 'reviewText' after TextBlob preprocessing:")
print(df["reviewText"].head(3))

# Remove outliers
print(f"\nReview length IQR: {iqr}")
print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
before_outlier_removal = len(df)
df = df[(review_lengths > 0) & (review_lengths <= upper_bound)]
print(f"Number of rows before removing outliers: {before_outlier_removal}")
print(f"Number of rows after removing outliers: {len(df)}")

# Randomly select 1000 reviews from the cleaned dataset
if len(df) >= 1000:
    df = df.sample(n=1000, random_state=42)
else:
    df = df.copy()
print(f"Sampled dataset size: {len(df)}")

# Visualize review lengths
plt.figure(figsize=(12, 6))
sns.histplot(review_lengths, bins=50, kde=True, color="blue")
plt.axvline(lower_bound, color="red", linestyle="--", label="Lower Bound")
plt.axvline(upper_bound, color="green", linestyle="--", label="Upper Bound")
plt.title("Review Length Distribution (After Removing Outliers)")
plt.xlabel("Review Length")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# ## Model Building
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from sklearn.metrics import classification_report

# Downloading necessary resources
nltk.download("vader_lexicon")
# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()


def vader_sentiment(text):
    """Classifies sentiment using VADER"""
    if not isinstance(text, str):  # Converting non-string values to empty strings
        text = ""
    score = sia.polarity_scores(text)
    return (
        "Positive"
        if score["compound"] > 0
        else "Negative" if score["compound"] < 0 else "Neutral"
    )


def textblob_sentiment(text):
    """Classifies sentiment using TextBlob"""
    if not isinstance(text, str):  # Converting non-string values to empty strings
        text = ""
    score = TextBlob(text).sentiment.polarity
    return "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"


# Applying sentiment analysis

df["VADER_Sentiment"] = df["reviewText"].apply(vader_sentiment)
df["TextBlob_Sentiment"] = df["reviewText"].apply(textblob_sentiment)

# Model performance comparison

print("VADER Sentiment Analysis Report:")
print(classification_report(df["sentiment"], df["VADER_Sentiment"]))

print("TextBlob Sentiment Analysis Report:")
print(classification_report(df["sentiment"], df["TextBlob_Sentiment"]))

# Saving the results to CSV
df.to_csv("./COMP262group1/sentiment_analysis_results.csv", index=False)
print("Results saved to sentiment_analysis_results.csv")

# Model Evaluation and Comparison

evaluation_results = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "VADER": [0.81, 0.53, 0.52, 0.52],
    "TextBlob": [0.80, 0.47, 0.46, 0.46],
}
df_eval = pd.DataFrame(evaluation_results)

# Saving the CSV files
# df_eval.to_csv("C:/Users/bless/COMP262group1/model_comparison.csv", index=False)
df_eval.to_csv(
    "./COMP262group1/model_comparison.csv",
    index=False,
)
print("model_comparison.csv saved!")
# df = pd.read_json("C:/Users/bless/COMP262group1/AMAZON_FASHION_5.json", lines=True)
df = pd.read_json("./COMP262group1/AMAZON_FASHION_5.json", lines=True)
# df.to_csv("C:/Users/bless/COMP262group1/sentiment_analysis_results.csv", index=False)
df.to_csv(
    "./COMP262group1/sentiment_analysis_results.csv",
    index=False,
)
print("sentiment_analysis_results.csv saved!")

# Visualizing Model Performance

# plt.figure(figsize=(8, 5))
df_eval.set_index("Metric").plot(kind="bar", colormap="coolwarm", edgecolor="black")
# plt.title("Model Performance Comparison: VADER vs. TextBlob")
# plt.xlabel("Metric")
# plt.ylabel("Score")
# plt.xticks(rotation=0)
# plt.legend(title="Models")
plt.show()
