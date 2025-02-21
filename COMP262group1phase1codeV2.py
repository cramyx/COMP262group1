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
file_path = r"C:\Users\aalgh\OneDrive\Documents\AI SoftWare  Eng Tech Sem 6\NLP\groupProject\COMP262group1\AMAZON_FASHION_5.json"
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

# distribution of reviews per user (D)
reviews_per_user = df.groupby("reviewerID")["reviewText"].count()
plt.hist(reviews_per_user, bins=50, edgecolor="black")
plt.xlabel("number of reviews per User")
plt.ylabel("frequency")
plt.title("distribution of reviews per user")
plt.show()

# review length analysis (E & F)
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

# %%
# Removing both too short and too long reviews (should decide to implement or not)

# df = df[(review_lengths > 0) & (review_lengths <= upper_bound)] # can rename to df_cleaned if needed

# print(f"Number of reviews after removing outliers: {len(df)}")

# %% [markdown]
# ## Pre-processing
import matplotlib.pyplot as plt
import seaborn as sns
import html
import contractions

print(df["reviewText"].isnull().sum())  # Check how many nulls exist

df["reviewText"] = df["reviewText"].fillna("")  # Replace NaN with empty string

print(df["reviewText"].isnull().sum())  # Check how many nulls exist

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

# Remove duplicates
print(len(df))
df = df.drop_duplicates(subset=["reviewText", "summary", "sentiment"], keep="first")
print(len(df))
# Remove empty reviews

before_empty_removal = len(df)
df = df[df["reviewText"].notna() & (df["reviewText"] != "")]
print(f"\nNumber of rows before removing empty reviews: {before_empty_removal}")
print(f"Number of rows after removing empty reviews: {len(df)}")
# Remove outliers (if needed)
df["review_length"] = df["reviewText"].apply(lambda x: len(x.split()))
q1, q3 = df["review_length"].quantile([0.25, 0.75])
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df = df[(df["review_length"] > 0) & (df["review_length"] <= upper_bound)]
# %% [markdown]
# ## Model Building
