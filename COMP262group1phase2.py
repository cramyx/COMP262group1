# -*- coding: utf-8 -*-
"""
Created on Sun Apr 6 14:18:49 2025

@author: coles
"""

#PROJECT PHASE 2
#GROUP 1 (AMAZON FASHION)

import pandas as pd
import json
import matplotlib.pyplot as plt

#DELIVERABLE 1) DATA EXPLORATION & PREPROCESSING

# loading a subset of the full dataset
def load_json_lines(path, limit=5000):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit: break
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON at line {i}: {e}")
    return pd.DataFrame(data)

# loading full Amazon Fashion dataset subset
file_path = "C:/Users/coles/Downloads/AMAZON_FASHION.json/AMAZON_FASHION.json"
df = load_json_lines(file_path, limit=5000)
print(f"Loaded {len(df)} reviews.")

# basic exploration
print(df.info())
print(df.head())
print(f"Total unique products: {df['asin'].nunique()}")
print(f"Total unique users: {df['reviewerID'].nunique()}")
print(f"Average rating: {df['overall'].mean():.2f}")
print(df['overall'].value_counts(normalize=True) * 100)

# review Length Analysis
df["review_length"] = df["reviewText"].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
print(f"Average review length: {df['review_length'].mean():.2f}")
print(f"Max: {df['review_length'].max()} | Min: {df['review_length'].min()}")

# distribution visualizations
# reviews per product
reviews_per_product = df.groupby("asin")["reviewText"].count()
plt.hist(reviews_per_product, bins=50, edgecolor='black')
plt.title("Distribution of Reviews per Product")
plt.xlabel("Reviews per Product")
plt.ylabel("Frequency")
plt.show()

# reviews per user
reviews_per_user = df.groupby("reviewerID")["reviewText"].count()
plt.hist(reviews_per_user, bins=50, edgecolor='black')
plt.title("Distribution of Reviews per User")
plt.xlabel("Reviews per User")
plt.ylabel("Frequency")
plt.show()

# review length distribution
plt.hist(df["review_length"], bins=50, edgecolor='black')
plt.title("Distribution of Review Lengths")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.show()

# handling duplicates
print(f"Initial dataset size: {len(df)}")
duplicates = df[df.duplicated(subset=["reviewText", "reviewerID", "asin"], keep=False)]
print(f"Duplicate reviews found: {len(duplicates)}")
df = df.drop_duplicates(subset=["reviewText", "reviewerID", "asin"], keep='first')
print(f"Dataset size after removing duplicates: {len(df)}")

# removing empty reviews
df = df[df["reviewText"].notna() & (df["reviewText"] != "")]

# label sentiment
def label_sentiment(score):
    if score >= 4:
        return "Positive"
    elif score == 3:
        return "Neutral"
    else:
        return "Negative"

df["sentiment"] = df["overall"].apply(label_sentiment)
df["reviewText"] = df["reviewText"].str.lower()
df["reviewText"] = df["reviewText"].str.replace(r"[^\w\s]", "", regex=True)

# keeping relevant columns
df = df[["reviewText", "sentiment"]]

# final overview
print("Sample labeled data:")
print(df.head(10))