# %%
#PROJECT PHASE 2
#GROUP 1 (AMAZON FASHION)

# %%
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# %% [markdown]
# # Data Exploration & Preprocessing

# %%
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
file_path = "AMAZON_FASHION.json"
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

# %% [markdown]
# # Text Representation & Data Splitting

# %%
# Check class distribution
print(df['sentiment'].value_counts())

# Plot the distribution of sentiments
plt.figure(figsize=(6, 4))
df['sentiment'].value_counts().plot(kind='bar')
plt.title('Sentiment Class Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# %%
# Text Representation
# Apply TF-IDF Vectorizer to 'reviewText' column
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["reviewText"])

# Original labels (non-encoded)
y_orig = df['sentiment']

# %%
# # Another vectorizer if you want to experiment

# from sklearn.feature_extraction.text import CountVectorizer

# # Apply Count Vectorizer
# vectorizer = CountVectorizer(max_features=5000)
# X = vectorizer.fit_transform(df["reviewText"])  # Convert text data into features

# # Original labels (non-encoded)
# y_orig = df['sentiment']

# %%
# Encode labels for models like XGBoost, MLP, etc.
le = LabelEncoder()
y_encoded = le.fit_transform(y_orig)

# Split the dataset into features (X) and labels (y) first, and make sure we are stratifying based on the labels
# 70% Training and 30% Testing with encoded labels for models that require encoding
X_train_enc, X_test_enc, y_train_enc, y_test_enc = train_test_split(X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)

# 70% Training and 30% Testing with original labels for models that don't require encoding
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X, y_orig, test_size=0.3, stratify=y_orig, random_state=42)

# There is (choose which one to use based on the model):
# 1. Encoded splits: X_train_enc, X_test_enc, y_train_enc, y_test_enc
# 2. Original splits: X_train_orig, X_test_orig, y_train_orig, y_test_orig

# %%
from imblearn.over_sampling import SMOTE

# Applying SMOTE only on the training set to balance the classes
smote = SMOTE(random_state=42)

# Apply SMOTE on the encoded labels training set
X_train_enc_smote, y_train_enc_smote = smote.fit_resample(X_train_enc, y_train_enc)

# Apply SMOTE on the original labels training set
X_train_orig_smote, y_train_orig_smote = smote.fit_resample(X_train_orig, y_train_orig)

# Print class distribution after SMOTE
smote_counts = pd.Series(y_train_enc_smote).value_counts().sort_index()
smote_labels = le.inverse_transform(smote_counts.index)

print("\nClass distribution after SMOTE:")
for label, count in zip(smote_labels, smote_counts):
    print(f"{label}: {count}")
    
# print total number of samples after SMOTE
print(f"Total samples after SMOTE: {X_train_enc_smote.shape[0]}")

# %% [markdown]
# # Gradient Boost (extra model, can remove later)

# %%
# Gradient Boosting Classifier using sklearn WITHOUT SMOTE
from sklearn.ensemble import GradientBoostingClassifier

# Train on original encoded training data
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train_enc, y_train_enc)
y_pred = gb_model.predict(X_test_enc)

# Evaluation using original labels
print("Gradient Boosting With Sklearn (Original Data) Classification Report:")
print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test_enc, y_pred))

print("\nGradient Boosting Classifier Accuracy:")
print(gb_model.score(X_test_enc, y_test_enc))

# %%
# Train with SMOTE-balanced training data
gb_model_smote = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model_smote.fit(X_train_enc_smote, y_train_enc_smote)
y_pred_smote = gb_model_smote.predict(X_test_enc)

# Evaluation using original labels
print("\nGradient Boosting With Sklearn (With SMOTE) Classification Report:")
print(classification_report(y_test_enc, y_pred_smote, target_names=le.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test_enc, y_pred_smote))

print("\nGradient Boosting Classifier Accuracy (With SMOTE):")
print(gb_model_smote.score(X_test_enc, y_test_enc))

# %%
# Gradient Boosting Classifier using XGBoost WITHOUT SMOTE
from xgboost import XGBClassifier

# Fit the model on original data
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train_orig, y_train_enc)  # Train with encoded labels
y_pred = xgb_model.predict(X_test_orig)  # Predict using the test set with original labels

# Inverse transform the predictions back to original labels
y_pred_labels = le.inverse_transform(y_pred)

# Evaluation
print("XGBoost Classification Report (Without SMOTE):")
print(classification_report(y_test_orig, y_pred_labels))

print("Confusion Matrix (Without SMOTE):")
print(confusion_matrix(y_test_orig, y_pred_labels))

print("\nXGBoost Classifier Accuracy (Without SMOTE):")
print(xgb_model.score(X_test_orig, y_test_enc))  # Use encoded labels for score calculation

# %%
# Fit the model on SMOTE balanced data
xgb_model_smote = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_model_smote.fit(X_train_enc_smote, y_train_enc_smote)  # Train with SMOTE balanced data
y_pred_smote = xgb_model_smote.predict(X_test_orig)  # Predict using the original test data (no SMOTE)

# Inverse transform the predictions back to original labels
y_pred_smote_labels = le.inverse_transform(y_pred_smote)

# Evaluation after SMOTE
print("XGBoost Classification Report (With SMOTE):")
print(classification_report(y_test_orig, y_pred_smote_labels))

print("Confusion Matrix (With SMOTE):")
print(confusion_matrix(y_test_orig, y_pred_smote_labels))

print("\nXGBoost Classifier Accuracy (With SMOTE):")
print(xgb_model_smote.score(X_test_orig, y_test_enc))  # Use encoded labels for score calculation

# Model 1 (Logistic Regression Model Building and Hyperparameter Tuning)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Initialize logistic regression
lr = LogisticRegression(random_state=42, max_iter=1000)

# Fit to training data
lr.fit(X_train_orig, y_train_orig)

# Predict on test data
y_pred_lr = lr.predict(X_test_orig)

# Evaluate
print("Accuracy:", accuracy_score(y_test_orig, y_pred_lr))
print("\nClassification Report:\n", classification_report(y_test_orig, y_pred_lr))

# Hyperparameter Tuning

# Define parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear']
}

# Grid Search
grid = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000), param_grid, cv=5, scoring='accuracy', verbose=1)
grid.fit(X_train_orig, y_train_orig)

print("Best Parameters:", grid.best_params_)

# Predict with best model
y_pred_best = grid.best_estimator_.predict(X_test_orig)

# Final evaluation
print("\nBest Model Accuracy:", accuracy_score(y_test_orig, y_pred_best))
print("\nBest Model Classification Report:\n", classification_report(y_test_orig, y_pred_best))

# Visualization

# Training vs. Testing Accuracy

import matplotlib.pyplot as plt

# Results you have
train_accuracy = 0.813  # After tuning 
test_accuracy = 0.8127  # Slight rounding

# Creating a bar chart
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(['Training Accuracy', 'Testing Accuracy'], [train_accuracy, test_accuracy], color=['skyblue', 'lightgreen'])

# Add text labels on bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  
                textcoords="offset points",
                ha='center', va='bottom')

ax.set_ylim(0, 1)
ax.set_title('Training vs. Testing Accuracy for Logistic Regression')
ax.set_ylabel('Accuracy')
plt.show()

# Visualization

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Confusion matrix
cm = confusion_matrix(y_test_orig, y_pred_best)

# Displaying the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lr.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for Logistic Regression")
plt.show()

import seaborn as sns

# Generate classification report
from sklearn.metrics import classification_report
import pandas as pd

report = classification_report(y_test_orig, y_pred_best, output_dict=True)
df_report = pd.DataFrame(report).transpose()

plt.figure(figsize=(8,6))
sns.heatmap(df_report.iloc[:-1, :-1], annot=True, cmap='Blues')
plt.title('Classification Report Heatmap for Logistic Regression')
plt.show()

# %% [markdown]
# # Model 2

# %%



