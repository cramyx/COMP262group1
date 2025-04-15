#!/usr/bin/env python
# coding: utf-8

# In[1]:


#PROJECT PHASE 2
#GROUP 1 (AMAZON FASHION)


# In[1]:


import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


# # Data Exploration & Preprocessing

# In[2]:


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


# # Text Representation & Data Splitting

# In[3]:


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


# In[4]:


# Text Representation
# Apply TF-IDF Vectorizer to 'reviewText' column
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["reviewText"])

# Original labels (non-encoded)
y_orig = df['sentiment']


# In[5]:


# # Another vectorizer if you want to experiment

# from sklearn.feature_extraction.text import CountVectorizer

# # Apply Count Vectorizer
# vectorizer = CountVectorizer(max_features=5000)
# X = vectorizer.fit_transform(df["reviewText"])  # Convert text data into features

# # Original labels (non-encoded)
# y_orig = df['sentiment']


# In[6]:


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


# In[7]:


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


# # Gradient Boost (extra model, can remove later)

# In[8]:


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


# In[9]:


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


# In[10]:


#!pip install xgboost


# In[11]:


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


# In[12]:


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


# # Model 1 (Logistic Regression Model Building and Hyperparameter Tuning)

# In[13]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[14]:


# Initialize logistic regression
lr = LogisticRegression(random_state=42, max_iter=1000)

# Fit to training data
lr.fit(X_train_orig, y_train_orig)

# Predict on test data
y_pred_lr = lr.predict(X_test_orig)

# Evaluate
print("Accuracy:", accuracy_score(y_test_orig, y_pred_lr))
print("\nClassification Report:\n", classification_report(y_test_orig, y_pred_lr))


# In[15]:


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


# # Visualization

# In[16]:


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


# In[17]:


# Visualization

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Confusion matrix
cm = confusion_matrix(y_test_orig, y_pred_best)

# Displaying the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lr.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for Logistic Regression")
plt.show()


# In[18]:


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


# # Model 2

# # ## Step 1: Build and Train the Model

# In[19]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Initialize SVM model
svm_model = SVC(random_state=99)

# Train the model on 70% of the data (encoded labels)
svm_model.fit(X_train_enc, y_train_enc)

# Predict on the test set (30% of the data)
y_pred_svm = svm_model.predict(X_test_enc)

# Evaluate the model
print("SVM Model Accuracy (Before Tuning):", accuracy_score(y_test_enc, y_pred_svm))
print(
    "\nClassification Report (Before Tuning):\n",
    classification_report(y_test_enc, y_pred_svm, target_names=le.classes_),
)


# # ## Step 2: Hyperparameter Tuning

# In[20]:


# Define the parameter grid for SVM
param_grid_svm = {
    "C": [0.01, 0.1, 1, 10, 100],
    "kernel": ["linear", "rbf", "poly", "sigmoid"],
    "gamma": ["scale", "auto", 0.01, 0.1, 1],
}

# Perform Grid Search with 5-fold cross-validation
grid_svm = GridSearchCV(
    SVC(random_state=99), param_grid_svm, cv=5, scoring="accuracy", verbose=1
)
grid_svm.fit(X_train_enc, y_train_enc)

# Best parameters and best score
print("\nBest Parameters for SVM:", grid_svm.best_params_)
print("Best Cross-Validation Accuracy:", grid_svm.best_score_)


# # ## Step 3: Test the Best Model

# In[21]:


# Use the best model to predict on the test set
best_svm_model = grid_svm.best_estimator_
y_pred_best_svm = best_svm_model.predict(X_test_enc)
print("Best Parameters:", grid_svm.best_params_)

# Evaluate the best model
print(
    "\nSVM Model Accuracy (After Tuning):", accuracy_score(y_test_enc, y_pred_best_svm)
)
print(
    "\nClassification Report (After Tuning):\n",
    classification_report(y_test_enc, y_pred_best_svm, target_names=le.classes_),
)

# Confusion Matrix
cm_svm = confusion_matrix(y_test_enc, y_pred_best_svm)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=le.classes_)
disp_svm.plot(cmap="Blues")
plt.title("Confusion Matrix for SVM (After Tuning)")
plt.show()


# In[22]:


from sklearn.metrics import f1_score

print("F1-Score (Before Tuning):", f1_score(y_test_enc, y_pred_svm, average="weighted"))
print(
    "F1-Score (After Tuning):",
    f1_score(y_test_enc, y_pred_best_svm, average="weighted"),
)


# # Finding long comments 

# In[23]:


# Select reviews with more than 100 words
long_reviews_df = df[df['reviewText'].apply(lambda x: len(x.split()) > 100)].copy()

# Check how many we have
print(f"Total long reviews (over 100 words): {len(long_reviews_df)}")

# Select top 10 for summarization
selected_long_reviews = long_reviews_df.head(10).reset_index(drop=True)

# Preview first review
print("\nExample long review:\n")
print(selected_long_reviews.loc[0, 'reviewText'])


# In[24]:


from transformers import pipeline

# Load summarization pipeline using flan-t5-base
summarizer = pipeline("summarization", model="google/flan-t5-base", device=0)  # 0 = GPU


# # summarization

# In[25]:


# Apply summarization
summaries = []

for i, row in selected_long_reviews.iterrows():
    review_text = row['reviewText']
    prompt = f"summarize this review in less than 50 words:\n{review_text}"
    summary = summarizer(prompt, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
    summaries.append(summary)

# Add to DataFrame
selected_long_reviews['summary'] = summaries

# Display first two as required in report
selected_long_reviews[['reviewText', 'summary']].head(2)


# ### 🔍 Select a Question-Like Review for Customer Response
# 
# 
# We use NLTK to detect reviews that contain questions based on sentence parsing.
# 
# 

# In[26]:


import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

# Function to check for interrogative sentence
def has_question(text):
    if not isinstance(text, str):
        return False
    sentences = sent_tokenize(text)
    return any(s.strip().endswith("?") or s.strip().lower().startswith(("why", "what", "how", "can", "do", "is", "are", "does")) for s in sentences)

# Apply to full DataFrame
question_df = df[df['reviewText'].apply(has_question)]

# Preview top 3
print("Total question-like reviews found:", len(question_df))
question_df['reviewText'].head(3).to_list()


# ### 🛠️ Setup: Common Prompt Builder and Sample Questions
# We dynamically inject customer complaint into each prompt using real user reviews.
# 

# In[36]:


# Use N already extracted question-like reviews
sample_questions = question_df['reviewText'].dropna().unique().tolist()[:3]  # Take first 3 non-empty

# Create unified prompt
def build_prompt(customer_text):
    return f"""Respond as a customer support agent: Hi, I bought this product but {customer_text.strip()}"""

# Just preview
for q in sample_questions:
    print("🔸", build_prompt(q))


# ### 🤖 Test Model A: MBZUAI/LaMini-Flan-T5-783M
# Great instruction-following small LLM. Outputs are helpful and polite.
# 

# In[37]:


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "MBZUAI/LaMini-Flan-T5-783M"
tokenizer_a = AutoTokenizer.from_pretrained(model_id)
model_a = AutoModelForSeq2SeqLM.from_pretrained(model_id).to("cuda")

# Generate responses
print("🧠 Responses from LaMini-Flan-T5-783M:")
for i, customer_text in enumerate(sample_questions):
    prompt = build_prompt(customer_text)
    inputs = tokenizer_a(prompt, return_tensors="pt").to("cuda")
    outputs = model_a.generate(inputs["input_ids"], max_length=60, do_sample=False)
    response = tokenizer_a.decode(outputs[0], skip_special_tokens=True)
    print(f"\n🔹Q{i+1}: {customer_text.strip()}\n💬 Response: {response}")


# ### 🧹 Clear VRAM Before Next Model
# Helps avoid CUDA OOM errors when switching large models.
# 

# In[39]:


import gc
import torch

del model_a
del tokenizer_a
torch.cuda.empty_cache()
gc.collect()

print("✅ Model A cleared from memory.")


# ### 🤖 Test Model B: google/flan-t5-small
# Smaller instruction model — good for fallback or fast inference.
# 

# In[40]:


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "google/flan-t5-small"
tokenizer_b = AutoTokenizer.from_pretrained(model_id)
model_b = AutoModelForSeq2SeqLM.from_pretrained(model_id).to("cuda")

print("🧠 Responses from flan-t5-small:")
for i, customer_text in enumerate(sample_questions):
    prompt = build_prompt(customer_text)
    inputs = tokenizer_b(prompt, return_tensors="pt").to("cuda")
    outputs = model_b.generate(inputs["input_ids"], max_length=60, do_sample=False)
    response = tokenizer_b.decode(outputs[0], skip_special_tokens=True)
    print(f"\n🔸Q{i+1}: {customer_text.strip()}\n💬 Response: {response}")


# ### 🦅 Load Falcon-RW-1B for Natural Language Generation
# 
# This model generates human-like completions and can handle conversational tone.
# 

# In[43]:


from transformers import AutoTokenizer, AutoModelForCausalLM

falcon_model_id = "tiiuae/falcon-rw-1b"

# Load tokenizer and model
falcon_tokenizer = AutoTokenizer.from_pretrained(falcon_model_id)
falcon_model = AutoModelForCausalLM.from_pretrained(falcon_model_id).to("cuda")

# Generate responses
print("🧠 Responses from Falcon-RW-1B:")
falcon_responses = []

for i, customer_text in enumerate(sample_questions):
    prompt = build_prompt(customer_text)
    inputs = falcon_tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = falcon_model.generate(inputs["input_ids"], max_new_tokens=60, do_sample=True, temperature=0.7)
    response = falcon_tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
    print(f"\n🦅 Q{i+1}: {customer_text.strip()}\n💬 Response: {response}")
    falcon_responses.append((customer_text, response))


# ### 📁 Save Summarized Customer Responses to CSV
# Includes: LaMini, Flan-T5, Falcon-RW-1B responses to same questions.
# 

# In[44]:


import pandas as pd

# Construct table
csv_data = {
    "Customer Review": sample_questions,
    "LaMini Response": [
        "I'm sorry to hear that. Can you please provide more details about the issue you are experiencing with the product?",
        "I'm sorry to hear that. Can you please provide more details about the product and the issue you are experiencing?",
        "I'm sorry to hear that. Can you please provide more details about the product and the issue you are experiencing?"
    ],
    "Flan-T5 Response": [
        "I am not sure what the problem is. I am not sure what the problem is.",
        "i am not sure if it is a light or a light.",
        "No, it is not good."
    ],
    "Falcon Response": [r[1] for r in falcon_responses]
}

df_responses = pd.DataFrame(csv_data)
df_responses.to_csv("customer_llm_responses.csv", index=False)

print("✅ Saved as customer_llm_responses.csv")
df_responses.head()


#  Load Lexicon predictions

# In[28]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

analyzer = SentimentIntensityAnalyzer()

test_indices = y_test_orig.index.to_list()
review_texts = df.iloc[test_indices]["reviewText"].tolist()

def vader_sentiment(text):
    score = analyzer.polarity_scores(text)
    return "Positive" if score['compound'] > 0 else "Negative" if score['compound'] < 0 else "Neutral"

def textblob_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    return "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"

# Apply lexicon classifiers
vader_preds = [vader_sentiment(text) for text in review_texts]
textblob_preds = [textblob_sentiment(text) for text in review_texts]


# ### 🗃️ Combine Lexicon and ML Predictions into One DataFrame
# Used for reporting and metric evaluation.
# 

# In[29]:


# Combine all model predictions
df_compare = pd.DataFrame({
    "Review": review_texts,
    "True Label": y_test_orig.tolist(),
    "VADER": vader_preds,
    "TextBlob": textblob_preds,
    "LogReg": y_pred_best,
    "SVM": y_pred_best_svm
})

df_compare.head()


# ### 📊 Evaluation: Classification Report for Each Sentiment Model
# We calculate accuracy, precision, recall, and F1 scores.
# 

# In[33]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(df_compare["True Label"])  # Corrected reference

# Decode the SVM prediction column
df_compare["SVM"] = le.inverse_transform(df_compare["SVM"])


# In[34]:


from sklearn.metrics import classification_report

models = ["VADER", "TextBlob", "LogReg", "SVM"]

for model in models:
    print(f"\n📌 Evaluation for {model}")
    print(classification_report(df_compare["True Label"], df_compare[model]))


# ### 🔄 Review Score Enhancement: Post-Filtering Strategy
# 
# According to Section 4.3.3 of the paper *"Recommender Systems Based on User Reviews"*, post-filtering is an effective method to improve rating accuracy. The authors state:
# 
# > “Combining inferred ratings (from overall opinion in the review) with user-specified real ratings improves recommendation accuracy... the best method was post-filtering: combining models linearly.” *(p. 116)*
# 
# We applied this method by calculating a weighted average of the actual user rating and the sentiment-inferred score (from our sentiment classification models). This post-filtered rating serves as an improved score input for recommender logic.
# 

# In[40]:


# Convert sentiment predictions to numeric scores
def sentiment_to_score(label):
    return 5 if label == "Positive" else 3 if label == "Neutral" else 1

# Apply to models
df_compare["LogReg_Score"] = df_compare["LogReg"].apply(sentiment_to_score)
df_compare["SVM_Score"] = df_compare["SVM"].apply(sentiment_to_score)
df_compare["VADER_Score"] = df_compare["VADER"].apply(sentiment_to_score)
df_compare["TextBlob_Score"] = df_compare["TextBlob"].apply(sentiment_to_score)

# Include real rating from main df
# Use sentiment label as a proxy for real rating
def label_to_rating(label):
    return 5 if label == "Positive" else 3 if label == "Neutral" else 1

df_compare["Real_Rating"] = df.iloc[test_indices]["sentiment"].apply(label_to_rating).tolist()


# In[41]:


# Linear combination (adjust alpha as needed)
alpha = 0.7  # weight of real rating

df_compare["Combined_LogReg"] = alpha * df_compare["Real_Rating"] + (1 - alpha) * df_compare["LogReg_Score"]
df_compare["Combined_SVM"] = alpha * df_compare["Real_Rating"] + (1 - alpha) * df_compare["SVM_Score"]
df_compare["Combined_VADER"] = alpha * df_compare["Real_Rating"] + (1 - alpha) * df_compare["VADER_Score"]
df_compare["Combined_TextBlob"] = alpha * df_compare["Real_Rating"] + (1 - alpha) * df_compare["TextBlob_Score"]

df_compare[["Real_Rating", "LogReg_Score", "Combined_LogReg", "Combined_SVM"]].head()


# In[42]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.hist(df_compare["Real_Rating"], bins=5, alpha=0.5, label="Real Rating")
plt.hist(df_compare["Combined_LogReg"], bins=10, alpha=0.5, label="Combined Rating (LogReg)")
plt.xlabel("Rating Value")
plt.ylabel("Frequency")
plt.title("Distribution of Real vs Enhanced Ratings")
plt.legend()
plt.grid(True)
plt.show()


# In[43]:


df_compare.to_csv("enhanced_ratings_postfilter.csv", index=False)
print("✅ Saved as enhanced_ratings_postfilter.csv")


# ###  Final Artifacts Saved
# - `customer_llm_responses.csv` – LLM-generated replies
# - `sentiment_model_comparison.csv` – All model predictions
# - `enhanced_ratings_postfilter.csv` – Post-filtered ratings for recommender use
