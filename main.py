#!/usr/bin/env python
# coding: utf-8

# In[19]:


import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.decomposition import TruncatedSVD

# Ensure NLTK resources are available
import nltk
nltk.download('stopwords')

# Improved text preprocessing
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    # Tokenize by whitespace
    tokens = text.split()
    # Remove stopwords and apply stemming
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def load_data(file_path, has_labels=True, preprocess_fn=None):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read().strip().split('#EOF')
    labels, texts = [], []
    for review in data:
        if not review.strip(): continue  # Skip any empty reviews
        if has_labels:
            parts = review.split()
            labels.append(int(parts[0]))  # Convert the first part to an int (label)
            text = ' '.join(parts[1:])  # The rest is the review text
            if preprocess_fn:
                text = preprocess_fn(text)
            texts.append(text)
        else:
            text = review.strip()  # The entire review is the text for test data
            if preprocess_fn:
                text = preprocess_fn(text)
            texts.append(text)
            labels.append(None)  # No labels for test data
    if has_labels:
        return labels, texts
    else:
        return texts  # Only return texts if there are no labels

# Use the updated function with preprocessing to load training and test data
train_labels, train_texts = load_data('train_new.txt', has_labels=True, preprocess_fn=preprocess_text)
test_texts = load_data('test_new.txt', has_labels=False, preprocess_fn=preprocess_text)  # No labels in test data

# Adjust the TfidfVectorizer with optimal settings
vectorizer = TfidfVectorizer(max_features=8700, ngram_range=(1, 2), min_df=2, max_df=0.9, norm='l2')
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# Optional: Apply dimensionality reduction
# svd = TruncatedSVD(n_components=500)  # Adjust n_components based on the dataset size and feature importance
# X_train_reduced = svd.fit_transform(X_train)
# X_test_reduced = svd.transform(X_test)


# In[20]:


from sklearn.metrics.pairwise import cosine_similarity  # For more efficient batch processing
from sklearn.model_selection import train_test_split
import numpy as np

def k_nearest_neighbors(X_train, y_train, X_test, k):
    predictions = []
    for test_vector in X_test:
        # No need to convert to dense format since we're using dense matrices after SVD
        test_vector_dense = test_vector.reshape(1, -1)  # Ensure it's 2D for cosine_similarity

        # Calculate cosine similarity for the batch
        distances = 1 - cosine_similarity(X_train, test_vector_dense)

        # Find the indices of the k smallest distances
        nearest_neighbors_indices = np.argsort(distances, axis=0)[:k].flatten()

        # Get the labels of the nearest neighbors
        nearest_neighbors_labels = [y_train[i] for i in nearest_neighbors_indices]

        # Predict the label based on majority vote
        prediction = max(set(nearest_neighbors_labels), key=nearest_neighbors_labels.count)
        predictions.append(prediction)
    return predictions

# def cross_validate(X, y, k_values):
#     best_k = k_values[0]
#     best_accuracy = 0
#     # Splitting the data for cross-validation
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
#     for k in k_values:
#         predictions = k_nearest_neighbors(X_train, y_train, X_val, k=k)
#         accuracy = np.mean([pred == true for pred, true in zip(predictions, y_val)])
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_k = k
#     print(f"Best k: {best_k} with accuracy: {best_accuracy}")
#     return best_k


from sklearn.model_selection import KFold
import numpy as np

def cross_validate(X, y, k_values, n_splits=5):
    best_k = k_values[0]
    best_accuracy = 0
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Ensure X and y are numpy arrays to support fancy indexing
    X = np.array(X)
    y = np.array(y)
    
    for k in k_values:
        accuracies = []
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            predictions = k_nearest_neighbors(X_train, y_train, X_val, k=k)
            accuracy = np.mean([pred == true for pred, true in zip(predictions, y_val)])
            accuracies.append(accuracy)
        
        avg_accuracy = np.mean(accuracies)
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_k = k
            
    print(f"Best k: {best_k} with accuracy: {best_accuracy}")
    return best_k
# # Use the dimensionality-reduced versions of your datasets for k-NN
# X_train_reduced, X_test_reduced = # Assuming you have these from the previous TruncatedSVD step
# train_labels = # Assuming you have your labels ready




# In[21]:


# k_values = range(70, 111)  # Adjust the range of k values as needed
k_values = [171, 173, 177, 167, 139, 181, 169, 175, 191, 185]

best_k = cross_validate(X_train_reduced, train_labels, k_values)



# In[22]:


final_predictions = k_nearest_neighbors(X_train, train_labels, X_test, k=best_k)


# In[23]:


with open('prediction_TEST-31.txt', 'w') as f:
    for pred in final_predictions:
        f.write(f"{pred}\n")


# In[ ]:




