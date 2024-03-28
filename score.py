# import joblib
# import sklearn
# import numpy as np
# import pandas as pd
# from typing import Tuple
# import numpy as np
# from sklearn.pipeline import Pipeline

# def score(text, model_path, threshold):
#     # Load the trained model pipeline
#     model = joblib.load('xgboost_model.joblib')
#     vectorizer = joblib.load('tfidf_vectorizer.joblib')
    
#     # Access the TF-IDF Vectorizer directly by its named step in the pipeline
#     text_vectorized = vectorizer.transform([text])
    
#     # Get the classifier step
#     classifier = model
    
#     # Get the propensity score using the classifier step directly
#     propensity = classifier.predict_proba(text_vectorized)[0][1]
    
#     # Generate the prediction based on the threshold
#     prediction = propensity > threshold
    
#     return bool(prediction), propensity
import joblib

# Load the trained model and vectorizer (assuming they are saved in the same directory)
model = joblib.load('xgboost_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def score(text, model, threshold):
    # Transform the input text using the loaded vectorizer
    text_vector = vectorizer.transform([text])
    propensity = model.predict_proba(text_vector)[0, 1]
    prediction = propensity > threshold
    return prediction, propensity
