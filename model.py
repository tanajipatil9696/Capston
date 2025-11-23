import os
import re
import pickle
import gzip
import numpy as np
import pandas as pd

# Minimal text preprocessing (aligns with notebook)
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except Exception:
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)


# Paths to serialized artifacts (support both .pkl and .pkl.gz)
BASE_DIR = os.path.dirname(__file__)
SENTIMENT_MODEL_PATH = os.path.join(BASE_DIR, 'best_sentiment_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')
REC_MATRIX_PATH = os.path.join(BASE_DIR, 'recommendation_matrix.pkl')
PRODUCT_MAP_PATH = os.path.join(BASE_DIR, 'product_review_mapping.pkl')


def _load_pickle(path):
    """Load pickle from .pkl or .pkl.gz file. Try both formats."""
    # Try uncompressed first
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    # Try compressed
    gz_path = path + '.gz'
    if os.path.exists(gz_path):
        with gzip.open(gz_path, 'rb') as f:
            return pickle.load(f)
    
    raise FileNotFoundError(f'Required file not found: {path} or {gz_path}')


# Load artifacts (fail-fast if missing)
print('\nLoading ML models and artifacts...')
best_model = None
tfidf_vectorizer = None
recommendation_matrix = None
product_review_mapping = None

try:
    best_model = _load_pickle(SENTIMENT_MODEL_PATH)
    tfidf_vectorizer = _load_pickle(VECTORIZER_PATH)
    recommendation_matrix = _load_pickle(REC_MATRIX_PATH)
    product_review_mapping = _load_pickle(PRODUCT_MAP_PATH)
    print('✓ All models loaded successfully\n')
except Exception as e:
    print(f'✗ Error loading models: {e}\n')
    raise

# Normalize recommendation_matrix to DataFrame with string index
if not isinstance(recommendation_matrix, pd.DataFrame):
    recommendation_matrix = pd.DataFrame(recommendation_matrix)
recommendation_matrix.index = recommendation_matrix.index.astype(str)
recommendation_matrix.columns = recommendation_matrix.columns.astype(str)


def predict_sentiment_for_text(raw_text: str) -> dict:
    """Return predicted label and probability for a single text."""
    proc = preprocess_text(raw_text)
    X = tfidf_vectorizer.transform([proc])
    proba = None
    pred = None
    try:
        proba = best_model.predict_proba(X)[0]
        pred = int(best_model.predict(X)[0])
    except Exception:
        # Some models may not implement predict_proba
        pred = int(best_model.predict(X)[0])
        proba = [None, None]
    return {'prediction': pred, 'probability_positive': float(proba[1]) if proba is not None and proba[1] is not None else None}


def recommend_for_user(user_id: str, top_n: int = 20):
    """
    Return top N recommended products for user_id using the stored recommendation matrix.
    Then compute sentiment-based stats for each product from product_review_mapping.
    """
    user_key = str(user_id)
    if user_key not in recommendation_matrix.index:
        return {'error': f'user_id "{user_id}" not found in recommendation matrix.'}

    user_scores = recommendation_matrix.loc[user_key].dropna()
    user_scores = user_scores[user_scores > 0].sort_values(ascending=False)
    if user_scores.empty:
        return {'error': 'No recommendations available for this user.'}

    top_products = list(user_scores.head(top_n).index)
    results = []

    for product in top_products:
        reviews_df = product_review_mapping.get(product, pd.DataFrame())
        if reviews_df.empty:
            results.append({
                'product': product,
                'positive_pct': 0.0,
                'avg_rating': None,
                'review_count': 0
            })
            continue

        # Use processed_text if available
        if 'processed_text' in reviews_df.columns:
            texts = reviews_df['processed_text'].astype(str).tolist()
        else:
            combined = (reviews_df.get('reviews_title', '').fillna('') + ' ' + reviews_df.get('reviews_text', '').fillna('')).astype(str)
            texts = [preprocess_text(t) for t in combined.tolist()]

        if len(texts) == 0:
            positive_pct = 0.0
        else:
            X = tfidf_vectorizer.transform(texts)
            preds = best_model.predict(X)
            positive_pct = float((preds == 1).sum() / len(preds) * 100)

        try:
            avg_rating = float(reviews_df['reviews_rating'].dropna().astype(float).mean())
        except Exception:
            avg_rating = None

        results.append({
            'product': product,
            'positive_pct': round(positive_pct, 2),
            'avg_rating': round(avg_rating, 2) if avg_rating is not None else None,
            'review_count': len(texts)
        })

    return {'user_id': user_id, 'recommendations': results}
