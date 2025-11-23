from flask import Flask, render_template, request, jsonify
import os
import model

app = Flask(__name__)

# Home page: show form and sample users
@app.route('/', methods=['GET'])
def home():
    try:
        sample_users = model.recommendation_matrix.index.tolist()[:20]
    except Exception:
        sample_users = []
    return render_template('index.html', sample_users=sample_users)

# Recommendation endpoint (form POST)
@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.form.get('user_id', '').strip()
    top_n = int(request.form.get('top_n', 5))
    try:
        sample_users = model.recommendation_matrix.index.tolist()[:20]
    except Exception:
        sample_users = []

    if not user_id:
        return render_template('index.html', error='Please provide a user id.', sample_users=sample_users)

    result = model.recommend_for_user(user_id, top_n=top_n)
    if 'error' in result:
        return render_template('index.html', error=result['error'], sample_users=sample_users)

    return render_template('index.html', recommendations=result['recommendations'], user_id=user_id, sample_users=sample_users)

# Sentiment prediction API (AJAX)
@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    review_text = request.form.get('review_text', '')
    if not review_text:
        return jsonify({'error': 'No text provided'}), 400
    res = model.predict_sentiment_for_text(review_text)
    return jsonify(res)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
