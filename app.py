
from flask import Flask, request, jsonify
import joblib
from score import score

app = Flask(__name__)

model = joblib.load('xgboost_model.pkl')

@app.route('/score', methods=['POST'])
def score_endpoint():
    data = request.get_json()
    text = data.get('text', '')
    threshold = data.get('threshold', 0.5)  # Default threshold
    prediction, propensity = score(text, model, threshold)
    return jsonify({'prediction': int(prediction), 'propensity': float(propensity)})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5050)
