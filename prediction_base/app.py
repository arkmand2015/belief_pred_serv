import os
from flask import Flask, request, jsonify
from predict import BeliefClassifier

app = Flask(__name__)

predictor = BeliefClassifier()
loaded_models = predictor.load_models()

app.config['DEBUG'] = os.environ.get('DEBUG', False)


@app.route('/health_check', methods=['GET'])
def health_check():
    return 'Belief prediction service is up and running.'


@app.route('/predict', methods=['POST'])
def get_prediction():
    users_batch = request.json['users_batch']
    zero_shot_enabled = request.json['zero_shot_enabled']
    prediction = predictor.predict(users_batch, zero_shot_enabled)
    return jsonify({'prediction': prediction})


app.run(host='0.0.0.0', port=5001)
