import os
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

app.config['DEBUG'] = os.environ.get('DEBUG', False)


@app.route('/health_check', methods=['GET'])
def health_check():
    return 'Belief inference service is up and running.'


@app.route('/predict', methods=['POST'])
def get_prediction():
    users_batch = request.json['users_batch']
    zero_shot_enabled = request.json['zero_shot_enabled']

    headers = {
        'Content-Type': 'application/json'
    }

    r = requests.post(
        f'http://localhost:5001/predict',
        headers=headers,
        json={'users_batch':users_batch, 'zero_shot_enabled':zero_shot_enabled}
    )
    print(r.content)
    return jsonify({'output': r.json()})


app.run(host='0.0.0.0')
