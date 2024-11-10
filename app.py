from flask import Flask, request, jsonify
import joblib
import numpy as np
import re
from urllib.parse import urlparse

app = Flask(__name__)

# Cargar el modelo XGBoost
model = joblib.load('xgb_model.pkl')

# Función para extraer características de la URL
def extract_features(url):
    features = {
        'usa_IP': int(bool(re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', url))),
        'anormal_URL': int('@' in url),
        'count_punto': url.count('.'),
        'count_www': url.count('www'),
        'count_@': url.count('@'),
        'count_dir': url.count('/'),
        'count_embed': url.count('embed'),
        'short_url': int(len(url) < 15),
        'count_https': url.count('https'),
        'count_http': url.count('http'),
        'count_%': url.count('%'),
        'count_?': url.count('?'),
        'count_-': url.count('-'),
        'count_=': url.count('='),
        'url_length': len(url),
        'sus_url': int(any(keyword in url for keyword in ['login', 'secure', 'account', 'update', 'verify'])),
        'fd_length': len(url.split('/')[2].split('.')[0]) if '//' in url else 0,
        'count_num': sum(c.isdigit() for c in url),
        'tld_length': len(url.split('.')[-1]) if '.' in url else 0,
        'hostname_length': len(url.split('/')[2]) if '//' in url else len(url),
        'count_letras': sum(c.isalpha() for c in url)
    }
    return np.array(list(features.values())).reshape(1, -1)

# Ruta para la predicción
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url = data.get('url')
    
    # Validar si la URL fue enviada
    if not url:
        return jsonify({"error": "URL no proporcionada"}), 400
    
    # Extraer características y predecir
    features = extract_features(url)
    prediction = model.predict(features)
    probabilidad = model.predict_proba(features)[:, 1]
    resultado = 'Peligrosa' if prediction[0] == 1 else 'Benigna'

    return jsonify({"resultado": resultado, "probabilidad": probabilidad[0]})

if __name__ == '__main__':
    app.run(debug=True)
