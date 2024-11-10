from flask import Flask, request, jsonify
import numpy as np
import joblib
import re

app = Flask(__name__)

# Cargar el modelo XGBoost previamente entrenado
xgb_model = joblib.load('xgb_model.pkl')  # Asegúrate de tener 'xgb_model.pkl' disponible

# Función para extraer características de la URL
def extract_features(url):
    features = {}
    features['usa_IP'] = int(bool(re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', url)))
    features['anormal_URL'] = int('@' in url)
    features['count_punto'] = url.count('.')
    features['count_www'] = url.count('www')
    features['count_@'] = url.count('@')
    features['count_dir'] = url.count('/')
    features['count_embed'] = url.count('embed')
    features['short_url'] = int(len(url) < 15)
    features['count_https'] = url.count('https')
    features['count_http'] = url.count('http')
    features['count_%'] = url.count('%')
    features['count_?'] = url.count('?')
    features['count_-'] = url.count('-')
    features['count_='] = url.count('=')
    features['url_length'] = len(url)
    features['sus_url'] = int(any(keyword in url for keyword in ['login', 'secure', 'account', 'update', 'verify']))
    features['fd_length'] = len(url.split('/')[2].split('.')[0]) if '//' in url else 0
    features['count_num'] = sum(c.isdigit() for c in url)
    features['tld_length'] = len(url.split('.')[-1]) if '.' in url else 0
    features['hostname_length'] = len(url.split('/')[2]) if '//' in url else len(url)
    features['count_letras'] = sum(c.isalpha() for c in url)
    return list(features.values())

# Ruta para predecir si una URL es peligrosa o no usando solo XGBoost
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url = data.get('input', "")

    # Extraer características de la URL
    features = np.array(extract_features(url)).reshape(1, -1)

    # Realizar predicción con XGBoost
    probabilidad_xgb = xgb_model.predict_proba(features)[:, 1][0]
    resultado_final = "Peligrosa" if probabilidad_xgb >= 0.80 else "Benigna"

    return jsonify({
        'resultado': resultado_final,
        'probabilidad_xgb': probabilidad_xgb
    })

if __name__ == '__main__':
    app.run(host='192.168.1.38', port=5000, debug=True)
