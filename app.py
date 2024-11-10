from flask import Flask, request, jsonify
import joblib
import re
import numpy as np
from urllib.parse import urlparse
from tld import get_tld

app = Flask(__name__)

# Cargar el modelo entrenado
modelo = joblib.load('xgb_model.pkl')

# Funciones de preprocesamiento
def Dominio_IP2(url):
    match = re.search(r'(([01]?\d\d?|2[0-4]\d|25[0-5])\.){3}([01]?\d\d?|2[0-4]\d|25[0-5])|([a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)
    return 1 if match else 0

def anormal_URL2(url):
    hostname = urlparse(url).hostname
    return 1 if re.search(str(hostname), url) else 0

def count_punto2(url):
    return url.count('.')

def count_www2(url):
    return url.count('www')

def count_a2(url):
    return url.count('@')

def count_dir2(url):
    return urlparse(url).path.count('/')

def count_embed2(url):
    return urlparse(url).path.count('//')

def servicio_acortamiento2(url):
    match = re.search(r'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs', url)
    return 1 if match else 0

def count_https2(url):
    return url.count('https')

def count_http2(url):
    return url.count('http')

def count_per2(url):
    return url.count('%')

def count_ques2(url):
    return url.count('?')

def count_hyphen2(url):
    return url.count('-')

def count_equal2(url):
    return url.count('=')

def tamaño_URL2(url):
    return len(str(url))

def tamaño_hostname2(url):
    return len(urlparse(url).netloc)

def suspicious_words2(url):
    match = re.search(r'PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr', url)
    return 1 if match else 0

def count_num2(url):
    return sum(1 for char in url if char.isnumeric())

def letter_count2(url):
    return sum(1 for char in url if char.isalpha())

def fd_length2(url):
    urlpath = urlparse(url).path
    return len(urlpath.split('/')[1]) if '/' in urlpath else 0

def tld_length2(url):
    tld = get_tld(url, fail_silently=True)
    return len(tld) if tld else -1

def extract_features(url):
    features = [
        Dominio_IP2(url),
        anormal_URL2(url),
        count_punto2(url),
        count_www2(url),
        count_a2(url),
        count_dir2(url),
        count_embed2(url),
        servicio_acortamiento2(url),
        count_https2(url),
        count_http2(url),
        count_per2(url),
        count_ques2(url),
        count_hyphen2(url),
        count_equal2(url),
        tamaño_URL2(url),
        tamaño_hostname2(url),
        suspicious_words2(url),
        count_num2(url),
        letter_count2(url),
        fd_length2(url),
        tld_length2(url)
    ]
    return np.array(features).reshape(1, -1)

# Endpoint de predicción
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Extraer características de la URL
    features = extract_features(url)
    
    # Realizar la predicción
    prediccion = modelo.predict(features)
    probabilidad = modelo.predict_proba(features)[:, 1]

    # Interpretar resultado
    resultado = 'Peligrosa' if probabilidad[0] >= 0.80 else 'Benigna'
    
    return jsonify({
        "url": url,
        "resultado": resultado,
        "probabilidad_peligro": probabilidad[0]
    })

if __name__ == '__main__':
    app.run(debug=True)
