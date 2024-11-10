from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Cargar el modelo entrenado una sola vez, que incluye el preprocesamiento
modelo = joblib.load('xgb_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Realizar la predicción usando el modelo que incluye preprocesamiento
    resultado, probabilidad = modelo.predecir_url3(url)  # Asegúrate de que 'predecir_url3' sea accesible desde el modelo cargado
    
    return jsonify({
        "url": url,
        "resultado": resultado,
        "probabilidad_peligro": probabilidad
    })

if __name__ == '__main__':
    app.run(debug=True)
