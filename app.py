from flask import Flask, request, jsonify
from PhishBusterAlgoritmo import predecir_url3  # Asegúrate de tener el archivo del modelo como modelo_predictivo.py

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Obtener el resultado de predicción
    resultado, probabilidad = predecir_url3(url)
    return jsonify({
        "url": url,
        "resultado": resultado,
        "probabilidad_peligro": probabilidad
    })

if __name__ == '__main__':
    app.run(debug=True)
