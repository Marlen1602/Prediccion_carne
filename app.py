from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Cargar el modelo entrenado
modelo = joblib.load('modelo_consumo_carne.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # Si tienes HTML, si no puedes quitar esta línea

@app.route('/api/prediccion-carne', methods=['POST'])
def predecir_carne():
    try:
        data = request.get_json()
        campos = ['hamburguesas', 'tacos', 'bolillos', 'burritos', 'gringas', 'baguettes']
        
        if not all(c in data for c in campos):
            return jsonify({'error': 'Faltan campos'}), 400

        entrada = np.array([[data[c] for c in campos]])
        prediccion = modelo.predict(entrada)[0]

        return jsonify({'prediccion_carne_kg': round(prediccion, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
