from flask import Flask, jsonify
import pandas as pd
import joblib
from sqlalchemy import create_engine
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)
# Configura tu conexión a la base de datos
engine = create_engine('mysql+pymysql://marlenhe_Marlen:Marlen1602@mx32.hostgator.mx:3306/marlenhe_smokeygrill')


@app.route('/api/prediccion-carne', methods=['GET'])
def predecir_carne():
    try:
        # 1. Consultar las últimas 4 semanas
        query = """
        SELECT hamburguesas, tacos, bolillos, burritos, gringas, baguettes
        FROM ventas_semanales
        ORDER BY end_date DESC
        LIMIT 4
        """
        ultimas_4 = pd.read_sql(query, engine)
        ultimas_4 = ultimas_4[::-1]  # Orden cronológico

        # 2. Calcular promedio
        features = ['hamburguesas', 'tacos', 'bolillos', 'burritos', 'gringas', 'baguettes']
        promedios = ultimas_4[features].mean().values.reshape(1, -1)
        df_promedios = pd.DataFrame(promedios, columns=features)

        # 3. Cargar modelo
        modelo = joblib.load('modelo_consumo_carne.pkl')

        # 4. Predecir
        prediccion = modelo.predict(df_promedios)[0]

        return jsonify({
            "mensaje": "Predicción realizada con éxito",
            "carne_estimacion_kg": round(prediccion, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Si ejecutas directamente
if __name__ == '__main__':
    app.run(debug=True)
