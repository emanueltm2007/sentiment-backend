from flask import Flask, request, jsonify
from flask_cors import CORS
import re
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Descargar recursos
nltk.download('vader_lexicon', quiet=True)

# Inicializar analizador
sia = SentimentIntensityAnalyzer()

# Ajustar para español (palabras comunes)
spanish_words = {
    'bueno': 2.0, 'malo': -2.0, 'excelente': 3.0, 'terrible': -3.0,
    'genial': 2.5, 'horrible': -2.5, 'amor': 2.0, 'odio': -2.0,
    'feliz': 2.0, 'triste': -2.0, 'bien': 1.5, 'mal': -1.5
}
sia.lexicon.update(spanish_words)

def limpiar_texto(texto):
    if not texto or len(texto.strip()) == 0:
        return ""
    texto = texto.lower()
    texto = re.sub(r'http\S+|www\S+', '', texto)
    texto = re.sub(r'@\w+', '', texto)
    texto = re.sub(r'#', '', texto)
    texto = re.sub(r'[^a-záéíóúñ\s]', '', texto)
    return ' '.join(texto.split())

def analizar_sentimiento(texto):
    if not texto or len(texto.strip()) == 0:
        return {"sentimiento": "Neutro", "confianza": 0.0, "texto_limpio": "", "detalle": []}
    texto_limpio = limpiar_texto(texto)
    scores = sia.polarity_scores(texto_limpio)
    compound = scores['compound']
    if compound >= 0.05:
        sentimiento = "Positivo"
    elif compound <= -0.05:
        sentimiento = "Negativo"
    else:
        sentimiento = "Neutro"
    confianza = abs(compound) * 100
    return {
        "sentimiento": sentimiento,
        "confianza": round(confianza, 2),
        "texto_limpio": texto_limpio,
        "detalle": scores
    }

def analizar_comentarios_masivo(lista):
    resultados = []
    contadores = {"Positivo": 0, "Negativo": 0, "Neutro": 0}
    for texto in lista:
        res = analizar_sentimiento(texto)
        resultados.append(res)
        contadores[res["sentimiento"]] += 1
    return {
        "estadisticas": {
            "contadores": contadores,
            "total": len(lista),
            "top_palabras": []
        }
    }

app = Flask(__name__)
CORS(app)

@app.route('/analizar', methods=['POST'])
def analizar():
    data = request.get_json()
    texto = data.get("texto", "")
    resultado = analizar_sentimiento(texto)
    return jsonify(resultado)

@app.route('/analizar-multiple', methods=['POST'])
def analizar_multiple():
    data = request.get_json()
    comentarios = data.get("comentarios", [])
    resultado = analizar_comentarios_masivo(comentarios)
    return jsonify(resultado)

if __name__ == '__main__':
    app.run(debug=True)
