from flask import Flask, request, jsonify
from flask_cors import CORS
import re
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Descargar recursos solo si es necesario
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# Ajuste para español
spanish_words = {
    'bueno': 2.0, 'malo': -2.0, 'excelente': 3.0, 'terrible': -3.0,
    'genial': 2.5, 'horrible': -2.5, 'amor': 2.0, 'odio': -2.0
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
    if not texto.strip():
        return {"sentimiento": "Neutro", "confianza": 0.0}
    scores = sia.polarity_scores(limpiar_texto(texto))
    compound = scores['compound']
    sentimiento = "Positivo" if compound >= 0.05 else "Negativo" if compound <= -0.05 else "Neutro"
    return {"sentimiento": sentimiento, "confianza": round(abs(compound)*100, 2)}

def analizar_comentarios_masivo(comentarios):
    contadores = {"Positivo": 0, "Negativo": 0, "Neutro": 0}
    for texto in comentarios:
        res = analizar_sentimiento(texto)
        contadores[res["sentimiento"]] += 1
    return {"estadisticas": {"contadores": contadores, "total": len(comentarios)}}

# Crear app Flask
app = Flask(__name__)
CORS(app)

@app.route('/analizar', methods=['POST'])
def analizar():
    data = request.get_json()
    resultado = analizar_sentimiento(data.get("texto", ""))
    return jsonify(resultado)

@app.route('/analizar-multiple', methods=['POST'])
def analizar_multiple():
    data = request.get_json()
    resultado = analizar_comentarios_masivo(data.get("comentarios", []))
    return jsonify(resultado)

# Handler para Vercel (obligatorio)
def handler(request):
    from werkzeug.serving import WSGIRequestHandler
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    return app
