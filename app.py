from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import nltk
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Descargar recursos
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Cargar modelo
clf = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    tokenizer="nlptown/bert-base-multilingual-uncased-sentiment",
    top_k=None
)
)

stop_words = set(stopwords.words('spanish'))

def limpiar_texto(texto):
    if not texto or len(texto.strip()) == 0:
        return ""
    texto = texto.lower()
    texto = re.sub(r'http\S+|www\S+', '', texto)
    texto = re.sub(r'@\w+', '', texto)
    texto = re.sub(r'#', '', texto)
    texto = re.sub(r'[^a-záéíóúñ\s]', '', texto)
    return ' '.join(texto.split())

def extraer_palabras_clave(texto):
    palabras = word_tokenize(texto.lower())
    return [p for p in palabras if p not in stop_words and len(p) > 3]

def analizar_sentimiento(texto):
    if not texto or len(texto.strip()) == 0:
        return {"sentimiento": "Neutro", "confianza": 0.0, "texto_limpio": "", "detalle": []}
    texto_limpio = limpiar_texto(texto)
    if not texto_limpio:
        return {"sentimiento": "Neutro", "confianza": 0.0, "texto_limpio": "", "detalle": []}
    predicciones = clf(texto_limpio)[0]
    mejor = max(predicciones, key=lambda x: x["score"])
    etiquetas_es = {"positive": "Positivo", "negative": "Negativo", "neutral": "Neutro"}
    return {
        "sentimiento": etiquetas_es.get(mejor["label"], "Neutro"),
        "confianza": round(mejor["score"] * 100, 2),
        "texto_limpio": texto_limpio,
        "detalle": predicciones
    }

def analizar_comentarios_masivo(lista):
    resultados = []
    contadores = {"Positivo": 0, "Negativo": 0, "Neutro": 0}
    todas_palabras = []
    for texto in lista:
        res = analizar_sentimiento(texto)
        resultados.append(res)
        contadores[res["sentimiento"]] += 1
        todas_palabras.extend(extraer_palabras_clave(res["texto_limpio"]))
    from collections import Counter
    top_palabras = [{"palabra": p, "frecuencia": f} for p, f in Counter(todas_palabras).most_common(10)]
    return {
        "estadisticas": {
            "contadores": contadores,
            "total": len(lista),
            "top_palabras": top_palabras
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
