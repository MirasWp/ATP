from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import face_recognition
import joblib
from base64 import b64decode

app = Flask(__name__)
clf = joblib.load("modelo_reconhecimento.pkl")

def process_image(frame_bytes):
    nparr = np.frombuffer(frame_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    localizacoes = face_recognition.face_locations(rgb)
    codificacoes = face_recognition.face_encodings(rgb, localizacoes)

    resultados = []
    for encoding, loc in zip(codificacoes, localizacoes):
        prob = clf.predict_proba([encoding]).max()
        nome = clf.predict([encoding])[0] if prob > 0.7 else "Desconhecido"
        resultados.append({"nome": nome, "probabilidade": round(float(prob), 2)})

    return resultados

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reconhecer', methods=['POST'])
def reconhecer():
    data_url = request.json['image']
    header, encoded = data_url.split(",", 1)
    img_bytes = b64decode(encoded)
    resultados = process_image(img_bytes)
    return jsonify(resultados)

if __name__ == "__main__":
    app.run(debug=True)