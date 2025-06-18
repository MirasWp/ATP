from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import face_recognition
import joblib
from base64 import b64decode
import mysql.connector
import json
from datetime import date, datetime

app = Flask(__name__)
(clf, scaler) = joblib.load("modelo_reconhecimento.pkl")

def conectar_bd():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="ATP3"
    )

def process_image(frame_bytes):
    nparr = np.frombuffer(frame_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    localizacoes = face_recognition.face_locations(rgb)
    codificacoes = face_recognition.face_encodings(rgb, localizacoes)

    resultados = []
    conexao = conectar_bd()
    cursor = conexao.cursor()

    for encoding in codificacoes:
        encoding_scaled = scaler.transform([encoding])
        prob = clf.predict_proba(encoding_scaled).max()
        nome = clf.predict(encoding_scaled)[0] if prob > 0.7 else "Desconhecido"

        if nome != "Desconhecido":
            # Buscar id_aluno
            cursor.execute("SELECT id_aluno FROM Alunos WHERE nome = %s", (nome,))
            resultado = cursor.fetchone()
            if resultado:
                id_aluno = resultado[0]

                # Verificar se já tem presença hoje
                hoje = date.today()
                cursor.execute("""
                    SELECT COUNT(*) FROM PRESENCA
                    WHERE id_aluno = %s AND DATE(NOW()) = %s
                """, (id_aluno, hoje))
                (presente,) = cursor.fetchone()

                if presente == 0:
                    cursor.execute("""
                        INSERT INTO PRESENCA (presenca, id_aluno, id_professor, id_turma, id_disciplina)
                        VALUES (True, %s, 1, 1, 1)
                    """, (id_aluno,))
                    conexao.commit()
                    print(f"[PRESENÇA] Presença registrada para {nome}")

        resultados.append({"nome": nome, "probabilidade": round(float(prob), 2)})

    cursor.close()
    conexao.close()
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
