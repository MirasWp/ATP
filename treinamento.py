from sklearn import svm
from sklearn.preprocessing import StandardScaler
from os import listdir
from os.path import isdir, join
import face_recognition
import joblib

DIR_FACES = r"C:\Users\lucas\OneDrive\Desktop\ATP\Faces"
CAMINHO_MODELO = "modelo_reconhecimento.pkl"

def treinar_reconhecedor():
    encodings, nomes = [], []

    for pessoa in listdir(DIR_FACES):
        caminho_pessoa = join(DIR_FACES, pessoa)
        if not isdir(caminho_pessoa): continue

        for img_nome in listdir(caminho_pessoa):
            img_path = join(caminho_pessoa, img_nome)
            imagem = face_recognition.load_image_file(img_path)
            codificacoes = face_recognition.face_encodings(imagem)
            if codificacoes:
                encodings.append(codificacoes[0])
                nomes.append(pessoa)

    if not nomes:
        print("Nenhuma imagem válida para treino.")
        return

    scaler = StandardScaler()
    encodings = scaler.fit_transform(encodings)

    clf = svm.SVC(gamma='scale', probability=True)
    clf.fit(encodings, nomes)
    joblib.dump(clf, CAMINHO_MODELO)
    print("Modelo salvo em:", CAMINHO_MODELO)

if __name__ == "__main__":
    treinar_reconhecedor()