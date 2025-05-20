from PIL import Image
from numpy import asarray
import cv2
import os
from os.path import join, exists, isdir
from os import listdir, makedirs
import mediapipe as mp

DIR_FOTOS = r"C:\Users\lucas\OneDrive\Desktop\ATP\Fotos"
DIR_FACES = r"C:\Users\lucas\OneDrive\Desktop\ATP\Faces"

detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def extrair_face(arquivo, size=(180, 180)):
    img = Image.open(arquivo).convert('RGB')
    vetor = asarray(img)
    img_bgr = cv2.cvtColor(vetor, cv2.COLOR_RGB2BGR)
    resultado = detector.process(img_bgr)

    if not resultado.detections:
        print(f"[AVISO] Nenhuma face detectada em: {arquivo}")
        return None

    bbox = resultado.detections[0].location_data.relative_bounding_box
    h, w, _ = vetor.shape
    x1 = int(bbox.xmin * w)
    y1 = int(bbox.ymin * h)
    x2 = int((bbox.xmin + bbox.width) * w)
    y2 = int((bbox.ymin + bbox.height) * h)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)

    face = vetor[y1:y2, x1:x2]
    image = Image.fromarray(face).resize(size)
    return image

def flip_image(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def load_fotos(diretorio_src, diretorio_target):
    if not exists(diretorio_target):
        makedirs(diretorio_target)

    for filename in listdir(diretorio_src):
        path = join(diretorio_src, filename)
        path_tg = join(diretorio_target, filename)
        path_tg_flip = join(diretorio_target, "flip-" + filename)

        if not path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        face = extrair_face(path)
        if face is not None:
            face.save(path_tg)
            flip = flip_image(face)
            flip.save(path_tg_flip)

def carregar_dir(diretorio_src, diretorio_target):
    for subdir in listdir(diretorio_src):
        path = join(diretorio_src, subdir)
        path_tg = join(diretorio_target, subdir)

        if not isdir(path):
            continue

        if not exists(path_tg):
            makedirs(path_tg)

        load_fotos(path, path_tg)

if __name__ == "__main__":
    carregar_dir(DIR_FOTOS, DIR_FACES)
