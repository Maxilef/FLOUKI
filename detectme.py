from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import sys

def flouter_image(image_path):
    # Charger l'image avec OpenCV
    im1 = Image.open(image_path)
    image_cv2 = cv2.cvtColor(np.array(im1), cv2.COLOR_RGB2BGR)

    # Faire la prédiction avec YOLO
    results = model.predict(source=im1) #indice de confiance conf=0.85
    boxes_data = results[0].boxes.data.cpu() #recuperation des coordonné de detection

    # Appliquer le flou aux boîtes englobantes (rectanlge coordonné de detection)
    for box_data in boxes_data:
        x_min, y_min, x_max, y_max, confidence, class_id = box_data.tolist()

        x, y, w, h = int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)
        region = image_cv2[y:y + h, x:x + w]
        blurred_region = cv2.GaussianBlur(region, (35, 35), 0)  # ajuster niveau de flou
        image_cv2[y:y + h, x:x + w] = blurred_region

    # Afficher l'image avec les boîtes floutées
    cv2.imshow('Image avec boîtes floutées', image_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # Enregistrer l'image floutée
    output_path = './runs/save/image_floue.jpg'
    cv2.imwrite(output_path, image_cv2)

def flouter_video(video_path):
    # Charger la vidéo
    cap = cv2.VideoCapture(video_path)

    # Obtenir les propriétés de la vidéo
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Créer un objet VideoWriter pour écrire la vidéo floue
    output_path = './runs/save/video_floue.avi'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir le cadre en format PIL
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Faire la prédiction avec YOLO
        results = model.predict(source=frame_pil)
        boxes_data = results[0].boxes.data.cpu()

        # Appliquer le flou aux boîtes englobantes
        for box_data in boxes_data:
            x_min, y_min, x_max, y_max, confidence, class_id = box_data.tolist()

            x, y, w, h = int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)
            region = frame[y:y + h, x:x + w]
            blurred_region = cv2.GaussianBlur(region, (35, 35), 0)  # ajuster niveau de flou
            frame[y:y + h, x:x + w] = blurred_region

        # Écrire le cadre traité dans la vidéo de sortie
        out.write(frame)

        # Afficher la vidéo originale avec les boîtes floutées
        cv2.imshow('Vidéo avec boîtes floutées', frame)

        # Appuyez sur 'q' pour quitter la boucle
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Libérer les ressources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def flouter_webcam(webcam_id=0):

    # Ouvrir la connexion à la webcam
    cap = cv2.VideoCapture(webcam_id)

    while True:
        # Lire le cadre de la webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir le cadre en format PIL
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Faire la prédiction avec YOLO
        results = model.predict(source=frame_pil)
        boxes_data = results[0].boxes.data.cpu()

        # Appliquer le flou aux boîtes englobantes
        for box_data in boxes_data:
            x_min, y_min, x_max, y_max, confidence, class_id = box_data.tolist()

            x, y, w, h = int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)
            region = frame[y:y + h, x:x + w]
            blurred_region = cv2.GaussianBlur(region, (35, 35), 0)  # ajuster niveau de flou
            frame[y:y + h, x:x + w] = blurred_region

        # Afficher le cadre avec les boîtes floutées
        cv2.imshow('Webcam avec boîtes floutées', frame)

        # Appuyez sur 'q' pour quitter la boucle
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Utilisation: python3 detectme.py <chemin_vers_fichier_ou_webcam_id>")
        sys.exit(1)

    # Charger le modèle YOLO
    model = YOLO(r".\runs\detect\train\weights\best.pt")

    file_path_or_webcam_id = sys.argv[1]

    # Vérifier si l'argument est un fichier ou un numéro de webcam
    if file_path_or_webcam_id.lower() == '0':
        flouter_webcam(0)
    elif file_path_or_webcam_id.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        flouter_image(file_path_or_webcam_id)
    elif file_path_or_webcam_id.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
        flouter_video(file_path_or_webcam_id)
    else:
        print("Format de fichier non pris en charge ou numéro de webcam incorrect.")
        sys.exit(1)
