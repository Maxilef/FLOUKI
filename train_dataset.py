from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

def main():
    # Créez une instance du modèle YOLOv8 pour l'entraînement
    model = YOLO("yolov8n.yaml") #nouveaux reseaux de neurones vides 
    
    # Plusieur modele d'apprentissage du moin au plus gourment :
    	# yolov8n : nano
    	# yolov8s : small
    	# yolov8m : medium
    	# yolov8l : large
    	# yolov8x : extra large

    path_dataset = r".\dataset\plate_and_face_detection.yolov8\data.yaml"

    # Entraînez le modèle sur le dataset
    results = model.train(data=path_dataset, epochs=100)
    results = model.val()
    success = model.export(format="onnx")

     # Enregistrez les poids après l'entraînement
    #torch.save(model.state_dict(), "your_model_weights.pt")

    # Chargez les poids pour l'évaluation ou la prédiction
    #loaded_model = YOLO("yolov8n.yaml")
    #loaded_model.load_state_dict(torch.load("your_model_weights.pt"))


if __name__ == '__main__':
    main()
