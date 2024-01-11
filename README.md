# YOLO Détection d'Objets avec Floutage

Ce projet combine la détection d'objets avec le modèle YOLO (You Only Look Once) et l'effet de floutage pour rendre anonymes certains éléments dans des images, des vidéos ou des flux de webcam. Il est composé de deux programmes :

## 1. Entraînement du Modèle YOLO

### Prérequis

- Python 3.x
- Ultralytics (`pip install yolov5`)

### Utilisation

1. Créez une instance du modèle YOLOv8 pour l'entraînement :

   ```python
   from ultralytics import YOLO
   import torch

   def main():
       # Créez une instance du modèle YOLOv8 pour l'entraînement
       model = YOLO("yolov8n.yaml")  # nouveaux réseaux de neurones vides

       path_dataset = r".\dataset\plate_and_face_detection.yolov8\data.yaml"

       # Entraînez le modèle sur le dataset
       results = model.train(data=path_dataset, epochs=100)
       results = model.val()
       success = model.export(format="onnx")

       # Enregistrez les poids après l'entraînement
       # torch.save(model.state_dict(), "your_model_weights.pt")

       # Chargez les poids pour l'évaluation ou la prédiction
       # loaded_model = YOLO("yolov8n.yaml")
       # loaded_model.load_state_dict(torch.load("your_model_weights.pt"))

   if __name__ == '__main__':
       main()
   ```

2. Assurez-vous que le fichier de poids YOLOv8 (`best.pt`) et les fichiers de configuration sont correctement configurés dans le répertoire `weights` du training que vous venez de faire.

## 2. Détection et Floutage

### Prérequis

- Python 3.x
- OpenCV (`pip install opencv-python`)
- Matplotlib (`pip install matplotlib`)
- Pillow (`pip install pillow`)
- Numpy (`pip install numpy`)
- ultralytics (`pip install ultralytics`)

### Utilisation

1. Clonez le dépôt :

   ```bash
   git clone https://github.com/votre-nom-utilisateur/yolo-detection-objets-floutage.git
   cd yolo-detection-objets-floutage
   ```

2. Installez les dépendances requises :

   ```bash
   pip install -r requirements.txt
   ```

   il s'agit des dependences par defaut pour ultralitcs

3. Exécutez le script pour la détection et le floutage :

   - Pour le floutage sur une image :

     ```bash
     python detectme.py chemin/vers/votre/image.jpg
     ```

   - Pour le floutage sur une vidéo :

     ```bash
     python detectme.py chemin/vers/votre/video.mp4
     ```

   - Pour le floutage sur la webcam :

     ```bash
     python detectme.py 0
     ```

4. Appuyez sur 'q' pour quitter le programme.

### Configuration

- Vous pouvez ajuster le niveau de flou en modifiant la taille du noyau dans les appels de la fonction `cv2.GaussianBlur` à l'intérieur des fonctions `flouter_image`, `flouter_video` et `flouter_webcam`.

- Les images et vidéos de sortie avec les objets floutés seront enregistrées dans le répertoire `runs/save`.

N'hésitez pas à modifier le code selon vos besoins spécifiques !
