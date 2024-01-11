from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

model = YOLO(r"C:\Users\maxim\Desktop\yolov8\runs\detect\train\weights\best.pt")

# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
#results = model.predict(source="0", show=True, conf=0.70)


# from PIL
im1 = Image.open(r"C:\Users\maxim\Desktop\yolov8\p1.jpg")
image_cv2 = cv2.cvtColor(np.array(im1), cv2.COLOR_RGB2BGR)

video = r"C:\Users\maxim\Desktop\yolov8\video_voiture.mp4"
results = model.predict(source=im1,show = True)  # save plotted images
boxes_data = results[0].boxes.data.cpu()

print(boxes_data)


# Appliquer le flou aux boîtes englobantes
for box_data in boxes_data:
    x_min, y_min, x_max, y_max, confidence, class_id = box_data.tolist()

    x, y, w, h = int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)
    roi = image_cv2[y:y + h, x:x + w]
    blurred_roi = cv2.GaussianBlur(roi, (25, 25), 0)  # Ajustez la taille du noyau selon vos besoins
    image_cv2[y:y + h, x:x + w] = blurred_roi

# Afficher l'image avec les boîtes floutées
cv2.imshow('Image avec boîtes floutées', image_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()


#
