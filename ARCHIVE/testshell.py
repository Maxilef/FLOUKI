import os
HOME = os.getcwd()
print(HOME) #racine du projet


#test des import
try :
    from IPython import display
    display.clear_output()

    import ultralytics
    ultralytics.checks()
    print("verification terminer : OK")
except :
    print("verification terminer : NON")
    print("essayé d'installer : pip install ultralytics==8.0.20")



from ultralytics import YOLO
from IPython.display import display, Image

from roboflow import Roboflow
rf = Roboflow(api_key="6bWWYPMBinZXGTZIvtDO")
project = rf.workspace("roboflow-100").project("chess-pieces-mjzgj")
dataset = project.version(2).download("yolov8")


# TRAINNING

# Construction de la commande à exécuter
command = f"yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=25 imgsz=800 plots=True"
# Exécution de la commande
subprocess.run(command, shell=True)


# Validate Custom Model
command = f"yolo task=detect mode=val model={HOME}/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml"
subprocess.run(command, shell=True)


# PREDICTION / DETECTION 
command = f"yolo task=detect mode=predict model={HOME}/runs/detect/train/weights/best.pt conf=0.25 source={dataset.location}/test/images save=True"
subprocess.run(command, shell=True)
