import cv2
import os
import numpy as np

def entrenar_modelo():

    
    labels = {}
    label_id = 0
    x_train = []
    y_labels = []


    for persona in os.listdir("dataset"):
        ruta_persona = os.path.join("dataset", persona)
        labels[label_id] = persona

        for imagen in os.listdir(ruta_persona):
            img_path = os.path.join(ruta_persona, imagen)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            x_train.append(img)
            y_labels.append(label_id)

        label_id += 1


    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("reconocedor.yml")
    print("Entrenamiento completado")
    print("Etiquetas: ", labels)





