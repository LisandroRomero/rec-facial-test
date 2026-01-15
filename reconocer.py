import cv2
import os

def reconocer():
    if not os.path.exists("dataset"):
        print("Error: No se encontró la carpeta dataset")
        return

    labels = {}
    id_actual = 0

    for persona in os.listdir("dataset"):
        labels[id_actual] = persona
        id_actual += 1

    if not labels:
        print("Error: No hay personas en el dataset. Por favor, captura rostros primero.")
        return


    if not os.path.exists("reconocedor.yml"):
        print("Error: No se encontró el archivo reconocedor.yml")
        print("Por favor, primero entrena el modelo con la opción 3 del menú.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("reconocedor.yml")


    face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


    cap = cv2.VideoCapture(0)


    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame")
            break



        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            rostro = gray[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (200, 200))

            id_pred, confianza = recognizer.predict(rostro)

            if confianza < 70:
                nombre = labels[id_pred]
            else:
                nombre = "Desconocido"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{nombre} ({int(confianza)})",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0,255,0), 2)


        cv2.imshow("Reconocimiento Facial", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()