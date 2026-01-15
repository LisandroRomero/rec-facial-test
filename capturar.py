import cv2
import os

def capturar(nombre):
    ruta = f"dataset/{nombre}"
    if not os.path.exists(ruta):
        os.makedirs(ruta)

    print(f"Carpeta creada: {ruta}")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    contador = 0

    while contador < 50:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3,5)

        for (x, y, w, h) in faces:
            rostro = gray[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (200, 200))
            cv2.imwrite(f"{ruta}/{contador}.jpg", rostro)
            contador += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Capturando rostros", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()