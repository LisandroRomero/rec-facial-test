import cv2

def detectar():

    ruta = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(ruta)
    # Detectar cámaras
    camaras = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            camaras.append(i)
            cap.release()


    if not camaras:
        print("No se detectaron cámaras")
        exit()


    print("Cámaras disponibles:")
    for cam in camaras:
        print(f"- Cámara {cam}")


    indice = int(input("Seleccioná el número de cámara: "))
    if indice not in camaras:
        print("Cámara no válida")
        exit()


    cap = cv2.VideoCapture(indice)


    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(30, 30)
            )


        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        cv2.imshow(f"Cámara {indice}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()