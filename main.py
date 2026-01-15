from capturar import capturar
from entrenar import entrenar_modelo
from reconocer import reconocer
from camara import detectar


def menu():
    print("\n=== SISTEMA DE RECONOCIMIENTO FACIAL ===")
    print("1 - Detectar cámaras")
    print("2 - Capturar rostros")
    print("3 - Entrenar modelo")
    print("4 - Reconocer rostros")
    print("0 - Salir")


def main():
    while True:
        menu()
        opcion = input("Seleccioná una opción: ")

        if opcion == "1":
            detectar()

        elif opcion == "2":
            nombre = input("Ingrese el nombre del usuario: ")
            capturar(nombre)

        elif opcion == "3":
            entrenar_modelo()

        elif opcion == "4":
            reconocer()

        elif opcion == "0":
            print("Saliendo...")
            break

        else:
            print("Opción inválida")


if __name__ == "__main__":
    main()
