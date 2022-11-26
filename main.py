#pruebas
#Importamos directorios
import sys
sys.path.insert(0,"/Users/Lorenzo/Documents/programacion/2.Desarrollo_OO/PrediccionDeNeuronas-1/codigo-10")
#Importamos funciones
from Perceptron_con_Tensorflow import Perceptron_Tensor
from Programacion_del_perceptron import Perceptron


def iniciar():
    print("Iniciando...")

def iniciar():
    while True:

        print("========================")
        print("  Bienvenido a la Prediccion de Neuronas")
        print("========================")
        print("[1] Programacion del Perceptron")
        print("[2] Perceptron con Tensorflow")
        print("[3] Cerrar el Gestor    ")
        print("========================")

        opcion = input("> ")

        if opcion == '1':
            print("Espere a que el programa termine la ejecucion y le saldra la grafica correspondiente")
            Perceptron()
        if opcion == '2':
            print("Espere a que el programa termine la ejecucion y le saldra la grafica correspondiente")
            Perceptron_Tensor()
        if opcion == '3':
            print("Saliendo...")
            break
    


if __name__ == "__main__":
    iniciar()