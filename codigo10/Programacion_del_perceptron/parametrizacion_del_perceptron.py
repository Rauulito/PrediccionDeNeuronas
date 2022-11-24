from numpy import exp, array, random

#      PARAMETRIZACIÓN DEL PERCEPTRÓN
#--------------------------------------

#Generación de los pesos en el intervalo [-1;1]
random.seed(1)
limiteMin = -1
limiteMax = 1

w11 = (limiteMax-limiteMin) * random.random() + limiteMin
w21 = (limiteMax-limiteMin) * random.random() + limiteMin
w31 = (limiteMax-limiteMin) * random.random() + limiteMin

#El sesgo
sesgo = 1
wb = 0

#Almacenamiento de los pesos iniciales, solo para visualización al final del aprendizaje
peso = [w11,w21,w31,wb]

#Tasa de aprendizaje
txAprendizaje = 0.1

#Cantidad de épocas
epochs = 300000
