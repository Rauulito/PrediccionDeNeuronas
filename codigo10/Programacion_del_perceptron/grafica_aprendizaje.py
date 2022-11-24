


#       GRÁFICA
#--------------------------------------



Grafica_MSE=[]


#--------------------------------------
#    APRENDIZAJE
#--------------------------------------

for epoch in range(0,EPOCH):
    print("EPOCH ("+str(epoch)+"/"+str(epochs)+")")
    predicciones_realizadas_durante_epoch = [];
    predicciones_esperadas = [];
    numObservacion = 0
    for observacion in observaciones_entradas:

        #Carga de la capa de entrada
        x1 = observacion[0];
        x2 = observacion[1];

        #Valor de predicción esperado
        valor_esperado = predicciones[numObservacion][0]

        #Etapa 1: Cálculo de la suma ponderada
        valor_suma_ponderada = suma_ponderada(x1,w11,x2,w21,sesgo,wb)


        #Etapa 2: Aplicación de la función de activación
        valor_predicho = funcion_activacion_sigmoide(valor_suma_ponderada)


        #Etapa 3: Cálculo del error
        valor_error = error_lineal(valor_esperado,valor_predicho)


        #Actualización del peso 1
        #Cálculo ddel gradiente del valor de ajuste y del peso nuevo
        gradiente_W11 = calculo_gradiente(x1,valor_predicho,valor_error)
        valor_ajuste_W11 = calculo_valor_ajuste(gradiente_W11,txAprendizaje)
        w11 = calculo_nuevo_peso(w11,valor_ajuste_W11)

        # Actualización del peso 2
        gradiente_W21 = calculo_gradiente(x2, valor_predicho, valor_error)
        valor_ajuste_W21 = calculo_valor_ajuste(gradiente_W21, txAprendizaje)
        w21 = calculo_nuevo_peso(w21, valor_ajuste_W21)


        # Actualización del peso del sesgo
        gradiente_Wb = calculo_gradiente(sesgo, valor_predicho, valor_error)
        valor_ajuste_Wb = calculo_valor_ajuste(gradiente_Wb, txAprendizaje)
        wb = calculo_nuevo_peso(wb, valor_ajuste_Wb)

        print("     EPOCH (" + str(epoch) + "/" + str(epochs) + ") -  Observación: " + str(numObservacion+1) + "/" + str(len(observaciones_entradas)))

        #Almacenamiento de la predicción realizada:
        predicciones_realizadas_durante_epoch.append(valor_predicho)
        predicciones_esperadas.append(predicciones[numObservacion][0])

        #Paso a la observación siguiente
        numObservacion = numObservacion+1

    MSE = calculo_MSE(predicciones_realizadas_durante_epoch, predicciones)
    Grafica_MSE.append(MSE[0])
    print("MSE: "+str(MSE))



import matplotlib.pyplot as plt
plt.plot(Grafica_MSE)
plt.ylabel('MSE')
plt.show()


print()
print()
print ("¡Aprendizaje terminado!")
print ("Pesos iniciales: " )
print ("W11 = "+str(peso[0]))
print ("W21 = "+str(peso[1]))
print ("Wb = "+str(peso[3]))

print ("Pesos finales: " )
print ("W11 = "+str(w11))
print ("W21 = "+str(w21))
print ("Wb = "+str(wb))

print()
print("--------------------------")
print ("PREDICCIÓN ")
print("--------------------------")
x1 = 1
x2 = 1

#Etapa 1: Cálculo de la suma ponderada
valor_suma_ponderada = suma_ponderada(x1,w11,x2,w21,sesgo,wb)


#Etapa 2: Aplicación de la función de activación
valor_predicho = funcion_activacion_sigmoide(valor_suma_ponderada)
#valor_predicho = funcion_activacion_relu(valor_suma_ponderada)

print("Predicción del [" + str(x1) + "," + str(x2)  + "]")
print("Predicción = " + str(valor_predicho))


