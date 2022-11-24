


#Visualización gráfica
import matplotlib.pyplot as plt
plt.plot(Grafica_MSE)
plt.ylabel('MSE')
plt.show()

print("--- VERIFICACIONES ----")

for i in range(0,4):
    print("Observación:"+str(valores_entradas_X[i])+ " - Esperado: "+str(valores_a_predecir_Y[i])+" - Predicción: "+str(sesion.run(prediccion, feed_dict={tf_neuronas_entradas_X: [valores_entradas_X[i]]})))



sesion.close()