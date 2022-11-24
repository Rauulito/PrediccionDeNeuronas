import parametros_generales
import parametros_de_la_red
import tensorflow as tf

#metemos lo que nos hace falta para que funcione
valores_entradas_X = [[1., 0.], [1., 1.], [0., 1.], [0., 0.]]
valores_a_predecir_Y = [[0.], [1.], [0.], [0.]]

#Variable TensorFLow correspondiente a los valores de neuronas de entrada
tf_neuronas_entradas_X = tf.placeholder(tf.float32, [None, 2])

#Variable TensorFlow correspondiente a la neurona de salida (predicción real)
tf_valores_reales_Y = tf.placeholder(tf.float32, [None, 1])

#-- Sesgo inicializado a 0 --
sesgo = tf.Variable(tf.zeros([1, 1]), tf.float32)

#Adición del sesgo a la suma ponderada
sumaponderada = tf.add(sumaponderada,sesgo)

#Función de activación de tipo sigmoide que permite calcular la predicción
prediccion = tf.sigmoid(sumaponderada)

#Función de error de media cuadrática MSE
funcion_error = tf.reduce_sum(tf.pow(tf_valores_reales_Y-prediccion,2))

#Descenso de gradiente con una tasa de aprendizaje fijada a 0,1
optimizador = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(funcion_error)

#    APRENDIZAJE
#-------------------------------------

#Cantidad de epochs
epochs = 10000

#Inicialización de la variable
init = tf.global_variables_initializer()

#Inicio de una sesión de aprendizaje
sesion = tf.Session()
sesion.run(init)

#Para la realización de la gráfica para la MSE
Grafica_MSE=[]


#Para cada epoch
for i in range(epochs):

    #Realización del aprendizaje con actualzación de los pesos
    sesion.run(optimizador, feed_dict = {tf_neuronas_entradas_X: valores_entradas_X, tf_valores_reales_Y:valores_a_predecir_Y})

    #Calcular el error
    MSE = sesion.run(funcion_error, feed_dict = {tf_neuronas_entradas_X: valores_entradas_X, tf_valores_reales_Y:valores_a_predecir_Y})

    #Visualización de la información
    Grafica_MSE.append(MSE)
    print("EPOCH (" + str(i) + "/" + str(epochs) + ") -  MSE: "+ str(MSE))
