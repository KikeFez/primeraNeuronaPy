# primeraNeuronaPy

¿Que es una Neurona?
Es un método de la inteligencia artificial (IA) que enseña a las computadoras a procesar datos de una manera similar a como lo hace el cerebro humano.

Que hace esta neurona?

Básicamente, es una mini red neuronal que aprende a clasificar cosas en dos categorías (0 o 1).
Piensa en algo como: "¿Este email es spam o no?" o "¿Este paciente tiene la enfermedad o no?".

Cómo funciona?
Recibe datos de entrada (X) → Cada muestra tiene tres números que representan sus características.
Los procesa con dos capas ocultas → Aquí la red aprende patrones en los datos.
La última capa usa una función sigmoide → Convierte la salida en un número entre 0 y 1 (como una probabilidad).
Se entrena con ejemplos → Al principio se equivoca mucho, pero poco a poco ajusta sus pesos hasta mejorar.
Hace predicciones → Después del entrenamiento, puedes darle nuevos datos y te dirá si cree que es 0 o 1.


Librerias
tensorflow y keras → Son las que crean la red neuronal y manejan el entrenamiento.
numpy → Se usa para manejar los datos como arreglos.

Funciones importantes

Dense(neuronas, activation) → Es como una capa de neuronas. Aquí decides cuántas neuronas quieres y cómo van a activarse.
BatchNormalization() → Normaliza los valores para que la red no se vuelva loca con números muy grandes o pequeños.
Dropout(0.2) → Apaga el 20% de las neuronas de forma aleatoria en cada entrenamiento para que la red no "memorice" los datos.
compile(optimizer, loss, metrics) → Configura el entrenamiento:

optimizer='RMSprop' → Ayuda a ajustar los pesos de la red de manera inteligente.
loss='binary_crossentropy' → Mide qué tanto se está equivocando la red (para clasificación binaria).
metrics=['accuracy'] → Queremos medir qué tan precisa es la red.
fit(X, Y, epochs=500, batch_size=2) → Aquí es donde la red aprende con los datos.
predict(nueva_entrada) → Una vez entrenada, esta función le pasa un dato nuevo a la red y devuelve una predicción.

Al final, esta red puede aprender y hacer predicciones sobre datos similares a los que vio en el entrenamiento.
