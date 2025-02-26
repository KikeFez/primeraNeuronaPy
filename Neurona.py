import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Desactivar el mensaje de oneDNN (opcional)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Definir el modelo de la red neuronal
modelo = keras.Sequential([
    layers.Dense(8, activation='relu', input_shape=(3,)),  
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Dense(6, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Dense(1, activation='sigmoid')  
])

# Compilar el modelo
modelo.compile(
    optimizer='RMSprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Datos de entrada (X) y salida esperada (Y)
X = np.array([
    [0.1, 0.2, 0.3], 
    [0.4, 0.5, 0.6], 
    [0.7, 0.8, 0.9], 
    [0.2, 0.4, 0.6]
], dtype=np.float32)  # Convertir a float32 para evitar advertencias de TensorFlow

Y = np.array([
    [0], [1], [1], [0]
], dtype=np.float32)

# Entrenar la red neuronal
modelo.fit(X, Y, epochs=500, verbose=1, batch_size=2, validation_split=0.25)

# Evaluar con una nueva entrada
nueva_entrada = np.array([[0.3, 0.5, 0.7]], dtype=np.float32)
salida = modelo.predict(nueva_entrada)

print("Salida de la red neuronal:", salida)
