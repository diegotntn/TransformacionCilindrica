import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Cargar la imagen
ruta_imagen = "igor.jpg"
imagen = Image.open(ruta_imagen)
array_imagen = np.array(imagen)

# Dimensiones de la imagen
m_y, m_x, _ = array_imagen.shape

# Crear una imagen de salida
array_imagen_salida = np.zeros_like(array_imagen)

# Color de fondo especificado (RGBA)
color_fondo = np.array([246, 176, 200])

# Función para transformar en X
def transformar_x(x, y, m_x):
    x_R = x
    alpha = np.arccos(1 - x_R / (m_x / 2))
    x_A = alpha * m_x / np.pi
    return int(x_A), y

# Función para transformar en Y
def transformar_y(x, y, m_y):
    y_R = y
    alpha = np.arccos(1 - y_R / (m_y / 2))
    y_A = alpha * m_y / np.pi
    return x, int(y_A)

# Aplicar transformaciones a los cuadrantes especificados
for y in range(m_y):
    for x in range(m_x):
        if x < m_x // 2 and y < m_y // 2:  # Cuadrante I
            nuevo_x, nuevo_y = transformar_x(x, y, m_x)
        elif x >= m_x // 2 and y < m_y // 2:  # Cuadrante II
            nuevo_x, nuevo_y = transformar_y(x, y, m_y)
        elif x < m_x // 2 and y >= m_y // 2:  # Cuadrante III
            nuevo_x, nuevo_y = transformar_y(x, y, m_y)
        elif x >= m_x // 2 and y >= m_y // 2:  # Cuadrante IV
            nuevo_x, nuevo_y = transformar_x(x, y, m_x)

        # Asegurarse de que las nuevas coordenadas estén dentro del rango de la imagen
        nuevo_x = min(max(nuevo_x, 0), m_x - 1)
        nuevo_y = min(max(nuevo_y, 0), m_y - 1)
        
        # Copiar el píxel a la nueva posición en la imagen de salida
        array_imagen_salida[nuevo_y, nuevo_x] = array_imagen[y, x]

# Rellenar espacios negros con el color de fondo
for y in range(m_y):
    for x in range(m_x):
        if np.all(array_imagen_salida[y, x] == 0):
            array_imagen_salida[y, x] = color_fondo

# Convertir la imagen de salida de vuelta a un objeto PIL y mostrar
imagen_salida = Image.fromarray(array_imagen_salida)
plt.imshow(imagen_salida)
plt.axis('off')
plt.show()
