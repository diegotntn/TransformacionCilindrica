import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Función para cargar una imagen
def cargar_imagen(ruta, modo='RGB'):
    img = Image.open(ruta).convert(modo)
    return np.array(img)

# Función para cargar una imagen en escala de grises
def cargar_imagen_gris(ruta):
    img = Image.open(ruta).convert('L')
    return np.array(img)

# Función para redimensionar una imagen
def redimensionar_imagen(imagen, nuevo_tamaño):
    img = Image.fromarray(imagen)
    img = img.resize(nuevo_tamaño, Image.LANCZOS)
    return np.array(img)

# Función para calcular la convolución de una imagen con un kernel
def convolucion(imagen, kernel):
    filas, columnas = imagen.shape
    k_filas, k_columnas = kernel.shape
    salida = np.zeros((filas, columnas))
    
    pad_height = k_filas // 2
    pad_width = k_columnas // 2
    
    imagen_padded = np.pad(imagen, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    
    for i in range(filas):
        for j in range(columnas):
            region = imagen_padded[i:i + k_filas, j:j + k_columnas]
            salida[i, j] = np.sum(region * kernel)
    
    return salida

# Definición de los kernels de Sobel
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Función para realizar la interpolación bilineal
def interpolacion_bilineal(imagen, x, y):
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, imagen.shape[1] - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, imagen.shape[0] - 1)
    
    Ia = imagen[y0, x0]
    Ib = imagen[y0, x1]
    Ic = imagen[y1, x0]
    Id = imagen[y1, x1]
    
    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)
    
    return wa * Ia + wb * Ib + wc * Ic + wd * Id

# Función para aplicar la deformación a la imagen
def aplicar_deformacion(imagen_arbitraria, superficie_deformante, a=0.01):
    Gx = convolucion(superficie_deformante, sobel_x)
    Gy = convolucion(superficie_deformante, sobel_y)
    
    filas, columnas = superficie_deformante.shape
    resultado = np.zeros_like(imagen_arbitraria)

    for canal in range(3):  # Iterar sobre cada canal de color (R, G, B)
        for i in range(filas):
            for j in range(columnas):
                nuevo_x = j + a * superficie_deformante[i, j] * Gx[i, j]
                nuevo_y = i + a * superficie_deformante[i, j] * Gy[i, j]
                
                nuevo_x = np.clip(nuevo_x, 0, columnas - 1)
                nuevo_y = np.clip(nuevo_y, 0, filas - 1)
                
                resultado[i, j, canal] = interpolacion_bilineal(imagen_arbitraria[:, :, canal], nuevo_x, nuevo_y)
    
    return resultado

# Cargar las imágenes
ruta_imagen_arbitraria = 'lover.jpg' 
ruta_superficie_deformante = 'ye.jpg'

imagen_arbitraria = cargar_imagen(ruta_imagen_arbitraria)
superficie_deformante = cargar_imagen_gris(ruta_superficie_deformante)

# Redimensionar la imagen arbitraria al tamaño de la superficie deformante
nuevo_tamaño = superficie_deformante.shape[::-1]  # Invertir para obtener (ancho, alto)
imagen_arbitraria_redimensionada = redimensionar_imagen(imagen_arbitraria, nuevo_tamaño)

# Aplicar la deformación
imagen_deformada = aplicar_deformacion(imagen_arbitraria_redimensionada, superficie_deformante, a=0.01)  # Ajusta el parámetro 'a' según sea necesario

# Mostrar las imágenes
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Imagen Original")
plt.imshow(imagen_arbitraria_redimensionada)
plt.subplot(1, 3, 2)
plt.title("Superficie Deformante")
plt.imshow(superficie_deformante, cmap='gray')
plt.subplot(1, 3, 3)
plt.title("Imagen Deformada")
plt.imshow(imagen_deformada)
plt.show()
