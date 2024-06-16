import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Función para generar la superficie deformante del efecto pinchar
def generar_superficie_pinchar(ancho, alto, cx, cy, sigma):
    x = np.linspace(0, ancho-1, ancho)
    y = np.linspace(0, alto-1, alto)
    X, Y = np.meshgrid(x, y)
    S = np.exp(-((X - cx)**2 + (Y - cy)**2) / sigma**2)
    return S

# Función para cargar la imagen
def cargar_imagen(ruta):
    imagen = Image.open(ruta).convert('RGB')  # Convertir a color
    return np.array(imagen)

# Función para interpolación bilineal
def interpolar_bilineal(imagen, x, y):
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, imagen.shape[0] - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, imagen.shape[1] - 1)

    Ia = imagen[x0, y0]
    Ib = imagen[x1, y0]
    Ic = imagen[x0, y1]
    Id = imagen[x1, y1]

    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id

# Función para transformar la imagen usando el efecto pinchar
def transformar_imagen_pinchar(A, S, a, epsilon=1e-5):
    MapaX = np.zeros_like(S)
    MapaY = np.zeros_like(S)
    
    cx, cy = A.shape[0] // 2, A.shape[1] // 2
    for x in range(A.shape[0]):
        for y in range(A.shape[1]):
            r = np.sqrt((x - cx)**2 + (y - cy)**2)
            denominador = 1 + a * S[x, y]
            if denominador < epsilon:
                denominador = epsilon
            r_nuevo = r / denominador
            theta = np.arctan2(y - cy, x - cx)
            MapaX[x, y] = cx + r_nuevo * np.cos(theta)
            MapaY[x, y] = cy + r_nuevo * np.sin(theta)
    
    R = np.zeros_like(A)
    for x in range(A.shape[0]):
        for y in range(A.shape[1]):
            nuevo_x = MapaX[x, y]
            nuevo_y = MapaY[x, y]
            if 0 <= nuevo_x < A.shape[0] and 0 <= nuevo_y < A.shape[1]:
                R[x, y] = interpolar_bilineal(A, nuevo_x, nuevo_y)
            else:
                R[x, y] = [0, 0, 0]  # Negro fuera de los límites
    
    return R

# Cargar la imagen arbitraria
ruta_imagen = 'igor.jpg'  # Cambia a la ruta de tu imagen
A = cargar_imagen(ruta_imagen)

# Parámetros iniciales para la primera transformación
ancho, alto = A.shape[1], A.shape[0]  # Tamaño de la imagen
cx, cy = ancho // 2, alto // 2  # Centro de las ondas
sigma1 = 100  # Anchura de la zona deformada para la primera transformación
a1 = -0.5  # Parámetro de fuerza para pinchar para la primera transformación

# Generar la superficie deformante y transformar la imagen con los primeros parámetros
S1 = generar_superficie_pinchar(ancho, alto, cx, cy, sigma1)
R1 = transformar_imagen_pinchar(A, S1, a1)

# Mostrar la superficie deformante y la imagen transformada para la primera transformación
plt.figure()
plt.imshow(S1, cmap='gray')  # Mostrar en escala de grises
plt.title('Superficie Deformante S1(x, y)')
plt.axis('off')  # Ocultar ejes
plt.show()

plt.figure()
plt.imshow(R1)
plt.title('Imagen Transformada R1(x, y)')
plt.axis('off')  # Ocultar ejes
plt.show()

# Parámetros iniciales para la segunda transformación
sigma2 = 50  # Anchura de la zona deformada para la segunda transformación
a2 = -1.0  # Parámetro de fuerza para pinchar para la segunda transformación

# Generar la superficie deformante y transformar la imagen con los segundos parámetros
S2 = generar_superficie_pinchar(ancho, alto, cx, cy, sigma2)
R2 = transformar_imagen_pinchar(A, S2, a2)

# Mostrar la superficie deformante y la imagen transformada para la segunda transformación
plt.figure()
plt.imshow(S2, cmap='gray')  # Mostrar en escala de grises
plt.title('Superficie Deformante S2(x, y)')
plt.axis('off')  # Ocultar ejes
plt.show()

plt.figure()
plt.imshow(R2)
plt.title('Imagen Transformada R2(x, y)')
plt.axis('off')  # Ocultar ejes
plt.show()
