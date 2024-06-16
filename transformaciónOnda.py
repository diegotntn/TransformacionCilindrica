import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Función para generar la superficie deformante
def generar_superficie_deformante(ancho, alto, f, cx, cy, p):
    x = np.linspace(0, ancho-1, ancho)
    y = np.linspace(0, alto-1, alto)
    X, Y = np.meshgrid(x, y)
    S = np.sin(f * np.sqrt((X - cx)**2 + (Y - cy)**2) + p)
    return S

# Función para cargar la imagen
def cargar_imagen(ruta):
    imagen = Image.open(ruta).convert('RGB')  # Convertir a color
    return np.array(imagen)

# Función para calcular derivadas amplificadas
def calcular_derivadas(S, escala):
    grad_x = np.gradient(S, axis=1) * escala  # Derivada en la dirección x amplificada
    grad_y = np.gradient(S, axis=0) * escala  # Derivada en la dirección y amplificada
    return grad_x, grad_y

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

# Función para transformar la imagen
def transformar_imagen(A, S, Gx, Gy, a):
    MapaX = np.zeros_like(S)
    MapaY = np.zeros_like(S)
    
    for x in range(A.shape[0]):
        for y in range(A.shape[1]):
            # Calcular nuevas coordenadas utilizando la superficie deformante y sus derivadas
            MapaX[x, y] = x + a * S[x, y] * Gx[x, y]
            MapaY[x, y] = y + a * S[x, y] * Gy[x, y]
    
    R = np.zeros_like(A)
    for x in range(A.shape[0]):
        for y in range(A.shape[1]):
            nuevo_x = MapaX[x, y]
            nuevo_y = MapaY[x, y]
            if 0 <= nuevo_x < A.shape[0] and 0 <= nuevo_y < A.shape[1]:
                # Usar interpolación bilineal para obtener el valor del píxel en las nuevas coordenadas
                R[x, y] = interpolar_bilineal(A, nuevo_x, nuevo_y)
            else:
                R[x, y] = [0, 0, 0]  # Negro fuera de los límites
    
    return R

# Cargar la imagen arbitraria
ruta_imagen = 'igor.jpg'  # Ruta de la imagen
A = cargar_imagen(ruta_imagen)

# Parámetros iniciales
ancho, alto = A.shape[1], A.shape[0]  # Tamaño de la imagen
f = 0.2  # Frecuencia
cx, cy = ancho // 2, alto // 2  # Centro de las ondas
p = 0  # Fase
a = 20  # Amplitud
escala_derivadas = 5  # Factor de escala para amplificar las derivadas

# Generar la superficie deformante
S = generar_superficie_deformante(ancho, alto, f, cx, cy, p)

# Mostrar la superficie deformante en escala de grises
plt.figure()
plt.imshow(S, cmap='gray')  # Mostrar en escala de grises
plt.title('Superficie Deformante S(x, y)')
plt.axis('off')  # Ocultar ejes
plt.show()

# Calcular derivadas amplificadas
Gx, Gy = calcular_derivadas(S, escala_derivadas)

# Transformar la imagen usando la superficie deformante
R = transformar_imagen(A, S, Gx, Gy, a)

# Mostrar la imagen transformada
plt.figure()
plt.imshow(R)
plt.title('Imagen Transformada R(x, y)')
plt.axis('off')  # Ocultar ejes
plt.show()

# Nuevos parámetros
f_nuevo = 0.4  # Aumentar aún más la frecuencia
p_nuevo = np.pi / 2
a_nueva = 20  # Mantener la amplitud alta

# Generar nueva superficie deformante
S_nueva = generar_superficie_deformante(ancho, alto, f_nuevo, cx, cy, p_nuevo)

# Mostrar la nueva superficie deformante en escala de grises
plt.figure()
plt.imshow(S_nueva, cmap='gray')  # Mostrar en escala de grises
plt.title('Nueva Superficie Deformante S(x, y)')
plt.axis('off')  # Ocultar ejes
plt.show()

# Calcular nuevas derivadas amplificadas
Gx_nuevo, Gy_nuevo = calcular_derivadas(S_nueva, escala_derivadas)

# Transformar imagen con nuevos parámetros usando la nueva superficie deformante
R_nueva = transformar_imagen(A, S_nueva, Gx_nuevo, Gy_nuevo, a_nueva)

# Mostrar la nueva imagen transformada
plt.figure()
plt.imshow(R_nueva)
plt.title('Nueva Imagen Transformada R(x, y)')
plt.axis('off')  # Ocultar ejes
plt.show()
