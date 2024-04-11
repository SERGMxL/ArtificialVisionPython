import cv2

# Carga el clasificador preentrenado para detección de personas
clasificador = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Lee la imagen de entrada
imagen = cv2.imread('persona.jpg')

# Convierte la imagen a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Detecta personas en la imagen
personas = clasificador.detectMultiScale(gris, 1.1, 4)

# Dibuja un rectángulo alrededor de cada persona detectada
for (x, y, w, h) in personas:
    cv2.rectangle(imagen, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Muestra la imagen con los rectángulos
cv2.imshow('Detección de Personas', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()
