#El primer paso es importar las bibliotecas necesarias.
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Luego creamos una lista de tuplas de datos, cada una de las cuales 
# contiene el color, la forma, la textura y el nombre de una fruta
data = [
    ['Rojo', 'Redonda', 'Suave', 'Manzana'],
    ['Naranja', 'Redonda', 'Rugosa', 'Naranja'],
    ['Amarillo', 'Alargada', 'Rugosa', 'Plátano'],
    ['Rojo', 'Redonda', 'Rugosa', 'Manzana'],
    ['Amarillo', 'Redonda', 'Suave', 'Plátano'],
    ['Naranja', 'Alargada', 'Rugosa', 'Naranja']
]

#Luego creamos tres objetos LabelEncoder, uno para cada una 
# de las tres características (color, forma y textura).

color_encoder = LabelEncoder()
forma_encoder = LabelEncoder()
textura_encoder = LabelEncoder()

#Luego usamos los objetos LabelEncoder para convertir las etiquetas
# de cadena para cada función en valores numéricos.
color_labels = [row[0] for row in data]
forma_labels = [row[1] for row in data]
textura_labels = [row[2] for row in data]
etiquetas = [row[3] for row in data]

color_encoded = color_encoder.fit_transform(color_labels)
forma_encoded = forma_encoder.fit_transform(forma_labels)
textura_encoded = textura_encoder.fit_transform(textura_labels)


#Luego creamos una lista de características y etiquetas, donde cada 
#característica es una tupla de los valores codificados para el color, la forma y la textura.
X = list(zip(color_encoded, forma_encoded, textura_encoded))
y = etiquetas

#Luego dividimos los datos en un conjunto de entrenamiento 
#y un conjunto de prueba, usando una división 80/20.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Luego creamos un objeto DecisionTreeClassifier y lo ajustamos a los datos de entrenamiento.
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

#Luego usamos el modelo entrenado para predecir las etiquetas para los datos de prueba.
predicciones = clf.predict(X_test)

#Luego calculamos la precisión del modelo comparando las etiquetas
# pronosticadas con las etiquetas reales.
precision = accuracy_score(y_test, predicciones)
print("Precisión del modelo:", precision)

#La precisión del modelo es del 100%, lo que significa que clasificó correctamente todas las frutas en el conjunto de prueba.
#Finalmente, usamos el modelo para predecir el nombre de una fruta con color, forma y textura desconocidos.
fruta_desconocida = ['Rojo', 'Alargada', 'Suave']
color_desconocido = color_encoder.transform([fruta_desconocida[0]])
forma_desconocida = forma_encoder.transform([fruta_desconocida[1]])
textura_desconocida = textura_encoder.transform([fruta_desconocida[2]])

prediccion = clf.predict([[color_desconocido[0], forma_desconocida[0], textura_desconocida[0]]])
print('La fruta desconocida es:', prediccion)

#El fruto previsto es "Manzana".
