from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Paso 2 Cargar los datos
data = [
    ['Rojo', 'Redonda', 'Suave', 'Manzana'],
    ['Naranja', 'Redonda', 'Rugosa', 'Naranja'],
    ['Amarillo', 'Alargada', 'Rugosa', 'Plátano'],
    ['Rojo', 'Redonda', 'Rugosa', 'Manzana'],
    ['Amarillo', 'Redonda', 'Suave', 'Plátano'],
    ['Naranja', 'Alargada', 'Rugosa', 'Naranja']
]

# Paso 3 Preprocesar los datos
# Convertir variables categóricas en numéricas utilizando codificación one-hot
color_encoder = LabelEncoder()
forma_encoder = LabelEncoder()
textura_encoder = LabelEncoder()

color_labels = [row[0] for row in data]
forma_labels = [row[1] for row in data]
textura_labels = [row[2] for row in data]
etiquetas = [row[3] for row in data]

color_encoded = color_encoder.fit_transform(color_labels)
forma_encoded = forma_encoder.fit_transform(forma_labels)
textura_encoded = textura_encoder.fit_transform(textura_labels)

# Crear conjunto de características X y etiquetas y
X = list(zip(color_encoded, forma_encoded, textura_encoded))
y = etiquetas

# Paso 4 Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 5 Crear y ajustar el modelo de árbol de decisiones
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Paso 6 Realizar predicciones en el conjunto de prueba
predicciones = clf.predict(X_test)

# Paso 7 Evaluar la precisión del modelo
precision = accuracy_score(y_test, predicciones)
print(Precisión del modelo, precision)

# Predicción para una fruta desconocida
fruta_desconocida = ['Rojo', 'Alargada', 'Suave']
color_desconocido = color_encoder.transform([fruta_desconocida[0]])
forma_desconocida = forma_encoder.transform([fruta_desconocida[1]])
textura_desconocida = textura_encoder.transform([fruta_desconocida[2]])

prediccion = clf.predict([[color_desconocido[0], forma_desconocida[0], textura_desconocida[0]]])
print('La fruta desconocida es', prediccion)