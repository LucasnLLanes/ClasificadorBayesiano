# ============================================================
# Clasificador Naive Bayes para detección de noticias falsas
# Autor: Lucas Nahuel Llanes
# Año: 2026
# Materia: Aprendizaje Automático
# Docente: Scudero, Yanina Ximena
# Instituto Superior de Formación Técnica N° 197
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# ------------------------------------------------------------
# 1. Carga del dataset
# ------------------------------------------------------------
noticias = {
    'texto': [
        "El presidente anunció una nueva reforma educativa",
        "Descubren que la vacuna convierte a las personas en robots",
        "La NASA confirma el hallazgo de agua en Marte",
        "Científicos afirman que la Tierra es plana",
        "El ministerio de salud lanza campaña contra el dengue",
        "Celebridades usan crema milagrosa para rejuvenecer 30 años",
        "Se inaugura el nuevo hospital en la ciudad",
        "Estudio revela que comer chocolate cura el cáncer",
        "Gobierno aprueba ley de protección ambiental",
        "Investigadores aseguran que los teléfonos espían nuestros sueños"
    ],
    'categoria': [
        "real", "fake", "real", "fake", "real",
        "fake", "real", "fake", "real", "fake"
    ]
}

df = pd.DataFrame(noticias)
X = df['texto']
y = df['categoria']

print("=== Conjunto de datos ===")
print(df)
print()


# ------------------------------------------------------------
# 2. División entrenamiento / prueba
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

print("=== División de datos ===")
print("Entrenamiento:", len(X_train), "noticias")
print("Prueba       :", len(X_test), "noticias")
print()


# ------------------------------------------------------------
# 3. Vectorización y entrenamiento del modelo
# ------------------------------------------------------------
vectorizador = CountVectorizer()
X_train_vec = vectorizador.fit_transform(X_train)

modelo = MultinomialNB()
modelo.fit(X_train_vec, y_train)


# ------------------------------------------------------------
# 4. Evaluación sobre el conjunto de prueba
# ------------------------------------------------------------
X_test_vec = vectorizador.transform(X_test)
y_pred = modelo.predict(X_test_vec)

print("=== Reporte de clasificación (conjunto de prueba) ===")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print()


# ------------------------------------------------------------
# 5. Matriz de confusión
# ------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred, labels=modelo.classes_)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=modelo.classes_,
    yticklabels=modelo.classes_
)
plt.xlabel('Predicción')
plt.ylabel('Valor real')
plt.title('Matriz de confusión - Naive Bayes')
plt.tight_layout()
plt.savefig('matriz_confusion_llanes.png')
plt.show()


# ------------------------------------------------------------
# 6. Validación cruzada con pipeline
# ------------------------------------------------------------
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

print("=== Validación cruzada (5 folds) ===")
print("Precisión por fold :", scores)
print("Precisión media    :", scores.mean())
print("Desviación estándar:", scores.std())
print()


# ------------------------------------------------------------
# 7. Clasificación de nuevas noticias
# ------------------------------------------------------------
nuevas_noticias = [
    "Nuevo estudio demuestra que el café mejora la memoria",
    "Expertos afirman que los gatos pueden hablar con humanos"
]

nuevas_vec = vectorizador.transform(nuevas_noticias)
predicciones = modelo.predict(nuevas_vec)

print("=== Predicciones sobre noticias nuevas ===")
for noticia, etiqueta in zip(nuevas_noticias, predicciones):
    print(f"- {noticia}  ->  {etiqueta}")