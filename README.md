# Clasificador Naive Bayes para Noticias

**Autor:** Lucas Nahuel Llanes  
**Año:** 2026  
**Materia:** Aprendizaje Automático  
**Docente:** Scudero, Yanina Ximena  
**Institución:** INSTITUTO SUPERIOR DE FORMACIÓN TÉCNICA N° 197

---

## Descripción

Este repositorio contiene una implementación básica de un clasificador **Naive Bayes Multinomial** orientado a la detección de noticias falsas. El flujo de trabajo incluye:

1. Carga de un dataset pequeño de noticias etiquetadas como `real` o `fake`.
2. Vectorización de texto con `CountVectorizer`.
3. Entrenamiento de un modelo `MultinomialNB`.
4. Evaluación mediante validación cruzada estratificada (5 folds).
5. Evaluación con partición 67/33 y matriz de confusión.
6. Predicción sobre noticias nuevas.

---

## Requisitos

- Python 3.10 o superior
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

Instalación:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn