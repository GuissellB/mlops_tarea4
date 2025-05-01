# 🤖 Modelo de Predicción de Diabetes

Este proyecto implementa un modelo de clasificación binaria para predecir si una persona tiene diabetes, utilizando un clasificador `LogisticRegression` entrenado sobre el dataset `diabetes.csv`.

El modelo es accesible mediante una API REST construida con **FastAPI**, a través del endpoint `/api/v1/predict-diabetes`.

---

## 🧠 Modelo de IA

- **Algoritmo**: Regresión Logística
- **Preprocesamiento**: Estandarización (`StandardScaler`)
- **Optimización**: GridSearch con validación cruzada (`GridSearchCV`)
- **Métrica evaluada**: AUC, precision, recall y matriz de confusión
- **Archivo del modelo entrenado**: `model/diabetes_logistic_model.pkl`

---

## 📥 Variables de Entrada

El endpoint requiere los siguientes parámetros numéricos (tipo `float`), todos enviados como parte de la solicitud POST:

| Nombre                    | Descripción                                                                |
|---------------------------|----------------------------------------------------------------------------|
| `Pregnancies`             | Número de embarazos                                                        |
| `Glucose`                 | Nivel de glucosa en sangre (mg/dL)                                        |
| `BloodPressure`           | Presión arterial diastólica (mm Hg)                                       |
| `SkinThickness`           | Espesor del pliegue cutáneo (mm)                                          |
| `Insulin`                 | Nivel de insulina sérica (mu U/ml)                                        |
| `BMI`                     | Índice de masa corporal (kg/m²)                                           |
| `DiabetesPedigreeFunction`| Historial familiar de diabetes                                             |
| `Age`                     | Edad de la persona (años)                                                 |

---

## 🎯 Salida del Modelo

| Campo        | Tipo    | Descripción                                                    |
|--------------|---------|----------------------------------------------------------------|
| `prediction` | `string`| Resultado: `"Diabético"` o `"No diabético"` según predicción   |

---

## 🚀 Ejemplo de Solicitud

**POST** `/api/v1/predict-diabetes`

```json
{
  "Pregnancies": 2,
  "Glucose": 150,
  "BloodPressure": 72,
  "SkinThickness": 35,
  "Insulin": 0,
  "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.627,
  "Age": 50
}
```

### ✅ Respuesta esperada:

```json
{
  "prediction": "Diabético"
}
```

---

## 🧪 Requisitos del proyecto

- Python 3.11
- scikit-learn
- pandas
- FastAPI
- joblib
- matplotlib

