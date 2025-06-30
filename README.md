# HEURÍSTICAS PARA LA OPTIMIZACIÓN DE LA EQUIDAD EN ÁRBOLES DE DECISIÓN (TFG)

Este repositorio contiene el código desarrollado como parte de un Trabajo de Fin de Grado (TFG), titulado **“HEURÍSTICAS PARA LA OPTIMIZACIÓN DE LA EQUIDAD EN ÁRBOLES DE DECISIÓN”**.

El proyecto implementa algoritmos de búsqueda local (*First Improvement*, *Best Improvement*) y un esquema de *Variable Neighborhood Descent (VND)* para modificar árboles de decisión ya entrenados, con el objetivo de reducir el sesgo medido mediante la métrica **Average Odds Difference (AOD)**, manteniendo al mismo tiempo la precisión predictiva del modelo.

## 🔗 Basado en

Este repositorio es una **extensión (fork)** del trabajo original disponible en:  
https://github.com/hortzdb/search-based-fairness-repair

Se han introducido modificaciones y mejoras que incluyen:

- Reestructuración del código para modularidad y reutilización.
- Inclusión de operadores adicionales (`pruning` y `relabeling`).
- Implementación de VND que combina ambos operadores.
- Registro automático de métricas y control temporal de ejecución.
- Documentación y ejemplos adaptados al contexto académico del TFG.

## 📂 Estructura del repositorio

├── codigo/ # Implementación principal de los algoritmos

├── datos/ # Conjuntos de datos utilizados (vía AIF360)

├── resultados/ # Métricas, logs y evaluaciones

├── doc/ # Documento del TFG y anexos

├── requirements.txt # Dependencias del proyecto

└── README.md # Este archivo


### Requisitos

- Python 3.8+
- Dependencias en `requirements.txt` (incluye AIF360, scikit-learn, numpy, etc.)

### Instrucciones

```bash
git clone https://github.com/dprezp/TFG.git
cd TFG
pip install -r requirements.txt


# Ejecuta búsqueda local con Todos los operadores
python approach_dt.py 

# Ejecuta búsqueda local con distintos operadores
python approach_dt.py --method "Método"

# Ejecuta VND combinando pruning y relabeling
python Prueba_VND.py

