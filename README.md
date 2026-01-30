# 🛡️ Detección de Fraude en Tarjetas de Crédito con Machine Learning

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Finalizado-green)

## 📌 Contexto del Negocio
El fraude electrónico es uno de los mayores desafíos financieros globales, con pérdidas anuales que superan los **33 mil millones de dólares** (Nilson Report, 2023). Para una Fintech o Entidad Bancaria, el costo no es solo el dinero robado, sino la pérdida de confianza del cliente y los costos operativos de gestión.

**El Desafío:** Detectar transacciones fraudulentas en un dataset altamente desbalanceado (solo el 0.5% son fraudes), minimizando los "Falsos Negativos" (dejar pasar un fraude) sin bloquear excesivamente a usuarios legítimos.

## 🎯 Objetivo del Proyecto
Desarrollar un modelo predictivo capaz de identificar patrones de comportamiento anómalos en transacciones de tarjetas de crédito, priorizando la **Sensibilidad (Recall)** para capturar la mayor cantidad de fraudes posibles.

## 🛠️ Tecnologías y Herramientas
* **Lenguaje:** Python 3.12
* **Análisis y Procesamiento:** Pandas, NumPy.
* **Visualización:** Matplotlib, Seaborn.
* **Machine Learning:** Scikit-learn (Random Forest, Isolation Forest, Regresión Logística).
* **Manejo de Desbalance:** Imbalanced-learn (SMOTE).

## 📊 Pipeline del Proyecto (Metodología)

### 1. Análisis Exploratorio de Datos (EDA)
Se trabajó con un dataset de **1.8 millones de transacciones**, identificando:
* **Desbalance extremo:** Clase fraudulenta < 1%.
* **Patrones:** Análisis de montos, geolocalización y patrones temporales (horas del día con mayor incidencia de fraude).

### 2. Ingeniería de Características y Preprocesamiento
* **Nuevas Variables:** Creación de variables de ventana temporal (ej. `avg_amt_24h` - monto promedio últimas 24hs) para detectar cambios bruscos de comportamiento.
* **Escalado:** StandardScaler para normalizar variables numéricas.
* **Balanceo de Clases:** Aplicación de **SMOTE** (Synthetic Minority Over-sampling Technique) en el set de entrenamiento para generar ejemplos sintéticos de fraudes y evitar el sesgo del modelo.

### 3. Modelado y Selección
Se entrenaron y compararon tres enfoques:
1.  **Regresión Logística:** Modelo base (Benchmark).
2.  **Isolation Forest:** Enfoque no supervisado (detección de anomalías).
3.  **Random Forest (Optimizado):** Enfoque supervisado.

## 🚀 Resultados del Modelo
El modelo seleccionado fue un **Random Forest Optimizado** mediante validación cruzada y ajuste de hiperparámetros (*GridSearchCV*).

| Métrica | Valor Obtenido | Interpretación de Negocio |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.9848** | Excelente capacidad de distinción entre clases. |
| **Recall (Sensibilidad)** | **0.89** | **Dato Clave:** De cada 100 fraudes, el modelo detecta 89. |
| **Precisión** | **0.73** | De cada 100 alertas, 73 son fraudes reales (baja tasa de falsas alarmas). |

### Visualización de Resultados

**Matriz de Confusión:**
<img width="500" height="400" alt="matriz confusion 2" src="https://github.com/user-attachments/assets/b31bd5b4-14a8-4090-841d-84bc69bcd147" />
*Se observa una fuerte reducción de Falsos Negativos, protegiendo el capital de la empresa.*

**Importancia de Variables:**
<img width="800" height="600" alt="Top 10 2" src="https://github.com/user-attachments/assets/57ac0bdd-1c66-4279-8284-d67fb92348b4" />
*Las variables derivadas del comportamiento histórico (ej. monto promedio últimas 24hs) resultaron ser los predictores más fuertes.*

## 💼 Conclusión e Impacto
Este prototipo demuestra que mediante técnicas avanzadas de **muestreo (SMOTE)** y la optimización de modelos de ensamble (**Random Forest**), es posible construir un sistema de seguridad robusto que:
1.  **Reduce pérdidas económicas** al detectar casi el 90% de los intentos de fraude.
2.  **Mantiene la operatividad**, asegurando que la mayoría de las alertas generadas sean genuinas (Precision aceptable).
3.  **Es escalable**, permitiendo su implementación en flujos de trabajo de *Compliance* en tiempo real.

---
**Autor:** [Maximiliano Federici](https://www.linkedin.com/in/tu-perfil)
*Data Scientist & Economista | Especialista en Análisis de Datos.*
