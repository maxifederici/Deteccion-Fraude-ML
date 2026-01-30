# ============================================================
# TRABAJO FINAL DE GRADO - UNIVERSIDAD SIGLO 21
# Autor: Maximiliano Federici
# Proyecto: Deteccion de fraude electronico en pagos con tarjetas de credito
# Dataset: Credit Card Fraud Detection (Kaggle)
# ============================================================

# -------------------------------------------
# A - Analisis Exploratorio de Datos (EDA)
# -------------------------------------------

print("--------------------------------------")
print("A-Analisis Exploratorio de Datos (EDA)")
print("--------------------------------------")

# A.1. Importacion de librerias
import matplotlib
matplotlib.use('Qt5Agg') 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, average_precision_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import learning_curve
from tqdm import tqdm

# A.2. Carga de los datasets

train = pd.read_csv('d:\\SIGLO 21\\2025\\SEM311 - SEMINARIO FINAL EN CIENCIA DE DATOS\\Prototipo\\Credit Card Fraud Detection\\fraudTrain.csv', 
                   usecols=lambda x: x != 'Unnamed: 0') 

test = pd.read_csv('d:\\SIGLO 21\\2025\\SEM311 - SEMINARIO FINAL EN CIENCIA DE DATOS\\Prototipo\\Credit Card Fraud Detection\\fraudTest.csv', 
                   usecols=lambda x: x != 'Unnamed: 0') 

print("Datasets cargados correctamente")
print(f"Filas y columnas Train: {train.shape}\n")
print(f"Filas y columnas Test: {test.shape}\n")

# A.3. Revision de la estructura Train

print("Primeras filas del dataset Train:")
print(train.head(),"\n")

print("Descripcion estadistica Train:")
print(train.describe(),"\n")      

print("Valores nulos por columnas Train:")
print(train.isnull().sum(),"\n")

# A.4. Analisis descriptivo y deteccion de outliers Train

# Distribucion de la variable objetivo
print("Distribucion de 'is_fraud' Train (0 = legitima, 1 = fraudulenta):")
print(train['is_fraud'].value_counts(normalize=True) * 100)
print("Distribucion de 'is_fraud' Test (0 = legitima, 1 = fraudulenta):")
print(test['is_fraud'].value_counts(normalize=True) * 100)

# Deteccion de outliers mediante el metodo IQR Train
train_numerica = train.select_dtypes(include=['number'])
Q1 = train_numerica.quantile(0.25)
Q3 = train_numerica.quantile(0.75)
IQR = Q3 - Q1
outliers = ((train_numerica < (Q1 - 1.5 * IQR)) | (train_numerica > (Q3 + 1.5 * IQR))).sum()
print("___________")
print("Cantidad de valores atípicos detectados por variable numerica Train:")
print(outliers[outliers > 0], "\n")

# Histogramas Train
train.hist(figsize=(12,8), bins=30)
plt.title("Distribucion gral de las variables Train", fontsize=14)
plt.show()

# Boxplt Train
plt.figure(figsize=(12,8))
sns.boxplot(x='category', y='amt', data=train)
plt.title("Boxplot de 'category' y 'amt' Train")
plt.xticks(rotation=45)
plt.show()

# Correlaciones Train
plt.figure(figsize=(12,8))
corr = train_numerica.corr()
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Mapa de correclacion entre variables Train")
plt.show()

# Distribucion del monto de transaccion por clase Train
plt.figure(figsize=(8,6))
sns.boxplot(x='is_fraud', y='amt', data=train)
plt.title('Distribucion del monto por tipo de transaccion Train')
plt.show()

# -------------------------------------------
# B - Preprocesamiento y balanceo de datos
# -------------------------------------------

print("--------------------------------------")
print("B-Preprocesamiento y balanceo de datos")
print("--------------------------------------")

# B.1. Eliminacion de datos duplicados

train=train.drop_duplicates()
print(f"Train luego de eliminar duplicados: {train.shape}\n")
test=test.drop_duplicates()
print(f"Test luego de eliminar duplicados: {test.shape}\n")

# B.2. Convertir las columnas de fecha/hora

train['trans_date_trans_time'] = pd.to_datetime(train['trans_date_trans_time'])
test['trans_date_trans_time'] = pd.to_datetime(test['trans_date_trans_time'])

train['trans_month'] = train['trans_date_trans_time'].dt.month
train['trans_day'] = train['trans_date_trans_time'].dt.day
train['trans_day_week'] = train['trans_date_trans_time'].dt.dayofweek
train['trans_hour'] = train['trans_date_trans_time'].dt.hour
train['trans_minute'] = train['trans_date_trans_time'].dt.minute

test['trans_month'] = test['trans_date_trans_time'].dt.month
test['trans_day'] = test['trans_date_trans_time'].dt.day
test['trans_day_week'] = test['trans_date_trans_time'].dt.dayofweek
test['trans_hour'] = test['trans_date_trans_time'].dt.hour
test['trans_minute'] = test['trans_date_trans_time'].dt.minute

# B.3. Creacion de variables de tiempo

# Ordenar por tarjeta y tiempo
train = train.sort_values(by=['cc_num', 'trans_date_trans_time'])
test = test.sort_values(by=['cc_num', 'trans_date_trans_time'])

# Crear una nueva variable para el monto promedio en 24h por tarjeta
train['avg_amt_24h_cc_num'] = 0.0
test['avg_amt_24h_cc_num'] = 0.0
for cliente, data in tqdm(train.groupby('cc_num'), desc='Calculando monto promedio 24h por tarjeta'):
    train.loc[data.index, 'avg_amt_24h_cc_num'] = (
        data.set_index('trans_date_trans_time')['amt']
        .rolling('1D', closed='left')  # ventana movil de 1 dia
        .mean()
        .values
    )
for cliente, data in tqdm(test.groupby('cc_num'), desc='Calculando monto promedio 24h por tarjeta'):
    test.loc[data.index, 'avg_amt_24h_cc_num'] = (
        data.set_index('trans_date_trans_time')['amt']
        .rolling('1D', closed='left')  # ventana movil de 1 dia
        .mean()
        .values
    )

# Crear una nueva variable para cantidad de operaciones en 24h por tarjeta
train['count_24h_cc_num'] = 0.0
test['count_24h_cc_num'] = 0.0
for cliente, data in tqdm(train.groupby('cc_num'), desc='Calculando cantidad 24h por tarjeta'):
    train.loc[data.index, 'count_24h_cc_num'] = (
        data.set_index('trans_date_trans_time')['amt']
        .rolling('1D', closed='left')  # ventana movil de 1 dia
        .count()
        .values
    )
for cliente, data in tqdm(test.groupby('cc_num'), desc='Calculando cantidad 24h por tarjeta'):
    test.loc[data.index, 'count_24h_cc_num'] = (
        data.set_index('trans_date_trans_time')['amt']
        .rolling('1D', closed='left')  # ventana movil de 1 dia
        .count()
        .values
    )

# Crear una nueva variable para cantidad de operaciones en 1h por tarjeta
train['count_1h_cc_num'] = 0.0
test['count_1h_cc_num'] = 0.0
for cliente, data in tqdm(train.groupby('cc_num'), desc='Calculando cantidad 1h por tarjeta'):
    train.loc[data.index, 'count_1h_cc_num'] = (
        data.set_index('trans_date_trans_time')['amt']
        .rolling('1h', closed='left')  # ventana movil de 1 hora
        .count()
        .values
    )
for cliente, data in tqdm(test.groupby('cc_num'), desc='Calculando cantidad 1h por tarjeta'):
    test.loc[data.index, 'count_1h_cc_num'] = (
        data.set_index('trans_date_trans_time')['amt']
        .rolling('1h', closed='left')  # ventana movil de 1 dia
        .count()
        .values
    )

# Rellenar valores NaN iniciales (con pocas operaciones)
train['avg_amt_24h_cc_num'] = train['avg_amt_24h_cc_num'].fillna(train['amt'].median())
train['count_24h_cc_num'] = train['count_24h_cc_num'].fillna(train['amt'].median())
train['count_1h_cc_num'] = train['count_1h_cc_num'].fillna(train['amt'].median())
test['avg_amt_24h_cc_num'] = test['avg_amt_24h_cc_num'].fillna(test['amt'].median())
test['count_24h_cc_num'] = test['count_24h_cc_num'].fillna(test['amt'].median())
test['count_1h_cc_num'] = test['count_1h_cc_num'].fillna(test['amt'].median())

# Elimina variable de tiempo original
train = train.drop('trans_date_trans_time', axis=1)
test = test.drop('trans_date_trans_time', axis=1)

# B.4. Codificar variables categoricas

train_encoded = pd.get_dummies(train, columns=['category'], drop_first=True)
train = train_encoded

test_encoded = pd.get_dummies(test, columns=['category'], drop_first=True)
test = test_encoded

# Alinear columnas de test con las de train 
missing_cols = set(train.columns) - set(test.columns)
for c in missing_cols:
    test[c] = 0
test = test[train.columns] 

# B.5. Escalado de columnas numericas 

scaler = StandardScaler()
numericas = train.select_dtypes(include=['number']).columns.drop('is_fraud')
scaler.fit(train[numericas])
train[numericas] = scaler.transform(train[numericas])
test[numericas]  = scaler.transform(test[numericas])

# B.6. Balanceo de clases

print("Distribucion original de clases en el conjunto de entrenamiento:")
print(train['is_fraud'].value_counts())

x_train = train.drop('is_fraud', axis=1)
y_train = train['is_fraud']
x_test = test.drop('is_fraud', axis=1)
y_test = test['is_fraud']

# Eliminar columnas con texto
x_train.drop(columns=['cc_num','merchant','trans_num','first','last', 'gender', 'street', 'state', 'city', 'dob', 'job'], errors='ignore', inplace=True)
x_test.drop(columns= ['cc_num','merchant','trans_num','first','last', 'gender', 'street', 'state', 'city', 'dob', 'job'], errors='ignore', inplace=True)

# Balanceo de clases con SMOTE (reduccion para evitar sobreajuste)
smote = SMOTE(sampling_strategy=0.1, random_state=42)
x_train_balan, y_train_balan = smote.fit_resample(x_train, y_train)

print("Distribucion despues de aplicar SMOTE:")
print(y_train_balan.value_counts())

# -------------------------------------------
# C - Entrenamiento inicial de modelos
# -------------------------------------------

print("--------------------------------------")
print("C-Entrenamiento inicial de modelos")
print("--------------------------------------")

# C.1. Regresion Logistica

print("---------------------------")
print("Modelo 1: Regresion Logistica")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(x_train_balan, y_train_balan)
y_pred_lr = lr.predict(x_test)

print("Metricas de Regresion Logistica")
print(classification_report(y_test, y_pred_lr))
print("AUC:", roc_auc_score(y_test, lr.predict_proba(x_test)[:,1]), "\n")

# C.2. Random Forest

print("---------------------------")
print("Modelo 2: Random Forest")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(x_train_balan, y_train_balan)
y_pred_rf = rf.predict(x_test)

print("Metricas de Random Forest")
print(classification_report(y_test, y_pred_rf))
print("AUC:", roc_auc_score(y_test, rf.predict_proba(x_test)[:,1]), "\n")

# C.3. Isolation Forest

print("---------------------------")
print("Modelo 3 - Isolation Forest (No supervisado)")
iso = IsolationForest(contamination=0.005, random_state=42)
iso.fit(x_train)
pred_iso = iso.predict(x_test)
pred_iso = np.where(pred_iso == -1,1,0)

print("Metricas Isolation Forest")
print(classification_report(y_test, pred_iso))
print("AUC (aprox.):", roc_auc_score(y_test, pred_iso), "\n")

# C.4. Comparacion de resultados

resultados = pd.DataFrame({
    "Modelo": ["Regresion Logistica", "Random Forest", "Isolation Forest"],
    "AUC": [
        roc_auc_score(y_test, lr.predict_proba(x_test)[:,1]),
        roc_auc_score(y_test, rf.predict_proba(x_test)[:,1]),
        roc_auc_score(y_test, pred_iso)
        ]
})
print("Resumen comparativo de desempeño de modelos")
print(resultados.sort_values(by="AUC", ascending=False))

# --------------------------------------------------
# D - Optimizacion y validacian de modelos predictivos
# --------------------------------------------------

print("--------------------------------------")
print("D-Optimizacion de hiperparametros - Random Forest")
print("--------------------------------------")

# D.1 - Muestreo para reducir tiempo de computo

x_train_sample = x_train_balan.sample(frac=0.3, random_state=42)
y_train_sample = y_train_balan.loc[x_train_sample.index]

print(f"Tamaño de muestra para optimizacion: {x_train_sample.shape}")

# D.2 - Definicion del modelo base y espacio de busqueda

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

param_distributions = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 5],
    'max_features': ['sqrt', 'log2'],
 }

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_distributions,
    n_iter=10,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=2
)

# Entrenamiento con búsqueda aleatoria
random_search.fit(x_train_sample, y_train_sample)

print("\nMejores hiperparametros encontrados:")
print(random_search.best_params_)
print("Mejor AUC promedio en validacion cruzada:", round(random_search.best_score_, 4))

# D.3 - Validacion rapida del modelo optimizado

print("---------------------------")
print("Validacion rapida del modelo optimizado (Random Forest)")
best_rf = random_search.best_estimator_

# Probabilidades y AUC preliminar
y_proba_best = best_rf.predict_proba(x_test)[:, 1]
auc_best = roc_auc_score(y_test, y_proba_best)
print(f"AUC preliminar del modelo optimizado en test: {auc_best:.4f}")

# D.4 - Comparacion resumida: Modelo Base vs Optimizado

y_pred_rf = rf.predict(x_test)
y_proba_rf_base = rf.predict_proba(x_test)[:, 1]

report_base = classification_report(y_test, y_pred_rf, output_dict=True)
report_best = classification_report(y_test, best_rf.predict(x_test), output_dict=True)
auc_base = roc_auc_score(y_test, y_proba_rf_base)

comparacion = pd.DataFrame({
    'Modelo': ['Random Forest Base', 'Random Forest Optimizado'],
    'Precision (Clase 1)': [
        round(report_base['1']['precision'], 4),
        round(report_best['1']['precision'], 4)
    ],
    'Recall (Clase 1)': [
        round(report_base['1']['recall'], 4),
        round(report_best['1']['recall'], 4)
    ],
    'F1-Score (Clase 1)': [
        round(report_base['1']['f1-score'], 4),
        round(report_best['1']['f1-score'], 4)
    ],
    'AUC': [round(auc_base, 4), round(auc_best, 4)]
})

print("\nResumen comparativo entre modelo base y optimizado:")
print(comparacion)

# --------------------------------------------------
# E - Evaluacion de desempeño y comparación de resultados
# --------------------------------------------------

print("--------------------------------------")
print("Evaluacion Final del Modelo Random Forest Optimizado")
print("--------------------------------------")

# E.1 - Predicciones finales y metricas

y_pred_best = best_rf.predict(x_test)
y_proba_best = best_rf.predict_proba(x_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred_best)
precision = precision_score(y_test, y_pred_best)
recall = recall_score(y_test, y_pred_best)
f1 = f1_score(y_test, y_pred_best)
roc_auc = roc_auc_score(y_test, y_proba_best)
pr_auc = average_precision_score(y_test, y_proba_best)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")

print("\nReporte detallado de clasificacion:")
print(classification_report(y_test, y_pred_best, digits=4))

# E.2 - Matriz de confusion

plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred_best), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Matriz de Confusion - Random Forest Optimizado")
plt.xlabel("Prediccion")
plt.ylabel("Valor Real")
plt.show()
plt.close('all')

# E.3 - Curva ROC

fpr, tpr, _ = roc_curve(y_test, y_proba_best)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color="darkorange")
plt.plot([0,1],[0,1],'--', color='gray')
plt.title("Curva ROC - Random Forest Optimizado")
plt.xlabel("Tasa de Falsos Positivos (FPR)")
plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
plt.legend()
plt.show()
plt.close('all')

# E.4 - Curva Precision-Recall

precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba_best)
plt.figure(figsize=(6,5))
plt.plot(recall_vals, precision_vals, color="blue", label=f"AP = {pr_auc:.3f}")
plt.title("Curva Precision-Recall - Random Forest Optimizado")
plt.xlabel("Recall (Sensibilidad)")
plt.ylabel("Precision")
plt.legend(loc='lower left')
plt.show()
plt.close('all')

# E.5 - Importancia de variables

importances = pd.Series(best_rf.feature_importances_, index=x_train_balan.columns).sort_values(ascending=False)
plt.figure(figsize=(8,6))
sns.barplot(x=importances[:10], y=importances.index[:10], color="skyblue")
plt.title("Top 10 Variables mas importantes - Random Forest Optimizado")
plt.xlabel("Importancia")
plt.ylabel("Variable")
plt.tight_layout()
plt.show()
plt.close('all')

# E.6 - Curva de aprendizaje

sample_frac = 0.2
X_sample = x_train_balan.sample(frac=sample_frac, random_state=42)
y_sample = y_train_balan.loc[X_sample.index]

train_sizes, train_scores, val_scores = learning_curve(
    estimator=best_rf,
    X=X_sample,
    y=y_sample,
    cv=3,                     
    scoring='roc_auc',
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5),  
    shuffle=True,
    random_state=42
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label="AUC (entrenamiento)", color="blue")
plt.plot(train_sizes, val_mean, label="AUC (validación)", color="orange")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.1)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color="orange", alpha=0.1)
plt.title("Curva de aprendizaje del Random Forest Optimizado")
plt.xlabel("Tamaño del conjunto de entrenamiento")
plt.ylabel("AUC ROC")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.close('all')

# E.7 - Guardar metricas finales

metricas_finales = pd.DataFrame({
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1_Score': [f1],
    'ROC_AUC': [roc_auc],
    'PR_AUC': [pr_auc]
})
metricas_finales.to_csv("d:\\SIGLO 21\\2025\\SEM311 - SEMINARIO FINAL EN CIENCIA DE DATOS\\Prototipo\\metricas_random_forest_final.csv", index=False)
print("\nMetricas finales guardadas en 'metricas_random_forest_final.csv'")

# E.8 - Interpretacion general

print("\nCONCLUSIONES E INTERPRETACION GENERAL:")
print("El modelo Random Forest optimizado muestra un rendimiento sólido, con un equilibrio adecuado entre precisión y sensibilidad.")
print("El alto valor de ROC-AUC confirma su capacidad para distinguir operaciones legítimas de las fraudulentas.")
print("La curva Precision-Recall demuestra que el modelo mantiene buena precisión incluso en un contexto de clases desbalanceadas.")
print("Las variables con mayor importancia se relacionan principalmente con el monto, promedio diario de montos, hora del dia y cantidad de operaciones diarias, coherentes con patrones típicos de fraude.")


