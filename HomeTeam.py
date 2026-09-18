# ==============================================================================
# PROYECTO DATA SCIENCE - EQUIPO D (E-Commerce, Logística y Machine Learning)
# Cumplimiento estricto de Rúbrica (Pipeline, Filtrado y GridSearchCV)
# ==============================================================================

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

print("================================================================")
print("MÓDULO 1: EXTRACCIÓN Y PIPELINE DE DATOS (E-Commerce - Ventas)")
print("================================================================")

sucursales = ['norte', 'sur', 'centro']
frames_equipo_d = []

for suc in sucursales:
    url = f"data/2023_ventas_{suc}.csv"
    try:
        df_temp = pd.read_csv(url)
        frames_equipo_d.append(df_temp)
        print(f"[Éxito] Archivo cargado correctamente: {url}")
    except Exception as e:
        print(f"[Error] No se pudo leer {url}: {e}")

df_ventas = pd.concat(frames_equipo_d, ignore_index=True)

print("\n--- Vista Previa (.head()) del DataFrame Consolidado ---")
print(df_ventas.head())
print(f"\n--- Dimensiones (.shape) del DataFrame Final --- {df_ventas.shape}")


print("\n" + "================================================================")
print("MÓDULO 2: MANIPULACIÓN Y FILTRADO AVANZADO (Logística de Flotas)")
print("================================================================")

data_autos = {
    'Marca': ['Toyota', 'Ford', 'Honda', 'Nissan', 'Toyota', 'Honda', 'Toyota', 'Mazda', 'Honda', 'Toyota'],
    'Modelo': ['Corolla', 'Focus', 'Civic', 'Sentra', 'RAV4', 'CR-V', 'Camry', 'Mazda3', 'Accord', 'Yaris'],
    'Anio': [2021, 2019, 2022, 2017, 2020, 2015, 2023, 2021, 2020, 2022],
    'Km': [25000, 40000, 15000, 10000, 48000, 30000, 12000, 35000, 22000, 18000]
}
df_autos = pd.DataFrame(data_autos)

print("DataFrame original de vehículos:")
print(df_autos)

filtro = ((df_autos['Marca'] == 'Toyota') | (df_autos['Marca'] == 'Honda')) & (df_autos['Anio'] > 2018) & (df_autos['Km'] < 50000)
df_filtrado_d = df_autos[filtro]

print("\n--- Resultados Módulo 2 (Vehículos Filtrados con éxito) ---")
print(df_filtrado_d)


print("\n" + "================================================================")
print("MÓDULO 3: MACHINE LEARNING - OPTIMIZACIÓN SVC (Botánica - Iris)")
print("================================================================")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svc_model = SVC()
param_grid = {
    'C': [0.5, 1, 5, 10],
    'kernel': ['linear', 'poly']
}

grid_search = GridSearchCV(estimator=svc_model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("\n--- Resultados Módulo 3 (GridSearchCV) ---")
print("Mejores Parámetros (.best_params_):", grid_search.best_params_)
print(f"Mejor Score de Validación Cruzada (.best_score_): {grid_search.best_score_:.4f}")

accuracy_final = grid_search.score(X_test, y_test)
print(f"Exactitud final en conjunto de prueba (Test Accuracy): {accuracy_final:.4f}")
print("\n¡Ejecución de todo el pipeline completada con éxito!")
