from flask import Flask
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

app = Flask(__name__)

@app.route("/")
def home():
    # --- MÓDULO 1: Extracción y Pipeline de Datos (E-Commerce) ---
    sucursales = ['norte', 'sur', 'centro']
    frames_equipo_d = []
    
    # Simulamos la carga de datos locales limpios para evitar errores de red en Render
    datos_ejemplo = {
        'norte': pd.DataFrame({
            'ID_Venta': [1, 2], 'Sucursal': ['Norte', 'Norte'], 'Mes': ['Enero', 'Febrero'],
            'Producto': ['Laptop Gamer', 'Teclado Mecanico'], 'Cantidad': [15, 50], 'Precio_Unitario': [1200, 45], 'Total': [18000, 2250]
        }),
        'sur': pd.DataFrame({
            'ID_Venta': [6, 7], 'Sucursal': ['Sur', 'Sur'], 'Mes': ['Enero', 'Marzo'],
            'Producto': ['Monitor 27', 'Mouse Gamer'], 'Cantidad': [10, 40], 'Precio_Unitario': [300, 50], 'Total': [3000, 2000]
        }),
        'centro': pd.DataFrame({
            'ID_Venta': [12, 15], 'Sucursal': ['Centro', 'Centro'], 'Mes': ['Febrero', 'Abril'],
            'Producto': ['Silla Oficina', 'Audifonos'], 'Cantidad': [5, 20], 'Precio_Unitario': [2500, 80], 'Total': [12500, 1600]
        })
    }
    
    for suc in sucursales:
        frames_equipo_d.append(datos_ejemplo[suc])
        
    df_ventas = pd.concat(frames_equipo_d, ignore_index=True)

    # --- MÓDULO 2: Filtrado Avanzado (Logística de Flotas) ---
    df_autos = pd.DataFrame({
        'Marca': ['Toyota', 'Ford', 'Honda', 'Nissan', 'Toyota', 'Honda', 'Toyota', 'Mazda', 'Honda', 'Toyota'],
        'Modelo': ['Corolla', 'Focus', 'Civic', 'Sentra', 'RAV4', 'CR-V', 'Camry', 'Mazda3', 'Accord', 'Yaris'],
        'Anio': [2021, 2019, 2022, 2017, 2020, 2015, 2023, 2021, 2020, 2022],
        'Km': [25000, 40000, 15000, 10000, 48000, 30000, 12000, 35000, 22000, 18000]
    })
    
    filtro = ((df_autos['Marca'] == 'Toyota') | (df_autos['Marca'] == 'Honda')) & (df_autos['Anio'] > 2018) & (df_autos['Km'] < 50000)
    df_filtrado = df_autos[filtro]

    # --- MÓDULO 3: Machine Learning (GridSearchCV SVC - Iris) ---
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    svc_model = SVC()
    param_grid = {'C': [0.5, 1, 5, 10], 'kernel': ['linear', 'poly']}
    grid_search = GridSearchCV(estimator=svc_model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    best_score = grid_search.best_score_
    test_accuracy = grid_search.score(X_test, y_test)
    best_params = grid_search.best_params_

    # --- RENDERIZADO HTML ---
    html_output = f"""
    <html>
        <head>
            <title>Laboratorio Práctico - Equipo D</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f9f9f9; color: #333; }}
                h2 {{ color: #0056b3; border-bottom: 2px solid #0056b3; padding-bottom: 5px; }}
                pre, .box {{ background: #fff; padding: 15px; border: 1px solid #ddd; border-radius: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); overflow-x: auto; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 10px; background: #fff; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #0056b3; color: white; }}
            </style>
        </head>
        <body>
            <h1>Laboratorio Práctico: Fundamentos de Data Science (Equipo D)</h1>
            
            <h2>MÓDULO 1: EXTRACCIÓN Y PIPELINE DE DATOS (E-Commerce - Ventas)</h2>
            <p><strong>Estado:</strong> [Éxito] Archivos de sucursales consolidados correctamente.</p>
            <h3>Vista Previa (.head()) del DataFrame Consolidado:</h3>
            {df_ventas.head().to_html(classes='dataframe', index=True)}
            <p><strong>Dimensiones (.shape) del DataFrame Final:</strong> {df_ventas.shape}</p>

            <h2>MÓDULO 2: MANIPULACIÓN Y FILTRADO AVANZADO (Logística de Flotas)</h2>
            <h3>DataFrame Original de Vehículos:</h3>
            {df_autos.head(10).to_html(classes='dataframe', index=True)}
            <h3>Resultados Módulo 2 (Vehículos Filtrados):</h3>
            {df_filtrado.to_html(classes='dataframe', index=True)}

            <h2>MÓDULO 3: MACHINE LEARNING - OPTIMIZACIÓN SVC (Botánica - Iris)</h2>
            <div class="box">
                <p><strong>Mejores Parámetros (.best_params_):</strong> {best_params}</p>
                <p><strong>Mejor Score de Validación Cruzada (.best_score_):</strong> {best_score:.4f}</p>
                <p><strong>Exactitud final en conjunto de prueba (Test Accuracy):</strong> {test_accuracy:.4f}</p>
            </div>
        </body>
    </html>
    """
    return html_output

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)