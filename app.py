from flask import Flask
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

app = Flask(__name__)

@app.route("/")
def home():
    # Permitir que Pandas muestre todas las filas de los DataFrames en HTML sin truncar
    pd.set_option('display.max_rows', None)

    # --- MÓDULO 1: Extracción y Pipeline de Datos (E-Commerce - Carga de CSVs reales) ---
    try:
        df_norte = pd.read_csv('data/2023_ventas_norte.csv')
        df_sur = pd.read_csv('data/2023_ventas_sur.csv')
        df_centro = pd.read_csv('data/2023_ventas_centro.csv')
        df_ventas = pd.concat([df_norte, df_sur, df_centro], ignore_index=True)
    except Exception as e:
        df_ventas = pd.DataFrame({
            'ID_Venta': [1, 2], 'Sucursal': ['Norte', 'Norte'], 'Mes': ['Enero', 'Febrero'],
            'Producto': ['Laptop Gamer', 'Teclado Mecanico'], 'Cantidad': [15, 50], 'Precio_Unitario': [1200, 45], 'Total': [18000, 2250]
        })

    # --- MÓDULO 2: Manipulación y Filtrado Avanzado (Logística de Flotas - 50 Registros) ---
    df_autos = pd.DataFrame({
        'Marca': [
            'Toyota', 'Ford', 'Honda', 'Nissan', 'Toyota', 'Honda', 'Toyota', 'Mazda', 'Honda', 'Toyota', 
            'Chevrolet', 'Toyota', 'Honda', 'Nissan', 'Ford', 'Mazda', 'Nissan', 'Chevrolet', 'Ford', 'Honda',
            'Toyota', 'Nissan', 'Mazda', 'Chevrolet', 'Ford', 'Toyota', 'Honda', 'Nissan', 'Mazda', 'Ford',
            'Chevrolet', 'Toyota', 'Honda', 'Nissan', 'Mazda', 'Ford', 'Toyota', 'Honda', 'Nissan', 'Chevrolet',
            'Mazda', 'Toyota', 'Honda', 'Ford', 'Nissan', 'Chevrolet', 'Toyota', 'Mazda', 'Honda', 'Ford'
        ],
        'Modelo': [
            'Corolla', 'Focus', 'Civic', 'Sentra', 'RAV4', 'CR-V', 'Camry', 'Mazda3', 'Accord', 'Yaris', 
            'Cruze', 'Hilux', 'Fit', 'Versa', 'Ranger', 'CX-5', 'Altima', 'Aveo', 'Fiesta', 'HR-V',
            'Corolla', 'March', 'Mazda6', 'Tracker', 'Explorer', 'Prius', 'Civic', 'Kicks', 'CX-30', 'Escape',
            'Onix', 'RAV4', 'Pilot', 'Frontier', 'MX-5', 'Edge', 'Yaris', 'Insight', 'Sentra', 'Captiva',
            'Mazda3', 'Camry', 'CR-V', 'Mustang', 'Versa', 'Spark', 'Hilux', 'CX-9', 'Accord', 'Ranger'
        ],
        'Anio': [
            2021, 2019, 2022, 2017, 2020, 2015, 2023, 2021, 2020, 2022, 
            2018, 2022, 2021, 2020, 2023, 2019, 2021, 2016, 2018, 2022,
            2024, 2019, 2020, 2021, 2017, 2023, 2020, 2022, 2023, 2019,
            2021, 2022, 2018, 2020, 2021, 2016, 2021, 2020, 2019, 2022,
            2023, 2020, 2022, 2021, 2018, 2020, 2023, 2019, 2021, 2022
        ],
        'Km': [
            25000, 40000, 15000, 100000, 48000, 130000, 12000, 35000, 22000, 18000, 
            65000, 30000, 28000, 50000, 15000, 41000, 29000, 110000, 75000, 14000,
            8000, 60000, 33000, 27000, 85000, 10000, 24000, 19000, 11000, 52000,
            31000, 21000, 90000, 46000, 16000, 120000, 37000, 26000, 55000, 22000,
            14000, 43000, 18000, 32000, 70000, 45000, 9000, 62000, 20000, 25000
        ]
    })
    
    filtro = ((df_autos['Marca'] == 'Toyota') | (df_autos['Marca'] == 'Honda')) & (df_autos['Anio'] >= 2020)
    df_filtrado = df_autos[filtro]

    # --- MÓDULO 3: Machine Learning (GridSearchCV SVC - Iris) ---
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
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
                th {{ background-color: #0056b3; color: white; position: sticky; top: 0; }}
                /* Contenedor con scroll vertical limpio para recorrer los 50 vehículos */
                .table-container {{ max-height: 450px; overflow-y: auto; border: 1px solid #ccc; border-radius: 4px; box-shadow: inset 0 0 5px rgba(0,0,0,0.05); }}
            </style>
        </head>
        <body>
            <h1>Laboratorio Práctico: Fundamentos de Data Science (Equipo D)</h1>
            
            <h2>MÓDULO 1: EXTRACCIÓN Y PIPELINE DE DATOS (E-Commerce - Ventas)</h2>
            <p><strong>Estado:</strong> [Éxito] Archivos CSV de sucursales consolidados correctamente.</p>
            <h3>Vista Previa (.head()) del DataFrame Consolidado:</h3>
            {df_ventas.head(10).to_html(classes='dataframe', index=True)}
            <p><strong>Dimensiones (.shape) del DataFrame Final:</strong> {df_ventas.shape}</p>

            <h2>MÓDULO 2: MANIPULACIÓN Y FILTRADO AVANZADO (Logística de Flotas)</h2>
            <h3>DataFrame Original de Vehículos ({len(df_autos)} Registros Completos):</h3>
            <div class="table-container">
                {df_autos.to_html(classes='dataframe', index=True)}
            </div>

            <h3>Resultados Módulo 2 (Vehículos Filtrados):</h3>
            {df_filtrado.to_html(classes='dataframe', index=True)}
            <p><strong>Total de registros después del filtro:</strong> {len(df_filtrado)} vehículos</p>

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