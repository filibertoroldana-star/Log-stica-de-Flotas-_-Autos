from flask import Flask, render_template, request
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def home():
    # --- MÓDULO 1: Extracción y Pipeline de Datos (E-Commerce - 50 Registros) ---
    try:
        df_norte = pd.read_csv('data/2023_ventas_norte.csv')
        df_sur = pd.read_csv('data/2023_ventas_sur.csv')
        df_centro = pd.read_csv('data/2023_ventas_centro.csv')
        df_ventas = pd.concat([df_norte, df_sur, df_centro], ignore_index=True)
    except Exception as e:
        import random
        random.seed(42)
        sucursales = ['Norte', 'Sur', 'Centro']
        meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio']
        
        productos_info = {
            'Laptop Gamer': 'https://picsum.photos/id/1/60/40',
            'Smartphone Alta Gama': 'https://picsum.photos/id/160/60/40',
            'Tablet 10 pulgadas': 'https://picsum.photos/id/2/60/40',
            'Teclado Mecanico': 'https://picsum.photos/id/3/60/40',
            'Audifonos Bluetooth': 'https://picsum.photos/id/4/60/40',
            'Monitor 24 pulgadas': 'https://picsum.photos/id/5/60/40',
            'Smartwatch Pro': 'https://picsum.photos/id/6/60/40',
            'Impresora Multifuncional': 'https://picsum.photos/id/9/60/40',
            'Router': 'https://picsum.photos/id/20/60/40',
            'Mouse Inalambrico': 'https://picsum.photos/id/30/60/40'
        }
        
        data_ventas = []
        prod_keys = list(productos_info.keys())
        for i in range(1, 51):
            prod = random.choice(prod_keys)
            cant = random.randint(5, 100)
            precio = random.randint(30, 1200)
            data_ventas.append({
                'ID_Venta': 1000 + i,
                'Sucursal': random.choice(sucursales),
                'Mes': random.choice(meses),
                'Imagen': f"<img src='{productos_info[prod]}' width='45' style='border-radius:4px; object-fit:cover;'>",
                'Producto': prod,
                'Cantidad': cant,
                'Precio_Unitario': precio,
                'Total': cant * precio
            })
        df_ventas = pd.DataFrame(data_ventas)

    # --- MÓDULO 2: Manipulación y Filtrado Avanzado (Logística de Flotas - 50 Registros) ---
    # Usamos un conjunto de IDs de imágenes completamente estables de Picsum que cargan al instante sin fallar
    urls_estables = [
        "https://picsum.photos/id/111/60/40", "https://picsum.photos/id/133/60/40", 
        "https://picsum.photos/id/145/60/40", "https://picsum.photos/id/183/60/40", 
        "https://picsum.photos/id/219/60/40", "https://picsum.photos/id/250/60/40", 
        "https://picsum.photos/id/28/60/40", "https://picsum.photos/id/36/60/40", 
        "https://picsum.photos/id/54/60/40", "https://picsum.photos/id/60/60/40"
    ] * 5

    df_autos = pd.DataFrame({
        'Imagen': [f"<img src='{url}' width='45' style='border-radius:4px; object-fit:cover;'>" for url in urls_estables],
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
    
    # Lógica de Filtrado y Alerta de Años
    ANIO_MINIMO_DB = 2015
    mensaje_alerta = None
    df_filtrado = df_autos.copy()

    if request.method == 'POST':
        marca_ingresada = request.form.get('marca', '').strip()
        km_max_str = request.form.get('km_max', '')
        anio_min_str = request.form.get('anio_min', '')

        if anio_min_str.isdigit():
            anio_min = int(anio_min_str)
            if anio_min < ANIO_MINIMO_DB:
                mensaje_alerta = f"⚠️ No hay registro de ahí para abajo: El año mínimo disponible en el sistema es {ANIO_MINIMO_DB}."
            df_filtrado = df_filtrado[df_filtrado['Anio'] >= anio_min]

        if marca_ingresada:
            df_filtrado = df_filtrado[df_filtrado['Marca'].str.contains(marca_ingresada, case=False, na=False)]

        if km_max_str.isdigit():
            km_max = int(km_max_str)
            df_filtrado = df_filtrado[df_filtrado['Km'] <= km_max]
    else:
        filtro_defecto = ((df_autos['Marca'] == 'Toyota') | (df_autos['Marca'] == 'Honda')) & (df_autos['Anio'] > 2018) & (df_autos['Km'] < 50000)
        df_filtrado = df_autos[filtro_defecto]

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

    html_output = f"""
    <html>
        <head>
            <title>Laboratorio Práctico - Equipo D</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 30px; background-color: #f8f9fa; color: #333; }}
                h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
                h2 {{ color: #2c3e50; border-bottom: 2px solid #cbd5e1; padding-bottom: 6px; margin-top: 45px; }}
                
                .box {{ background: #fff; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }}
                .alert {{ background-color: #fff3cd; color: #856404; padding: 12px; border: 1px solid #ffeeba; border-radius: 6px; margin-bottom: 15px; font-weight: 600; }}
                
                table {{ border-collapse: collapse; width: 100%; margin-top: 10px; background: #fff; font-size: 14px; }}
                th, td {{ border: 1px solid #e2e8f0; padding: 10px; text-align: left; vertical-align: middle; }}
                
                .table-autos th {{ background-color: #065f46; color: white; }}
                .table-filtrados th {{ background-color: #581c87; color: white; }}

                .table-container {{ max-height: 420px; overflow-y: auto; border: 1px solid #cbd5e1; border-radius: 8px; }}
                .controls-container {{ background: #ffffff; padding: 18px; border: 1px solid #e2e8f0; border-radius: 8px; margin-bottom: 15px; display: flex; gap: 15px; align-items: center; flex-wrap: wrap; }}
                .controls-container input, .controls-container button {{ padding: 8px 12px; font-size: 14px; border: 1px solid #cbd5e1; border-radius: 6px; }}
                
                .btn-primary {{ background-color: #2563eb; color: white; cursor: pointer; border: none; font-weight: 600; }}
                .btn-primary:hover {{ background-color: #1d4ed8; }}
            </style>
        </head>
        <body>
            <h1>Laboratorio Práctico: Fundamentos de Data Science (Equipo D)</h1>
            
            <h2>MÓDULO 2: MANIPULACIÓN Y FILTRADO AVANZADO (Logística de Flotas)</h2>
            
            <form method="POST" class="controls-container">
                <div>
                    <label><strong>Marca:</strong></label><br>
                    <input type="text" name="marca" placeholder="Ej. Toyota, Honda...">
                </div>
                <div>
                    <label><strong>Km máximo:</strong></label><br>
                    <input type="number" name="km_max" placeholder="Ej. 50000">
                </div>
                <div>
                    <label><strong>Año mínimo:</strong></label><br>
                    <input type="number" name="anio_min" placeholder="Ej. 2018">
                </div>
                <div style="align-self: flex-end;">
                    <button type="submit" class="btn-primary">Aplicar Buscador</button>
                </div>
            </form>

            {f'<div class="alert">{mensaje_alerta}</div>' if mensaje_alerta else ''}

            <h3>Resultado del Filtro (Estilo Púrpura):</h3>
            <div style="overflow-x: auto;">
                {df_filtrado.to_html(classes='table-filtrados', index=True, escape=False)}
            </div>
            <p><strong>Total de registros coincidentes:</strong> {len(df_filtrado)} vehículos</p>

            <h3>Todos los Vehículos ({len(df_autos)} Registros - Estilo Esmeralda):</h3>
            <div class="table-container">
                {df_autos.to_html(classes='table-autos', index=True, escape=False)}
            </div>

            <h2>MÓDULO 3: MACHINE LEARNING - OPTIMIZACIÓN SVC (Iris Dataset)</h2>
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
    app.run(host="0.0.0.0", port=10000, debug=True)
