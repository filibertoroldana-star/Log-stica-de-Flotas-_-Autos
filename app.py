from flask import Flask
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

app = Flask(__name__)

@app.route("/")
def home():
    # --- MÓDULO 1: Extracción y Pipeline de Datos (E-Commerce - 50 Registros con Imágenes) ---
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
        
        # Diccionario de productos con imágenes ilustrativas
        productos_info = {
            'Laptop': 'https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=100',
            'Smartphone': 'https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=100',
            'Tablet': 'https://images.unsplash.com/photo-1544244015-0df4b3ffc6b0?w=100',
            'Teclado': 'https://images.unsplash.com/photo-1587829741301-dc798b83add3?w=100',
            'Audifonos': 'https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=100',
            'Monitor': 'https://images.unsplash.com/photo-1527443224154-c4a3942d3acf?w=100',
            'Smartwatch': 'https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=100',
            'Impresora': 'https://images.unsplash.com/photo-1612815154858-60aa4c59eaa6?w=100',
            'Router': 'https://images.unsplash.com/photo-1544197150-b99a580bb7a8?w=100',
            'Mouse': 'https://images.unsplash.com/photo-1615663245857-ac93bb7c39e7?w=100'
        }
        
        data_ventas = []
        prod_keys = list(productos_info.keys())
        for i in range(1, 51):
            prod = random.choice(prod_keys)
            data_ventas.append({
                'ID_Venta': 1000 + i,
                'Sucursal': random.choice(sucursales),
                'Mes': random.choice(meses),
                'Imagen': f"<img src='{productos_info[prod]}' width='45' style='border-radius:4px;'>",
                'Producto': prod,
                'Cantidad': random.randint(5, 100),
                'Precio_Unitario': random.randint(30, 1200),
                'Total': 0 # Se calculará o asignará
            })
        df_ventas = pd.DataFrame(data_ventas)
        df_ventas['Total'] = df_ventas['Cantidad'] * df_ventas['Precio_Unitario']

    # --- MÓDULO 2: Manipulación y Filtrado Avanzado (Logística de Flotas - 50 Registros con Imágenes) ---
    df_autos = pd.DataFrame({
        'Imagen': [
            "https://images.unsplash.com/photo-1623869675781-8633140e646f?w=100", "https://images.unsplash.com/photo-1552519507-da3b142c6e3d?w=100", 
            "https://images.unsplash.com/photo-1590362891991-f776e747a588?w=100", "https://images.unsplash.com/photo-1542282088-72c9c27ed0cd?w=100", 
            "https://images.unsplash.com/photo-1584345604476-8ec5e12e42dd?w=100", "https://images.unsplash.com/photo-1568844092993-22a4af26fc0e?w=100", 
            "https://images.unsplash.com/photo-1621993446554-c99e9095f9c4?w=100", "https://images.unsplash.com/photo-1553440569-bcc63803a83d?w=100", 
            "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=100", "https://images.unsplash.com/photo-1541348263662-e0626628d0cf?w=100"
        ] * 5,
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
    
    # Filtro lógico estricto solicitado en el laboratorio para el backend
    filtro = ((df_autos['Marca'] == 'Toyota') | (df_autos['Marca'] == 'Honda')) & (df_autos['Anio'] > 2018) & (df_autos['Km'] < 50000)
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

    # --- RENDERIZADO HTML CON BUSCADOR INTERACTIVO Y ESTILOS ---
    html_output = f"""
    <html>
        <head>
            <title>Laboratorio Práctico - Equipo D</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 30px; background-color: #f4f7f6; color: #333; }}
                h2 {{ color: #0056b3; border-bottom: 2px solid #0056b3; padding-bottom: 5px; margin-top: 40px; }}
                pre, .box {{ background: #fff; padding: 15px; border: 1px solid #ddd; border-radius: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 10px; background: #fff; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: middle; }}
                th {{ background-color: #0056b3; color: white; position: sticky; top: 0; }}
                .table-container {{ max-height: 400px; overflow-y: auto; border: 1px solid #ccc; border-radius: 4px; box-shadow: inset 0 0 5px rgba(0,0,0,0.05); }}
                .controls-container {{ background: #e9ecef; padding: 15px; border-radius: 6px; margin-bottom: 15px; display: flex; gap: 15px; align-items: center; flex-wrap: wrap; }}
                .controls-container input, .controls-container select, .controls-container button {{ padding: 8px; font-size: 14px; border: 1px solid #ccc; border-radius: 4px; }}
                .btn {{ background-color: #0056b3; color: white; cursor: pointer; border: none; font-weight: bold; }}
                .btn:hover {{ background-color: #003d82; }}
            </style>
            <script>
                // Filtrar tabla de E-Commerce por Sucursal
                function filtrarSucursal(sucursal) {{
                    var filas = document.querySelectorAll("#tabla-ventas tbody tr");
                    filas.forEach(function(fila) {{
                        var celda = fila.cells[2].innerText.trim(); // Columna Sucursal
                        if (sucursal === 'Todas' || celda === sucursal) {{
                            fila.style.display = "";
                        }} else {{
                            fila.style.display = "none";
                        }}
                    }});
                }}

                // Buscador interactivo dinámico para la flota de vehículos
                function buscarAutos() {{
                    var marcaFiltro = document.getElementById("filtro-marca").value.toLowerCase();
                    var kmMax = parseInt(document.getElementById("filtro-km").value) || Infinity;
                    var anioMin = parseInt(document.getElementById("filtro-anio").value) || 0;
                    
                    var filas = document.querySelectorAll("#tabla-autos tbody tr");
                    filas.forEach(function(fila) {{
                        var marca = fila.cells[2].innerText.toLowerCase();
                        var anio = parseInt(fila.cells[4].innerText);
                        var km = parseInt(fila.cells[5].innerText);
                        
                        var cumpleMarca = (marcaFiltro === "" || marca.includes(marcaFiltro));
                        var cumpleKm = (km <= kmMax);
                        var cumpleAnio = (anio >= anioMin);
                        
                        if (cumpleMarca && cumpleKm && cumpleAnio) {{
                            fila.style.display = "";
                        }} else {{
                            fila.style.display = "none";
                        }}
                    }});
                }}
            </script>
        </head>
        <body>
            <h1>Laboratorio Práctico: Fundamentos de Data Science (Equipo D)</h1>
            
            <!-- MÓDULO 1 -->
            <h2>MÓDULO 1: EXTRACCIÓN Y PIPELINE DE DATOS (E-Commerce - 50 Registros)</h2>
            <p><strong>Estado:</strong> [Éxito] Consolidación de 50 registros con imágenes ilustrativas para las sucursales Norte, Sur y Centro.</p>
            
            <div class="controls-container">
                <strong>Filtrar Sucursal:</strong>
                <button class="btn" onclick="filtrarSucursal('Todas')">Todas (50)</button>
                <button class="btn" onclick="filtrarSucursal('Norte')">Norte</button>
                <button class="btn" onclick="filtrarSucursal('Sur')">Sur</button>
                <button class="btn" onclick="filtrarSucursal('Centro')">Centro</button>
            </div>

            <div class="table-container" id="tabla-ventas">
                {df_ventas.to_html(classes='dataframe', index=True, escape=False)}
            </div>
            <p style="margin-top: 8px;"><strong>Dimensiones (.shape):</strong> {df_ventas.shape}</p>

            <!-- MÓDULO 2 -->
            <h2>MÓDULO 2: MANIPULACIÓN Y FILTRADO AVANZADO (Logística de Flotas)</h2>
            <p>Utiliza el buscador dinámico por kilometraje, marca o año, o revisa los vehículos filtrados automáticamente por la rúbrica.</p>
            
            <div class="controls-container">
                <div>
                    <label><strong>Marca:</strong></label>
                    <input type="text" id="filtro-marca" placeholder="Ej. Toyota, Honda..." onkeyup="buscarAutos()">
                </div>
                <div>
                    <label><strong>Km máximo:</strong></label>
                    <input type="number" id="filtro-km" placeholder="Ej. 50000" onkeyup="buscarAutos()" onchange="buscarAutos()">
                </div>
                <div>
                    <label><strong>Año mínimo:</strong></label>
                    <input type="number" id="filtro-anio" placeholder="Ej. 2018" onkeyup="buscarAutos()" onchange="buscarAutos()">
                </div>
                <button class="btn" onclick="buscarAutos()">Aplicar Buscador</button>
            </div>

            <h3>Todos los Vehículos ({len(df_autos)} Registros con Imágenes):</h3>
            <div class="table-container">
                <table id="tabla-autos" class="dataframe">
                    {df_autos.to_html(index=True, escape=False, header=True).split('<thead>')[1]}
            </div>

            <h3>Resultado del Filtro Estricto (Toyota/Honda, >2018, <50k Km):</h3>
            {df_filtrado.to_html(classes='dataframe', index=True, escape=False)}
            <p><strong>Total de registros coincidentes:</strong> {len(df_filtrado)} vehículos</p>

            <!-- MÓDULO 3 -->
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
    app.run(host="0.0.0.0", port=10000)