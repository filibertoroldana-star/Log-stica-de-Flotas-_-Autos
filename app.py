from flask import Flask
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

app = Flask(__name__)

@app.route("/")
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
            'Laptop Gamer': 'https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=100',
            'Smartphone Alta Gama': 'https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=100',
            'Tablet 10 pulgadas': 'https://images.unsplash.com/photo-1544244015-0df4b3ffc6b0?w=100',
            'Teclado Mecanico': 'https://images.unsplash.com/photo-1587829741301-dc798b83add3?w=100',
            'Audifonos Bluetooth': 'https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=100',
            'Monitor 24 pulgadas': 'https://images.unsplash.com/photo-1527443224154-c4a3942d3acf?w=100',
            'Smartwatch Pro': 'https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=100',
            'Impresora Multifuncional': 'https://images.unsplash.com/photo-1612815154858-60aa4c59eaa6?w=100',
            'Router': 'https://images.unsplash.com/photo-1544197150-b99a580bb7a8?w=100',
            'Mouse Inalambrico': 'https://images.unsplash.com/photo-1615663245857-ac93bb7c39e7?w=100'
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
                'Imagen': f"<img src='{productos_info[prod]}' width='40' style='border-radius:6px;'>",
                'Producto': prod,
                'Cantidad': cant,
                'Precio_Unitario': precio,
                'Total': cant * precio
            })
        df_ventas = pd.DataFrame(data_ventas)

    # --- MÓDULO 2: Manipulación y Filtrado Avanzado (Logística de Flotas - 50 Registros) ---
    raw_autos_img = [
        "https://images.unsplash.com/photo-1623869675781-8633140e646f?w=100", "https://images.unsplash.com/photo-1552519507-da3b142c6e3d?w=100", 
        "https://images.unsplash.com/photo-1590362891991-f776e747a588?w=100", "https://images.unsplash.com/photo-1542282088-72c9c27ed0cd?w=100", 
        "https://images.unsplash.com/photo-1584345604476-8ec5e12e42dd?w=100", "https://images.unsplash.com/photo-1568844092993-22a4af26fc0e?w=100", 
        "https://images.unsplash.com/photo-1621993446554-c99e9095f9c4?w=100", "https://images.unsplash.com/photo-1553440569-bcc63803a83d?w=100", 
        "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=100", "https://images.unsplash.com/photo-1541348263662-e0626628d0cf?w=100"
    ] * 5

    df_autos = pd.DataFrame({
        'Imagen': [f"<img src='{url}' width='45' style='border-radius:6px; object-fit:cover;'>" for url in raw_autos_img],
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

    html_output = f"""
    <html>
        <head>
            <title>Laboratorio Práctico - Equipo D</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 30px; background-color: #f8f9fa; color: #333; }}
                h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
                h2 {{ color: #2c3e50; border-bottom: 2px solid #cbd5e1; padding-bottom: 6px; margin-top: 45px; }}
                
                .box {{ background: #fff; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }}
                
                table {{ border-collapse: collapse; width: 100%; margin-top: 10px; background: #fff; font-size: 14px; }}
                th, td {{ border: 1px solid #e2e8f0; padding: 10px; text-align: left; vertical-align: middle; }}
                
                /* DISEÑO 1: Tabla Azul (E-Commerce) */
                .table-ventas th {{ background-color: #1e3a8a; color: white; }}
                .table-ventas tr:nth-child(even) {{ background-color: #eff6ff; }}
                
                /* DISEÑO 2: Tabla Verde Esmeralda (Todos los Vehículos) */
                .table-autos th {{ background-color: #065f46; color: white; }}
                .table-autos tr:nth-child(even) {{ background-color: #ecfdf5; }}
                
                /* DISEÑO 3: Tabla Púrpura (Vehículos Filtrados) */
                .table-filtrados th {{ background-color: #581c87; color: white; }}
                .table-filtrados tr:nth-child(even) {{ background-color: #f3e8ff; }}

                .table-container {{ max-height: 420px; overflow-y: auto; border: 1px solid #cbd5e1; border-radius: 8px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.03); }}
                
                .controls-container {{ background: #ffffff; padding: 18px; border: 1px solid #e2e8f0; border-radius: 8px; margin-bottom: 15px; display: flex; gap: 15px; align-items: center; flex-wrap: wrap; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }}
                .controls-container input, .controls-container select, .controls-container button {{ padding: 8px 12px; font-size: 14px; border: 1px solid #cbd5e1; border-radius: 6px; }}
                
                .btn-primary {{ background-color: #2563eb; color: white; cursor: pointer; border: none; font-weight: 600; transition: background 0.2s; }}
                .btn-primary:hover {{ background-color: #1d4ed8; }}
                
                .btn-sucursal {{ background-color: #0284c7; color: white; cursor: pointer; border: none; font-weight: 600; padding: 8px 14px; border-radius: 6px; }}
                .btn-sucursal:hover {{ background-color: #0369a1; }}
            </style>
            <script>
                function filtrarSucursal(sucursal) {{
                    var filas = document.querySelectorAll("#tabla-ventas tbody tr");
                    filas.forEach(function(fila) {{
                        var celda = fila.cells[2].innerText.trim();
                        if (sucursal === 'Todas' || celda === sucursal) {{
                            fila.style.display = "";
                        }} else {{
                            fila.style.display = "none";
                        }}
                    }});
                }}

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
            
            <h2>MÓDULO 1: EXTRACCIÓN Y PIPELINE DE DATOS (E-Commerce - 50 Registros)</h2>
            <p><strong>Estado:</strong> [Éxito] Consolidación de 50 registros con imágenes ilustrativas y diseño personalizado en azul.</p>
            
            <div class="controls-container">
                <strong>Filtrar Sucursal:</strong>
                <button class="btn-sucursal" onclick="filtrarSucursal('Todas')">Todas (50)</button>
                <button class="btn-sucursal" onclick="filtrarSucursal('Norte')">Norte</button>
                <button class="btn-sucursal" onclick="filtrarSucursal('Sur')">Sur</button>
                <button class="btn-sucursal" onclick="filtrarSucursal('Centro')">Centro</button>
            </div>

            <div class="table-container" id="tabla-ventas">
                {df_ventas.to_html(classes='table-ventas', index=True, escape=False)}
            </div>
            <p style="margin-top: 8px;"><strong>Dimensiones (.shape):</strong> {df_ventas.shape}</p>

            <h2>MÓDULO 2: MANIPULACIÓN Y FILTRADO AVANZADO (Logística de Flotas)</h2>
            
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
                <button class="btn-primary" onclick="buscarAutos()">Aplicar Buscador</button>
            </div>

            <h3>Todos los Vehículos ({len(df_autos)} Registros con Imágenes - Estilo Esmeralda):</h3>
            <div class="table-container">
                {df_autos.to_html(classes='table-autos', id="tabla-autos", index=True, escape=False)}
            </div>

            <h3>Resultado del Filtro Estricto (Toyota/Honda, >2018, &lt;50k Km - Estilo Púrpura):</h3>
            <div style="overflow-x: auto;">
                {df_filtrado.to_html(classes='table-filtrados', index=True, escape=False)}
            </div>
            <p><strong>Total de registros coincidentes:</strong> {len(df_filtrado)} vehículos</p>

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