from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

app = Flask(__name__)
app.secret_key = 'clave_secreta_laboratorio_equipo_d_crud'

@app.route("/", methods=['GET', 'POST'])
def home():
    # --- INICIALIZAR DATOS EN SESIÓN (CRUD COMPLETO) ---
    
    # MÓDULO 1: Ventas E-Commerce (Pipeline de Datos)
    if 'ventas' not in session:
        import random
        random.seed(42)
        sucursales = ['Norte', 'Sur', 'Centro']
        meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio']
        
        lista_ventas = []
        for i in range(1, 51):
            cant = random.randint(5, 100)
            precio = random.randint(30, 1200)
            lista_ventas.append({
                'ID': 1000 + i,
                'Sucursal': random.choice(sucursales),
                'Mes': random.choice(meses),
                'Producto': 'Producto Tech',
                'Cantidad': cant,
                'Precio_Unitario': precio,
                'Total': cant * precio
            })
        session['ventas'] = lista_ventas

    # MÓDULO 2: Vehículos (Logística de Flotas)
    if 'autos' not in session:
        marcas_base = [
            'Toyota', 'Ford', 'Honda', 'Nissan', 'Toyota', 'Honda', 'Toyota', 'Mazda', 'Honda', 'Toyota', 
            'Chevrolet', 'Toyota', 'Honda', 'Nissan', 'Ford', 'Mazda', 'Nissan', 'Chevrolet', 'Ford', 'Honda',
            'Toyota', 'Nissan', 'Mazda', 'Chevrolet', 'Ford', 'Toyota', 'Honda', 'Nissan', 'Mazda', 'Ford',
            'Chevrolet', 'Toyota', 'Honda', 'Nissan', 'Mazda', 'Ford', 'Toyota', 'Honda', 'Nissan', 'Chevrolet',
            'Mazda', 'Toyota', 'Honda', 'Ford', 'Nissan', 'Chevrolet', 'Toyota', 'Mazda', 'Honda', 'Ford'
        ]
        modelos_base = [
            'Corolla', 'Focus', 'Civic', 'Sentra', 'RAV4', 'CR-V', 'Camry', 'Mazda3', 'Accord', 'Yaris', 
            'Cruze', 'Hilux', 'Fit', 'Versa', 'Ranger', 'CX-5', 'Altima', 'Aveo', 'Fiesta', 'HR-V',
            'Corolla', 'March', 'Mazda6', 'Tracker', 'Explorer', 'Prius', 'Civic', 'Kicks', 'CX-30', 'Escape',
            'Onix', 'RAV4', 'Pilot', 'Frontier', 'MX-5', 'Edge', 'Yaris', 'Insight', 'Sentra', 'Captiva',
            'Mazda3', 'Camry', 'CR-V', 'Mustang', 'Versa', 'Spark', 'Hilux', 'CX-9', 'Accord', 'Ranger'
        ]
        anios_base = [
            2021, 2019, 2022, 2017, 2020, 2015, 2023, 2021, 2020, 2022, 
            2018, 2022, 2021, 2020, 2023, 2019, 2021, 2016, 2018, 2022,
            2024, 2019, 2020, 2021, 2017, 2023, 2020, 2022, 2023, 2019,
            2021, 2022, 2018, 2020, 2021, 2016, 2021, 2020, 2019, 2022,
            2023, 2020, 2022, 2021, 2018, 2020, 2023, 2019, 2021, 2022
        ]
        kms_base = [
            25000, 40000, 15000, 100000, 48000, 130000, 12000, 35000, 22000, 18000, 
            65000, 30000, 28000, 50000, 15000, 41000, 29000, 110000, 75000, 14000,
            8000, 60000, 33000, 27000, 85000, 10000, 24000, 19000, 11000, 52000,
            31000, 21000, 90000, 46000, 16000, 120000, 37000, 26000, 55000, 22000,
            14000, 43000, 18000, 32000, 70000, 45000, 9000, 62000, 20000, 25000
        ]
        
        session['autos'] = [
            {
                'ID': idx + 1,
                'Marca': marcas_base[idx],
                'Modelo': modelos_base[idx],
                'Anio': anios_base[idx],
                'Km': kms_base[idx]
            } for idx in range(50)
        ]

    # --- CONTROLADORES DE ACCIÓN ---
    mensaje_alerta = None
    mensaje_exito = None

    if request.method == 'POST':
        action = request.form.get('action')

        # Acciones Módulo 1 (Ventas)
        if action == 'agregar_venta':
            suc = request.form.get('sucursal', '').strip().capitalize()
            mes = request.form.get('mes', '').strip()
            cant = request.form.get('cantidad', '')
            precio = request.form.get('precio', '')

            if suc and mes and cant.isdigit() and precio.isdigit():
                c = int(cant)
                p = int(precio)
                nuevo_id = max([v['ID'] for v in session['ventas']], default=1000) + 1
                session['ventas'].append({
                    'ID': nuevo_id,
                    'Sucursal': suc,
                    'Mes': mes,
                    'Producto': 'Producto Tech',
                    'Cantidad': c,
                    'Precio_Unitario': p,
                    'Total': c * p
                })
                session.modified = True
                mensaje_exito = f"✅ Venta registrada correctamente en sucursal {suc} (Precio y Total calculados)."
            else:
                mensaje_alerta = "⚠️ Complete todos los campos de la venta correctamente."

        elif action == 'eliminar_venta':
            venta_id = int(request.form.get('id', 0))
            session['ventas'] = [v for v in session['ventas'] if v['ID'] != venta_id]
            session.modified = True
            mensaje_exito = "🗑️ Venta eliminada correctamente."

        elif action == 'modificar_venta':
            venta_id = int(request.form.get('id', 0))
            suc = request.form.get('sucursal', '').strip().capitalize()
            mes = request.form.get('mes', '').strip()
            cant = request.form.get('cantidad', '')
            precio = request.form.get('precio', '')

            if suc and mes and cant.isdigit() and precio.isdigit():
                c = int(cant)
                p = int(precio)
                for v in session['ventas']:
                    if v['ID'] == venta_id:
                        v['Sucursal'] = suc
                        v['Mes'] = mes
                        v['Cantidad'] = c
                        v['Precio_Unitario'] = p
                        v['Total'] = c * p
                session.modified = True
                mensaje_exito = f"✏️ Venta ID {venta_id} modificada correctamente."

        # Acciones Módulo 2 (Flota de Autos)
        elif action == 'agregar_auto':
            m = request.form.get('marca', '').strip().capitalize()
            modelo = request.form.get('modelo', '').strip()
            anio = request.form.get('anio', '')
            km = request.form.get('km', '')

            if m and modelo and anio.isdigit() and km.isdigit():
                nuevo_id = max([a['ID'] for a in session['autos']], default=0) + 1
                session['autos'].append({
                    'ID': nuevo_id,
                    'Marca': m,
                    'Modelo': modelo,
                    'Anio': int(anio),
                    'Km': int(km)
                })
                session.modified = True
                mensaje_exito = f"✅ Vehículo {m} {modelo} agregado correctamente."
            else:
                mensaje_alerta = "⚠️ Complete correctamente todos los campos del vehículo."

        elif action == 'eliminar_auto':
            auto_id = int(request.form.get('id', 0))
            session['autos'] = [a for a in session['autos'] if a['ID'] != auto_id]
            session.modified = True
            mensaje_exito = "🗑️ Vehículo eliminado correctamente."

        elif action == 'modificar_auto':
            auto_id = int(request.form.get('id', 0))
            m = request.form.get('marca', '').strip().capitalize()
            modelo = request.form.get('modelo', '').strip()
            anio = request.form.get('anio', '')
            km = request.form.get('km', '')

            if m and modelo and anio.isdigit() and km.isdigit():
                for a in session['autos']:
                    if a['ID'] == auto_id:
                        a['Marca'] = m
                        a['Modelo'] = modelo
                        a['Anio'] = int(anio)
                        a['Km'] = int(km)
                session.modified = True
                mensaje_exito = f"✏️ Vehículo ID {auto_id} modificado correctamente."

    # --- PREPARAR DATAFRAMES Y FILTROS ---
    df_ventas = pd.DataFrame(session['ventas'])
    df_autos = pd.DataFrame(session['autos'])

    imagenes_por_marca = {
        'Toyota': 'https://images.unsplash.com/photo-1590362891991-f776e747a588?w=100&auto=format&fit=crop&q=60',
        'Ford': 'https://images.unsplash.com/photo-1552519507-da3b142c6e3d?w=100&auto=format&fit=crop&q=60',
        'Honda': 'https://images.unsplash.com/photo-1617814076367-b759c7d7e738?w=100&auto=format&fit=crop&q=60',
        'Nissan': 'https://images.unsplash.com/photo-1583121274602-3e2820c69888?w=100&auto=format&fit=crop&q=60',
        'Mazda': 'https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?w=100&auto=format&fit=crop&q=60',
        'Chevrolet': 'https://images.unsplash.com/photo-1563720223185-11003d516935?w=100&auto=format&fit=crop&q=60'
    }

    # Lógica de Filtro para Módulo 1 (Ventas por Sucursal)
    df_ventas_filtrado = df_ventas.copy()
    if request.method == 'POST' and request.form.get('action') == 'filtrar_ventas':
        sucursal_buscada = request.form.get('filtro_sucursal', '').strip()
        if sucursal_buscada:
            df_ventas_filtrado = df_ventas_filtrado[df_ventas_filtrado['Sucursal'].str.contains(sucursal_buscada, case=False, na=False)]

    # Lógica de Filtro para Módulo 2 (Flota de Autos)
    df_filtrado = df_autos.copy()
    if request.method == 'POST' and request.form.get('action') == 'filtrar_autos':
        marca_ingresada = request.form.get('filtro_marca', '').strip()
        km_str = request.form.get('filtro_km', '').strip()
        anio_str = request.form.get('filtro_anio', '').strip()

        if marca_ingresada:
            df_filtrado = df_filtrado[df_filtrado['Marca'].str.contains(marca_ingresada, case=False, na=False)]

        if km_str.isdigit():
            km_buscado = int(km_str)
            df_filtrado = df_filtrado[df_filtrado['Km'] == km_buscado]

        if anio_str.isdigit():
            anio_buscado = int(anio_str)
            if anio_buscado not in df_autos['Anio'].values:
                mensaje_alerta = f"⚠️ Año no establecido: No existen registros para el año {anio_buscado}."
                df_filtrado = pd.DataFrame(columns=df_autos.columns)
            else:
                df_filtrado = df_filtrado[df_filtrado['Anio'] == anio_buscado]

    if not df_filtrado.empty:
        df_filtrado_display = df_filtrado.copy()
        df_filtrado_display.insert(1, 'Imagen', [f'<img src="{imagenes_por_marca.get(m, imagenes_por_marca["Toyota"])}" width="45" style="border-radius:4px; object-fit:cover;">' for m in df_filtrado['Marca']])
    else:
        df_filtrado_display = df_filtrado.copy()

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

    # Funciones de renderizado de tablas HTML
    def generar_html_tabla_ventas(df):
        html = "<table class='table-ventas'><thead><tr><th>ID</th><th>Sucursal</th><th>Mes</th><th>Producto</th><th>Cantidad</th><th>Precio Unitario</th><th>Total</th><th>Acciones</th></tr></thead><tbody>"
        for _, row in df.iterrows():
            vid = row['ID']
            vsuc = row['Sucursal']
            vmes = row['Mes']
            vprod = row['Producto']
            vcant = row['Cantidad']
            vprec = row['Precio_Unitario']
            vtot = row['Total']
            
            html += f"<tr>"
            html += f"<td>{vid}</td><td>{vsuc}</td><td>{vmes}</td><td>{vprod}</td><td>{vcant}</td><td></td><td></td>"
            html += f"<td>"
            html += f"<button type='button' onclick='abrirModalVenta({vid}, \"{vsuc}\", \"{vmes}\", {vcant}, {vprec})' class='btn-warning'>Modificar</button> "
            html += f"<form action='/' method='POST' style='display:inline;' onsubmit='return confirm(\"¿Estás seguro de eliminar estos datos de venta?\");'>"
            html += f"<input type='hidden' name='action' value='eliminar_venta'><input type='hidden' name='id' value='{vid}'>"
            html += f"<button type='submit' class='btn-danger'>Eliminar</button></form>"
            html += f"</td></tr>"
        html += "</tbody></table>"
        return html

    def generar_html_tabla_autos(df):
        html = "<table class='table-autos'><thead><tr><th>ID</th><th>Imagen</th><th>Marca</th><th>Modelo</th><th>Año</th><th>Km</th><th>Acciones</th></tr></thead><tbody>"
        for _, row in df.iterrows():
            aid = row['ID']
            amarca = row['Marca']
            amodelo = row['Modelo']
            aanio = row['Anio']
            akm = row['Km']
            img_url = imagenes_por_marca.get(amarca, imagenes_por_marca['Toyota'])
            img_tag = f'<img src="{img_url}" width="45" style="border-radius:4px; object-fit:cover;">'
            
            html += f"<tr>"
            html += f"<td>{aid}</td><td>{img_tag}</td><td>{amarca}</td><td>{amodelo}</td><td>{aanio}</td><td>{akm}</td>"
            html += f"<td>"
            html += f"<button type='button' onclick='abrirModalAuto({aid}, \"{amarca}\", \"{amodelo}\", {aanio}, {akm})' class='btn-warning'>Modificar</button> "
            html += f"<form action='/' method='POST' style='display:inline;' onsubmit='return confirm(\"¿Estás seguro de eliminar este vehículo?\");'>"
            html += f"<input type='hidden' name='action' value='eliminar_auto'><input type='hidden' name='id' value='{aid}'>"
            html += f"<button type='submit' class='btn-danger'>Eliminar</button></form>"
            html += f"</td></tr>"
        html += "</tbody></table>"
        return html

    html_output = f"""
    <html>
        <head>
            <title>Laboratorio Práctico - Equipo D (3 Módulos Integrados)</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 30px; background-color: #f8f9fa; color: #333; }}
                h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
                h2 {{ color: #2c3e50; border-bottom: 2px solid #cbd5e1; padding-bottom: 6px; margin-top: 45px; }}
                .box {{ background: #fff; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); margin-bottom: 20px; }}
                .alert {{ background-color: #fff3cd; color: #856404; padding: 12px; border: 1px solid #ffeeba; border-radius: 6px; margin-bottom: 15px; font-weight: 600; }}
                .success {{ background-color: #d1e7dd; color: #0f5132; padding: 12px; border: 1px solid #badbcc; border-radius: 6px; margin-bottom: 15px; font-weight: 600; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 10px; background: #fff; font-size: 14px; }}
                th, td {{ border: 1px solid #e2e8f0; padding: 10px; text-align: left; vertical-align: middle; }}
                .table-autos th {{ background-color: #065f46; color: white; }}
                .table-filtrados th {{ background-color: #581c87; color: white; }}
                .table-ventas th {{ background-color: #1e3a8a; color: white; }}
                .table-container {{ max-height: 400px; overflow-y: auto; border: 1px solid #cbd5e1; border-radius: 8px; }}
                .controls-container {{ background: #ffffff; padding: 18px; border: 1px solid #e2e8f0; border-radius: 8px; margin-bottom: 15px; display: flex; gap: 15px; align-items: center; flex-wrap: wrap; }}
                .controls-container input, .controls-container select, .controls-container button {{ padding: 8px 12px; font-size: 14px; border: 1px solid #cbd5e1; border-radius: 6px; }}
                .btn-primary {{ background-color: #2563eb; color: white; cursor: pointer; border: none; font-weight: 600; }}
                .btn-primary:hover {{ background-color: #1d4ed8; }}
                .btn-success {{ background-color: #059669; color: white; cursor: pointer; border: none; font-weight: 600; }}
                .btn-success:hover {{ background-color: #047857; }}
                .btn-warning {{ background-color: #d97706; color: white; cursor: pointer; border: none; padding: 5px 10px; border-radius: 4px; font-weight: 600; }}
                .btn-warning:hover {{ background-color: #b45309; }}
                .btn-danger {{ background-color: #dc2626; color: white; cursor: pointer; border: none; padding: 5px 10px; border-radius: 4px; font-weight: 600; }}
                .btn-danger:hover {{ background-color: #b91c1c; }}
                .modal {{ display: none; position: fixed; z-index: 10; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.5); }}
                .modal-content {{ background-color: #fff; margin: 10% auto; padding: 25px; border: 1px solid #888; width: 400px; border-radius: 8px; box-shadow: 0 5px 15px rgba(0,0,0,0.3); }}
                .close {{ color: #aaa; float: right; font-size: 28px; font-weight: bold; cursor: pointer; }}
                .close:hover {{ color: black; }}
            </style>
            <script>
                function abrirModalAuto(id, marca, modelo, anio, km) {{
                    document.getElementById('edit_auto_id').value = id;
                    document.getElementById('edit_auto_marca').value = marca;
                    document.getElementById('edit_auto_modelo').value = modelo;
                    document.getElementById('edit_auto_anio').value = anio;
                    document.getElementById('edit_auto_km').value = km;
                    document.getElementById('modalAuto').style.display = 'block';
                }}
                function cerrarModalAuto() {{ document.getElementById('modalAuto').style.display = 'none'; }}

                function abrirModalVenta(id, sucursal, mes, cantidad, precio) {{
                    document.getElementById('edit_venta_id').value = id;
                    document.getElementById('edit_venta_sucursal').value = sucursal;
                    document.getElementById('edit_venta_mes').value = mes;
                    document.getElementById('edit_venta_cantidad').value = cantidad;
                    document.getElementById('edit_venta_precio').value = precio;
                    document.getElementById('modalVenta').style.display = 'block';
                }}
                function cerrarModalVenta() {{ document.getElementById('modalVenta').style.display = 'none'; }}
            </script>
        </head>
        <body>
            <h1>Laboratorio Práctico: Fundamentos de Data Science (Equipo D)</h1>
            
            {f'<div class="success">{mensaje_exito}</div>' if mensaje_exito else ''}
            {f'<div class="alert">{mensaje_alerta}</div>' if mensaje_alerta else ''}

            <!-- MÓDULO 1: Ventas E-Commerce -->
            <h2>MÓDULO 1: PIPELINE DE DATOS Y VENTAS (Centro, Sur y Norte)</h2>
            <div class="box">
                <h4>➕ Registrar Nueva Venta (Cálculo automático de Precio y Total):</h4>
                <form method="POST" class="controls-container">
                    <input type="hidden" name="action" value="agregar_venta">
                    <div>
                        <label><strong>Sucursal:</strong></label><br>
                        <select name="sucursal">
                            <option value="Norte">Norte</option>
                            <option value="Sur">Sur</option>
                            <option value="Centro">Centro</option>
                        </select>
                    </div>
                    <div>
                        <label><strong>Mes:</strong></label><br>
                        <input type="text" name="mes" placeholder="Ej. Julio" required>
                    </div>
                    <div>
                        <label><strong>Cantidad:</strong></label><br>
                        <input type="number" name="cantidad" placeholder="Ej. 25" required>
                    </div>
                    <div>
                        <label><strong>Precio Unitario ($):</strong></label><br>
                        <input type="number" name="precio" placeholder="Ej. 450" required>
                    </div>
                    <div style="align-self: flex-end;">
                        <button type="submit" class="btn-success">Guardar Venta</button>
                    </div>
                </form>

                <h4>🔍 Buscador de Ventas por Sucursal (Norte, Sur o Centro):</h4>
                <form method="POST" class="controls-container">
                    <input type="hidden" name="action" value="filtrar_ventas">
                    <div>
                        <label><strong>Escribe Sucursal (Norte, Sur, Centro):</strong></label><br>
                        <input type="text" name="filtro_sucursal" placeholder="Ej. Centro" value="{request.form.get('filtro_sucursal', '') if request.method == 'POST' and request.form.get('action') == 'filtrar_ventas' else ''}">
                    </div>
                    <div style="align-self: flex-end;">
                        <button type="submit" class="btn-primary">Filtrar Sucursal</button>
                    </div>
                </form>
                
                <p><strong>Resultados en Ventas ({len(df_ventas_filtrado)} registros coincidentes / {len(df_ventas)} totales):</strong></p>
                <div class="table-container">
                    {generar_html_tabla_ventas(df_ventas_filtrado)}
                </div>
            </div>

            <!-- MÓDULO 2: Logística de Flotas -->
            <h2>MÓDULO 2: MANIPULACIÓN Y FILTRADO AVANZADO (Logística de Flotas)</h2>
            
            <div class="box">
                <h4>➕ Agregar Nuevo Vehículo:</h4>
                <form method="POST" class="controls-container">
                    <input type="hidden" name="action" value="agregar_auto">
                    <div>
                        <label><strong>Marca:</strong></label><br>
                        <input type="text" name="marca" placeholder="Toyota, Ford..." required>
                    </div>
                    <div>
                        <label><strong>Modelo:</strong></label><br>
                        <input type="text" name="modelo" placeholder="Corolla..." required>
                    </div>
                    <div>
                        <label><strong>Año:</strong></label><br>
                        <input type="number" name="anio" placeholder="2022" required>
                    </div>
                    <div>
                        <label><strong>Kilometraje:</strong></label><br>
                        <input type="number" name="km" placeholder="15000" required>
                    </div>
                    <div style="align-self: flex-end;">
                        <button type="submit" class="btn-success">Guardar Vehículo</button>
                    </div>
                </form>
            </div>

            <form method="POST" class="controls-container">
                <input type="hidden" name="action" value="filtrar_autos">
                <div>
                    <label><strong>Marca:</strong></label><br>
                    <input type="text" name="filtro_marca" placeholder="Ej. Toyota...">
                </div>
                <div>
                    <label><strong>Kilometraje exacto:</strong></label><br>
                    <input type="number" name="filtro_km" placeholder="Ej. 25000">
                </div>
                <div>
                    <label><strong>Año exacto:</strong></label><br>
                    <input type="number" name="filtro_anio" placeholder="Ej. 2018">
                </div>
                <div style="align-self: flex-end;">
                    <button type="submit" class="btn-primary">Aplicar Buscador</button>
                </div>
            </form>

            <h3>Resultado del Filtro de Flota:</h3>
            <div style="overflow-x: auto;">
                {df_filtrado_display.to_html(classes='table-filtrados', index=False, escape=False) if not df_filtrado.empty else '<p>No se encontraron vehículos con los filtros seleccionados.</p>'}
            </div>
            <p><strong>Total de registros coincidentes en filtro:</strong> {len(df_filtrado)} vehículos</p>

            <h3>Todos los Vehículos Registrados ({len(df_autos)} Totales):</h3>
            <div class="table-container">
                {generar_html_tabla_autos(df_autos)}
            </div>

            <!-- MÓDULO 3: Machine Learning -->
            <h2>MÓDULO 3: MACHINE LEARNING - OPTIMIZACIÓN SVC (Iris Dataset)</h2>
            <div class="box">
                <p><strong>Mejores Parámetros (.best_params_):</strong> {best_params}</p>
                <p><strong>Mejor Score de Validación Cruzada (.best_score_):</strong> {best_score:.4f}</p>
                <p><strong>Exactitud final en conjunto de prueba (Test Accuracy):</strong> {test_accuracy:.4f}</p>
            </div>

            <!-- MODAL EDITAR VEHÍCULO -->
            <div id="modalAuto" class="modal">
                <div class="modal-content">
                    <span class="close" onclick="cerrarModalAuto()">&times;</span>
                    <h3>Modificar Vehículo</h3>
                    <form method="POST">
                        <input type="hidden" name="action" value="modificar_auto">
                        <input type="hidden" name="id" id="edit_auto_id">
                        <p><label>Marca:</label><br><input type="text" name="marca" id="edit_auto_marca" required style="width:100%; padding:6px;"></p>
                        <p><label>Modelo:</label><br><input type="text" name="modelo" id="edit_auto_modelo" required style="width:100%; padding:6px;"></p>
                        <p><label>Año:</label><br><input type="number" name="anio" id="edit_auto_anio" required style="width:100%; padding:6px;"></p>
                        <p><label>Kilometraje:</label><br><input type="number" name="km" id="edit_auto_km" required style="width:100%; padding:6px;"></p>
                        <button type="submit" class="btn-primary" style="width:100%; padding:8px;">Guardar Cambios</button>
                    </form>
                </div>
            </div>

            <!-- MODAL EDITAR VENTA -->
            <div id="modalVenta" class="modal">
                <div class="modal-content">
                    <span class="close" onclick="cerrarModalVenta()">&times;</span>
                    <h3>Modificar Venta</h3>
                    <form method="POST">
                        <input type="hidden" name="action" value="modificar_venta">
                        <input type="hidden" name="id" id="edit_venta_id">
                        <p><label>Sucursal:</label><br>
                           <select name="sucursal" id="edit_venta_sucursal" style="width:100%; padding:6px;">
                               <option value="Norte">Norte</option>
                               <option value="Sur">Sur</option>
                               <option value="Centro">Centro</option>
                           </select>
                        </p>
                        <p><label>Mes:</label><br><input type="text" name="mes" id="edit_venta_mes" required style="width:100%; padding:6px;"></p>
                        <p><label>Cantidad:</label><br><input type="number" name="cantidad" id="edit_venta_cantidad" required style="width:100%; padding:6px;"></p>
                        <p><label>Precio Unitario ($):</label><br><input type="number" name="precio" id="edit_venta_precio" required style="width:100%; padding:6px;"></p>
                        <button type="submit" class="btn-primary" style="width:100%; padding:8px;">Guardar Cambios</button>
                    </form>
                </div>
            </div>
        </body>
    </html>
    """
    return html_output

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
