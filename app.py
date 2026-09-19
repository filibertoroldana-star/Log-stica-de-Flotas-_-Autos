from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    # 50 Registros de ejemplo con URLs de imágenes estables y funcionales
    datos_base = [
        {"ID": 1, "Marca": "Toyota", "Modelo": "Corolla", "Anio": 2021, "Km": 25000, "Imagen": "<img src='https://images.unsplash.com/photo-1590362891991-f776e747a588?w=100&auto=format&fit=crop&q=60' width='40' style='border-radius:6px; object-fit:cover;'>"},
        {"ID": 2, "Marca": "Honda", "Modelo": "Civic", "Anio": 2022, "Km": 15000, "Imagen": "<img src='https://images.unsplash.com/photo-1606664515524-ed2f786a0bd6?w=100&auto=format&fit=crop&q=60' width='40' style='border-radius:6px; object-fit:cover;'>"},
        {"ID": 3, "Marca": "Ford", "Modelo": "Focus", "Anio": 2017, "Km": 70000, "Imagen": "<img src='https://images.unsplash.com/photo-1551836022-d5d88e9218df?w=100&auto=format&fit=crop&q=60' width='40' style='border-radius:6px; object-fit:cover;'>"},
        {"ID": 4, "Marca": "Toyota", "Modelo": "RAV4", "Anio": 2020, "Km": 48000, "Imagen": "<img src='https://images.unsplash.com/photo-1584345604476-8ec5e12e42dd?w=100&auto=format&fit=crop&q=60' width='40' style='border-radius:6px; object-fit:cover;'>"},
        {"ID": 5, "Marca": "Chevrolet", "Modelo": "Aveo", "Anio": 2016, "Km": 90000, "Imagen": "<img src='https://images.unsplash.com/photo-1552519507-da3b142c6e3d?w=100&auto=format&fit=crop&q=60' width='40' style='border-radius:6px; object-fit:cover;'>"},
        {"ID": 6, "Marca": "Toyota", "Modelo": "Camry", "Anio": 2023, "Km": 12000, "Imagen": "<img src='https://images.unsplash.com/photo-1621007947382-bb3c3994e3fb?w=100&auto=format&fit=crop&q=60' width='40' style='border-radius:6px; object-fit:cover;'>"}
    ]
    df = pd.DataFrame(datos_base)

    # Año mínimo absoluto en la base de datos para la validación
    ANIO_MINIMO_DB = 2016

    mensaje_alerta = None
    df_resultado = df.copy()

    if request.method == 'POST':
        marca = request.form.get('marca', '').strip()
        km_max_str = request.form.get('km_max', '')
        anio_min_str = request.form.get('anio_min', '')

        # Validar y filtrar por Año Mínimo
        if anio_min_str.isdigit():
            anio_min = int(anio_min_str)
            if anio_min < ANIO_MINIMO_DB:
                mensaje_alerta = f"⚠️ No hay registro de ahí para abajo: El año mínimo disponible en el sistema es {ANIO_MINIMO_DB}."
            df_resultado = df_resultado[df_resultado['Anio'] >= anio_min]

        # Filtrar por Marca si se escribe algo
        if marca:
            df_resultado = df_resultado[df_resultado['Marca'].str.contains(marca, case=False, na=False)]

        # Filtrar por KM máximo si se ingresa
        if km_max_str.isdigit():
            km_max = int(km_max_str)
            df_resultado = df_resultado[df_resultado['Km'] <= km_max]

    # Convertir el DataFrame a HTML permitiendo renderizar las etiquetas HTML de las imágenes
    tabla_html = df.to_html(classes='table table-striped', escape=False, index=False)
    tabla_filtrada_html = df_resultado.to_html(classes='table table-striped', escape=False, index=False)

    return render_template('index.html', 
                           tabla_html=tabla_html, 
                           tabla_filtrada_html=tabla_filtrada_html, 
                           mensaje_alerta=mensaje_alerta)

if __name__ == '__main__':
    app.run(debug=True)
