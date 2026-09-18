from flask import Flask
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

app = Flask(__name__)

@app.route("/")
def home():
    # Aquí puedes ejecutar tu lógica o retornar un texto con los resultados
    # Ejemplo rápido del Módulo 3:
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svc_model = SVC()
    param_grid = {'C': [0.5, 1, 5, 10], 'kernel': ['linear', 'poly']}
    grid_search = GridSearchCV(estimator=svc_model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    return f"¡Pipeline del Equipo D ejecutado con éxito! <br> Mejor Accuracy: {grid_search.score(X_test, y_test):.4f}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)