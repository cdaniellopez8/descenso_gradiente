import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
from io import BytesIO
import requests
from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

# --- Función objetivo y gradiente ---
def f(x, y):
    return x**2 + y**2

def grad_f(x, y):
    return np.array([2*x, 2*y])

# --- Algoritmo de descenso de gradiente ---
def gradient_descent(lr, steps, x0, y0):
    path = [(x0, y0, f(x0, y0))]
    x, y = x0, y0
    for _ in range(steps):
        grad = grad_f(x, y)
        x -= lr * grad[0]
        y -= lr * grad[1]
        path.append((x, y, f(x, y)))
    return np.array(path)

# --- Interfaz Streamlit ---
st.title("Simulador de Descenso de Gradiente en 3D con Plotly")
st.markdown("### Función: $f(x, y) = x^2 + y^2$")

# Controles
learning_rate = st.slider("Tasa de aprendizaje", min_value=0.001, max_value=5.0, value=0.1, step=0.01)
steps = st.slider("Número de pasos", min_value=5, max_value=100, value=30, step=1)
x0 = st.slider("Valor inicial x", -5.0, 5.0, value=3.0, step=0.1)
y0 = st.slider("Valor inicial y", -5.0, 5.0, value=3.0, step=0.1)

# Calcular trayectoria
path = gradient_descent(learning_rate, steps, x0, y0)

# Superficie
X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(X, Y)
Z = f(X, Y)

# Crear figura con Plotly
fig = go.Figure()

# Superficie
fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', opacity=0.7, showscale=False))

# Trayectoria de descenso
fig.add_trace(go.Scatter3d(
    x=path[:,0], y=path[:,1], z=path[:,2],
    mode='lines+markers',
    marker=dict(size=4, color='red'),
    line=dict(color='red', width=3),
    name='Descenso'
))

# Punto inicial
fig.add_trace(go.Scatter3d(
    x=[path[0,0]], y=[path[0,1]], z=[path[0,2]],
    mode='markers',
    marker=dict(size=6, color='blue'),
    name='Inicio'
))

# Punto final
fig.add_trace(go.Scatter3d(
    x=[path[-1,0]], y=[path[-1,1]], z=[path[-1,2]],
    mode='markers',
    marker=dict(size=6, color='green'),
    name='Final'
))

# Layout
fig.update_layout(
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='f(x, y)'
    ),
    title="Descenso de Gradiente en f(x, y) = x² + y²",
    width=800,
    height=600
)

st.plotly_chart(fig)
