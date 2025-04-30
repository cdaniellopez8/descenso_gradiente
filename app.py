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

# --- Descenso de gradiente ---
def gradient_descent(lr, steps, x0, y0, max_val=1e5):
    path = [(x0, y0, f(x0, y0))]
    x, y = x0, y0
    for _ in range(steps):
        grad = grad_f(x, y)
        x -= lr * grad[0]
        y -= lr * grad[1]
        z = f(x, y)
        if abs(x) > max_val or abs(y) > max_val or abs(z) > max_val:
            st.warning("¡La trayectoria diverge con este learning rate! Reduce la tasa de aprendizaje.")
            break
        path.append((x, y, z))
    return np.array(path)

# --- Interfaz Streamlit ---
st.title("Simulador de Descenso de Gradiente en 3D")
st.markdown("### Función: $f(x, y) = x^2 + y^2$")

# Sidebar inputs
st.sidebar.header("Parámetros de simulación")
x0 = st.sidebar.number_input("Valor inicial de x", value=3.0)
y0 = st.sidebar.number_input("Valor inicial de y", value=3.0)
learning_rate = st.sidebar.number_input("Tasa de aprendizaje", value=0.1, min_value=0.001, step=0.01)
steps = st.sidebar.slider("Número de pasos", min_value=5, max_value=100, value=30, step=1)

with st.sidebar:
    st.markdown("""
    <hr>
    <div style="text-align: center; font-size: 0.9em; color: gray;">
        Desarrollado por Carlos D. López P.
    </div>
    """, unsafe_allow_html=True)

# Calcular trayectoria
path = gradient_descent(learning_rate, steps, x0, y0)

# Generar superficie con margen dinámico ajustado
if path.shape[0] > 1:
    x_center = np.mean(path[:, 0])
    y_center = np.mean(path[:, 1])

    max_range = max(
        abs(path[:, 0].max() - path[:, 0].min()),
        abs(path[:, 1].max() - path[:, 1].min()),
    )
    spread = max(max_range * 1.2, 3.5)  # margen más ajustado

    x_min, x_max = x_center - spread, x_center + spread
    y_min, y_max = y_center - spread, y_center + spread

    # Crear malla
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # Crear figura
    fig = go.Figure()

    # Superficie
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', opacity=0.7, showscale=False))

    # Trayectoria
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

    # Ajustar eje z con margen
    z_margin = 1.0
    z_min = min(path[:,2].min(), Z.min()) - z_margin
    z_max = max(path[:,2].max(), Z.max()) + z_margin

    # Layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='x', range=[x_min, x_max]),
            yaxis=dict(title='y', range=[y_min, y_max]),
            zaxis=dict(title='f(x, y)', range=[z_min, z_max])
        ),
        title="Descenso de Gradiente en f(x, y) = x² + y²",
        width=800,
        height=600
    )

    st.plotly_chart(fig)
