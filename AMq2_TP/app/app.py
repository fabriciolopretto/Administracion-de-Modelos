import streamlit as st
import pandas as pd
import numpy as np
import requests 
from utils.functions import extract_features
import matplotlib.pyplot as plt


st.markdown("<h1 style='font-size: 24px; color: blue;'>Modelo de clasificación de tumores de cáncer de mama</h1>", unsafe_allow_html=True)

st.markdown("<p style='font-size: 18px;'><i>Este modelo evalúa las características del tumor para predecir si es maligno o benigno.</i></p>", unsafe_allow_html=True)

st.markdown("<h1 style='font-size: 20px;'>Proporcione la imagen del tumor o ingrese los datos:</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Subir archivo", type='png')

if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False
    
# Inicializar datos
if 'radius_1' not in st.session_state:
    st.session_state.radius_1 = 0.0
if 'texture_1' not in st.session_state:
    st.session_state.texture_1 = 0.0
if 'perimeter_1' not in st.session_state:
    st.session_state.perimeter_1 = 0.0
if 'area_1' not in st.session_state:
    st.session_state.area_1 = 0.0
if 'smoothness_1' not in st.session_state:
    st.session_state.smoothness_1 = 0.0
if 'compactness_1' not in st.session_state:
    st.session_state.compactness_1 = 0.0
if 'concavity_1' not in st.session_state:
    st.session_state.concavity_1 = 0.0
if 'concave_points_1' not in st.session_state:
    st.session_state.concave_points_1 = 0.0
if 'symetry_1' not in st.session_state:
    st.session_state.symetry_1 = 0.0
if 'fractal_dimension_1' not in st.session_state:
    st.session_state.fractal_dimension_1 = 0.0
    
if 'radius_2' not in st.session_state:
    st.session_state.radius_2 = 0.0
if 'texture_2' not in st.session_state:
    st.session_state.texture_2 = 0.0
if 'perimeter_2' not in st.session_state:
    st.session_state.perimeter_2 = 0.0
if 'area_2' not in st.session_state:
    st.session_state.area_2 = 0.0
if 'smoothness_2' not in st.session_state:
    st.session_state.smoothness_2 = 0.0
if 'compactness_2' not in st.session_state:
    st.session_state.compactness_2 = 0.0
if 'concavity_2' not in st.session_state:
    st.session_state.concavity_2 = 0.0
if 'concave_points_2' not in st.session_state:
    st.session_state.concave_points_2 = 0.0
if 'symetry_2' not in st.session_state:
    st.session_state.symetry_2 = 0.0
if 'fractal_dimension_2' not in st.session_state:
    st.session_state.fractal_dimension_2 = 0.0
    
if 'radius_3' not in st.session_state:
    st.session_state.radius_3 = 0.0
if 'texture_3' not in st.session_state:
    st.session_state.texture_3 = 0.0
if 'perimeter_3' not in st.session_state:
    st.session_state.perimeter_3 = 0.0
if 'area_3' not in st.session_state:
    st.session_state.area_3 = 0.0
if 'smoothness_3' not in st.session_state:
    st.session_state.smoothness_3 = 0.0
if 'compactness_3' not in st.session_state:
    st.session_state.compactness_3 = 0.0
if 'concavity_3' not in st.session_state:
    st.session_state.concavity_3 = 0.0
if 'concave_points_3' not in st.session_state:
    st.session_state.concave_points_3 = 0.0
if 'symetry_3' not in st.session_state:
    st.session_state.symetry_3 = 0.0
if 'fractal_dimension_3' not in st.session_state:
    st.session_state.fractal_dimension_3 = 0.0

if uploaded_file is not None:
# Se subió archivo
    if st.session_state.file_processed == False :
        # Simular extracción de features
        st.session_state.file_processed = True
        extract_features()
    else:
        st.session_state.file_processed = False
        
# Crear tres columnas
col1, col2, col3 = st.columns(3)

# Inputs en columnas
with col1:
    st.markdown("<p style='font-size: 16px;'><strong>Medidas promedio</strong></p>", unsafe_allow_html=True)
    radius_1 = st.number_input("Radio (medio)", min_value=0.0, value=st.session_state.radius_1, format="%.5f")
    texture_1 = st.number_input("Textura (media)", min_value=0.0, value=st.session_state.texture_1, format="%.5f")
    perimeter_1 = st.number_input("Perímetro (medio)", min_value=0.0, value=st.session_state.perimeter_1, format="%.5f")
    area_1 = st.number_input("Área (media)", min_value=0.0, value=st.session_state.area_1, format="%.5f")
    smoothness_1 = st.number_input("Suavidad (media)", min_value=0.0, value=st.session_state.smoothness_1, format="%.5f")   
    compactness_1 = st.number_input("Compacidad (media)", min_value=0.0, value=st.session_state.compactness_1, format="%.5f")
    concavity_1 = st.number_input("Concavidad (media)", min_value=0.0, value=st.session_state.concavity_1, format="%.5f")
    concave_points_1 = st.number_input("Puntos cóncavos (medios)", min_value=0.0, value=st.session_state.concave_points_1, format="%.5f")
    symetry_1 = st.number_input("Simetría (media)", min_value=0.0, value=st.session_state.symetry_1, format="%.5f")
    fractal_dimension_1 = st.number_input("Dimensión fractal (media)", min_value=0.0, value=st.session_state.fractal_dimension_1, format="%.5f")
    
with col2:
    st.markdown("<p style='font-size: 16px;'><strong>Medidas de variabilidad</strong></p>", unsafe_allow_html=True)
    radius_2 = st.number_input("Radio (desv. std.)", min_value=0.0, value=st.session_state.radius_2, format="%.5f")
    texture_2 = st.number_input("Textura (desv. std.)", min_value=0.0, value=st.session_state.texture_2, format="%.5f")
    perimeter_2 = st.number_input("Perímetro (desv. std.)", min_value=0.0, value=st.session_state.perimeter_2, format="%.5f")
    area_2 = st.number_input("Área (desv. std.)", min_value=0.0, value=st.session_state.area_2, format="%.5f")
    smoothness_2 = st.number_input("Suavidad (desv. std.)", min_value=0.0, value=st.session_state.smoothness_2, format="%.5f")   
    compactness_2 = st.number_input("Compacidad (desv. std.)", min_value=0.0, value=st.session_state.compactness_2, format="%.5f")
    concavity_2 = st.number_input("Concavidad (desv. std.)", min_value=0.0, value=st.session_state.concavity_2, format="%.5f")
    concave_points_2 = st.number_input("Puntos cóncavos (desv. std.)", min_value=0.0, value=st.session_state.concave_points_2, format="%.5f")
    symetry_2 = st.number_input("Simetría (desv. std.)", min_value=0.0, value=st.session_state.symetry_2, format="%.5f")
    fractal_dimension_2 = st.number_input("Dimensión fractal (desv. std.)", min_value=0.0, value=st.session_state.fractal_dimension_2, format="%.5f")

with col3:
    st.markdown("<p style='font-size: 16px;'><strong>Medidas de tendencia central</strong></p>", unsafe_allow_html=True)
    radius_3 = st.number_input("Radio (peor valor)", min_value=0.0, value=st.session_state.radius_3, format="%.5f")
    texture_3 = st.number_input("Textura (peor valor)", min_value=0.0, value=st.session_state.texture_3, format="%.5f")
    perimeter_3 = st.number_input("Perímetro (peor valor)", min_value=0.0, value=st.session_state.perimeter_3, format="%.5f")
    area_3 = st.number_input("Área (peor valor)", min_value=0.0, value=st.session_state.area_3, format="%.5f")
    smoothness_3 = st.number_input("Suavidad (peor valor)", min_value=0.0, value=st.session_state.smoothness_3, format="%.5f")   
    compactness_3 = st.number_input("Compacidad (peor valor)", min_value=0.0, value=st.session_state.compactness_3, format="%.5f")
    concavity_3 = st.number_input("Concavidad (peor valor)", min_value=0.0, value=st.session_state.concavity_3, format="%.5f")
    concave_points_3 = st.number_input("Puntos cóncavos (peor valor)", min_value=0.0, value=st.session_state.concave_points_3, format="%.5f")
    symetry_3 = st.number_input("Simetría (peor valor)", min_value=0.0, value=st.session_state.symetry_3, format="%.5f")
    fractal_dimension_3 = st.number_input("Dimensión fractal (peor valor)", min_value=0.0, value=st.session_state.fractal_dimension_3, format="%.5f")


    
# Botón para predecir
if st.button("PREDECIR", key="predict_button"):
    data = {
        "mean radius": radius_1,
        "mean texture": texture_1,
        "mean perimeter": perimeter_1,
        "mean area": area_1,
        "mean smoothness": smoothness_1,
        "mean compactness": compactness_1,
        "mean concavity": concavity_1,
        "mean concave points": concave_points_1,
        "mean symmetry": symetry_1,
        "mean fractal dimension": fractal_dimension_1,
        "radius error": radius_2,
        "texture error": texture_2,
        "perimeter error": perimeter_2,
        "area error": area_2,
        "smoothness error": smoothness_2,
        "compactness error": compactness_2,
        "concavity error": concavity_2,
        "concave points error": concave_points_2,
        "symmetry error": symetry_2,
        "fractal dimension error": fractal_dimension_2,
        "worst radius": radius_3,
        "worst texture": texture_3,
        "worst perimeter": perimeter_3,
        "worst area": area_3,
        "worst smoothness": smoothness_3,
        "worst compactness": compactness_3,
        "worst concavity": concavity_3,
        "worst concave points": concave_points_3,
        "worst symmetry": symetry_3,
        "worst fractal dimension": fractal_dimension_3
    }
	    
    try:
      
        response = requests.post("http://localhost:8800/predict/", json=data)
    
    	# Gráfico circular con las probabilidades resultantes de la predicción
        if response.status_code == 200:
            labels = ['Maligno', 'Benigno']
            probab = []
            json_response = response.json()
		    
            for m, b in json_response.items():
                probab.append(m)  
                probab.append(b)
                colors = ['#ff9999', '#99ff99']

                plt.figure(figsize=(10, 5))
                plt.pie(probab, labels=labels, colors=colors, autopct='%1.2f%%', startangle=140)
                plt.axis('equal')  
                st.pyplot(plt)
                plt.clf()
		    
                st.session_state.show_graph = True
		    
        else:
            st.error("Error en la predicción: " + str(response.status_code) + " " + response.reason)

    except requests.exceptions.HTTPError as err:
        st.error(f"Error en la predicción: {err}")
            
    except requests.exceptions.RequestException as e:
        st.error(f"Error de conexión: {e}")


	    
	    
	    
	     


