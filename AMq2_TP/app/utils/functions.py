import streamlit as st
import pandas as pd

# Esta función simula la extracción de características de la imagen radiológica subida
# Dado que no es el alcance de la materia, simulamos la obtención de features a partir de la imagen
# con una muestra aleatoria para que sirva de input para la predicción.

def extract_features():

    # Especificar la ruta del archivo CSV
    file_path = './utils/sample_values.csv'  

    # Leer el archivo CSV en un DataFrame
    try:
        df = pd.read_csv(file_path, sep=',', header=0, decimal='.')

    except FileNotFoundError:
        st.error(f"Archivo no encontrado: {file_path}")
    except Exception as e:
        st.error(f"Ocurrió un error inesperado: {e}")

  
    df = df.apply(pd.to_numeric, errors='coerce')
    df_cleaned = df.dropna()
  
    # Tomar una fila de muestra aleatoria
    sample_values = df.sample(n=1).iloc[0]
    
    st.session_state.radius_1 = sample_values['mean radius']
    st.session_state.texture_1 = sample_values['mean texture']
    st.session_state.perimeter_1 = sample_values['mean perimeter']
    st.session_state.area_1 = sample_values['mean area']
    st.session_state.smoothness_1 = sample_values['mean smoothness']
    st.session_state.compactness_1 = sample_values['mean compactness']
    st.session_state.concavity_1 = sample_values['mean concavity']
    st.session_state.concave_points_1 = sample_values['mean concave points']
    st.session_state.symetry_1 = sample_values['mean symmetry']
    st.session_state.fractal_dimension_1 = sample_values['mean fractal dimension']
    
    st.session_state.radius_2 = sample_values['radius error']
    st.session_state.texture_2 = sample_values['texture error']
    st.session_state.perimeter_2 = sample_values['perimeter error']
    st.session_state.area_2 = sample_values['area error']
    st.session_state.smoothness_2 = sample_values['smoothness error']
    st.session_state.compactness_2 = sample_values['compactness error']
    st.session_state.concavity_2 = sample_values['concavity error']
    st.session_state.concave_points_2 = sample_values['concave points error']
    st.session_state.symetry_2 = sample_values['symmetry error']
    st.session_state.fractal_dimension_2 = sample_values['fractal dimension error']
    
    st.session_state.radius_3 = sample_values['worst radius']
    st.session_state.texture_3 = sample_values['worst texture']
    st.session_state.perimeter_3 = sample_values['worst perimeter']
    st.session_state.area_3 = sample_values['worst area']
    st.session_state.smoothness_3 = sample_values['worst smoothness']
    st.session_state.compactness_3 = sample_values['worst compactness']
    st.session_state.concavity_3 = sample_values['worst concavity']
    st.session_state.concave_points_3 = sample_values['worst concave points']
    st.session_state.symetry_3 = sample_values['worst symmetry']
    st.session_state.fractal_dimension_3 = sample_values['worst fractal dimension']

