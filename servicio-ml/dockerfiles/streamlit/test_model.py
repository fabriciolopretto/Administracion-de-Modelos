import streamlit as st
import pydeck as pdk
import numpy as np
import requests
from utils.utils import load_location_info

# Mapeo de direcciones a ángulos en grados
direccion_to_angulo = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5,
    'SE': 135, 'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
}

def codificar_direccion(direccion):
    """
    Codifica una dirección de viento en los componentes seno y coseno.
    """
    angulo = direccion_to_angulo.get(direccion, None)
    if angulo is None:
        return None, None
    angulo_rad = np.deg2rad(angulo)
    return np.sin(angulo_rad), np.cos(angulo_rad)

def show():
    # Obtener información de las ubicaciones
    location_info = load_location_info()

    st.title("Modelo de Predicción de Clima")

    # Crear una columna para centrar el contenido
    col1, col2 = st.columns([1, 3])  # Ajustar las proporciones para centrar el contenido

    with col2:
        # Selección de ubicación y fecha
        st.subheader("Seleccione Fecha y Ubicación")
        selected_location = st.selectbox('Seleccione Ubicación', list(location_info.keys()), key="location_select")
        selected_date = st.date_input('Seleccione Fecha', key="date_select")

        # Deslizadores para las variables del modelo
        st.subheader('Ajuste de Parámetros del Modelo')
        
        # Variables obligatorias
        min_temp = st.slider('MinTemp', min_value=-10.0, max_value=50.0, value=12.3, format="%.1f")
        max_temp = st.slider('MaxTemp', min_value=-10.0, max_value=50.0, value=24.7, format="%.1f")
        rainfall = st.slider('Rainfall', min_value=0.0, max_value=100.0, value=0.0, format="%.1f")
        evaporation = st.slider('Evaporation', min_value=0.0, max_value=20.0, value=5.6, format="%.1f")
        sunshine = st.slider('Sunshine', min_value=0.0, max_value=15.0, value=9.8, format="%.1f")
        wind_gust_speed = st.slider('WindGustSpeed', min_value=0.0, max_value=100.0, value=35.0, format="%.1f")
        wind_speed9am = st.slider('WindSpeed9am', min_value=0.0, max_value=100.0, value=20.0, format="%.1f")
        wind_speed3pm = st.slider('WindSpeed3pm', min_value=0.0, max_value=100.0, value=24.0, format="%.1f")
        humidity9am = st.slider('Humidity9am', min_value=0.0, max_value=100.0, value=82.0, format="%.1f")
        humidity3pm = st.slider('Humidity3pm', min_value=0.0, max_value=100.0, value=55.0, format="%.1f")
        cloud9am = st.slider('Cloud9am', min_value=0.0, max_value=9.0, value=4.0, format="%.1f")
        cloud3pm = st.slider('Cloud3pm', min_value=0.0, max_value=9.0, value=3.0, format="%.1f")
        rain_today = st.slider('RainToday', min_value=0, max_value=1, value=0, format="%d")
        
        # Variables opcionales
        pressure9am = st.slider('Pressure9am', min_value=950.0, max_value=1050.0, value=None, format="%.1f")
        pressure3pm = st.slider('Pressure3pm', min_value=950.0, max_value=1050.0, value=None, format="%.1f")
        temp9am = st.slider('Temp9am', min_value=-10.0, max_value=50.0, value=None, format="%.1f")
        temp3pm = st.slider('Temp3pm', min_value=-10.0, max_value=50.0, value=None, format="%.1f")

        # Selección de dirección del viento
        wind_gust_dir = st.selectbox('Dirección del Viento (WindGustDir)', options=list(direccion_to_angulo.keys()), key="wind_gust_dir")
        wind_dir9am = st.selectbox('Dirección del Viento a las 9am (WindDir9am)', options=list(direccion_to_angulo.keys()), key="wind_dir9am")
        wind_dir3pm = st.selectbox('Dirección del Viento a las 3pm (WindDir3pm)', options=list(direccion_to_angulo.keys()), key="wind_dir3pm")

        # Calcular valores de seno y coseno para las direcciones del viento
        wind_gust_dir_sin, wind_gust_dir_cos = codificar_direccion(wind_gust_dir)
        wind_dir9am_sin, wind_dir9am_cos = codificar_direccion(wind_dir9am)
        wind_dir3pm_sin, wind_dir3pm_cos = codificar_direccion(wind_dir3pm)

        # Enviar los datos con el botón de predicción
        if st.button("Predecir"):
            with st.spinner('Realizando la predicción...'):
                # Obtener las coordenadas de la ubicación seleccionada
                latitude, longitude = location_info[selected_location]

                # Preparar el payload para la solicitud
                payload = {
                        "features": {
                        "MinTemp": min_temp,
                        "MaxTemp": max_temp,
                        "Rainfall": rainfall,
                        "Evaporation": evaporation,
                        "Sunshine": sunshine,
                        "WindGustSpeed": wind_gust_speed,
                        "WindSpeed9am": wind_speed9am,
                        "WindSpeed3pm": wind_speed3pm,
                        "Humidity9am": humidity9am,
                        "Humidity3pm": humidity3pm,
                        "Pressure9am":pressure9am,
                        "Pressure3pm":pressure3pm,
                        "Cloud9am": cloud9am,
                        "Cloud3pm": cloud3pm,
                        "Temp9am":temp9am,
                        "Temp3pm":temp3pm,
                        "RainToday": rain_today,
                        "Year": selected_date.year,
                        "Month": selected_date.month,
                        "Day": selected_date.day,
                        "Latitude": latitude,
                        "Longitude": longitude,
                        "WindGustDir_sin": wind_gust_dir_sin,
                        "WindGustDir_cos": wind_gust_dir_cos,
                        "WindDir9am_sin": wind_dir9am_sin,
                        "WindDir9am_cos": wind_dir9am_cos,
                        "WindDir3pm_sin": wind_dir3pm_sin,
                        "WindDir3pm_cos": wind_dir3pm_cos
                    }
                }

                # Enviar la solicitud POST usando el paquete `requests`
                try:
                    print(f"PAYLOAD: {payload}")
                    response = requests.post("http://fastapi:8800/predict/", json=payload)
                    response.raise_for_status()
                    prediction = response.json()
                except requests.exceptions.RequestException as e:
                    st.error(f"Error al realizar la solicitud: {e}")
                    return
                # Mostrar los resultados
                st.subheader(f"Predicción para el {selected_date} en la ubicación {selected_location}:")
                # Determinar el mensaje basado en la predicción
                if prediction.get('int_output', False):
                    st.write("¿Lloverá mañana? 🌧️ Sí, lloverá mañana ☔")
                else:
                    st.write("¿Lloverá mañana? 🌞 No lloverá mañana 😊")

                # Visualización de los resultados en el mapa
                view_state = pdk.ViewState(
                    latitude=latitude,
                    longitude=longitude,
                    zoom=8,
                    bearing=0,
                    pitch=0,
                )

                # Capa del mapa
                scatter_layer_selected = pdk.Layer(
                    'ScatterplotLayer',
                    data=[{"latitude": latitude, "longitude": longitude}],
                    get_position='[longitude, latitude]',
                    get_radius=8000,
                    get_fill_color=[255, 0, 0] if prediction.get('result') == "Yes" else [0, 255, 0],
                    opacity=0.8,
                )

                r = pdk.Deck(layers=[scatter_layer_selected], initial_view_state=view_state,
                             map_provider="mapbox", map_style=pdk.map_styles.SATELLITE)

                # Mostrar el mapa
                st.pydeck_chart(r)



