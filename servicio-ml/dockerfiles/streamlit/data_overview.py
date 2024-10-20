import streamlit as st
import awswrangler as wr
import pandas as pd
import os

def show():
    # Configurar las variables de entorno
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_endpoint_url_s3 = os.getenv('AWS_ENDPOINT_URL_S3')
    aws_endpoint_url_s3 = "http://s3:9000"

    # Configurar las credenciales
    wr.config.s3_endpoint_url = aws_endpoint_url_s3

    # Especificar el bucket y el objeto en S3
    bucket_name = 'data'
    object_key = 'raw/weatherAUS_corregido.csv'
    s3_path = f's3://{bucket_name}/{object_key}'


    try:
        # Leer el archivo desde S3 usando awswrangler
        df = wr.s3.read_csv(path=s3_path)
        data_loaded = True
    except Exception as e:
        st.markdown("""
        # Datos no cargados

        El archivo del conjunto de datos aún no se ha cargado. Asegúrate de que Airflow se esté ejecutando y de que los datos estén disponibles en el bucket de S3.

        Puedes probar con un modelo base. Si quieres ver la página de prueba de modelo, revisa la pagina *Test Model*.

        ### Detalles del Error
        ```plaintext
        {}
        ```
        """.format(e))
        data_loaded = False

    if data_loaded:
        # Resto del código sigue igual
        st.sidebar.title("Data Overview")

        selected_section = st.sidebar.radio("Selección", [
            "Visión General de Datos",
        ])
        
        if selected_section == "Visión General de Datos":
            st.sidebar.title('Visión General de Datos')
            section = st.sidebar.selectbox("Seleccionar Sub-Sección", [
                "Conjunto de Datos", "Valores Únicos", "Tipos de Datos", 
                "Estadísticas Resumidas", "Observaciones y Variables", "Valores Faltantes"
            ])
            
            if section == "Conjunto de Datos":
                st.title('Visión General')
                st.markdown("""
                El conjunto de datos Weather Australia es una colección de observaciones meteorológicas históricas registradas en varias estaciones meteorológicas en Australia entre 2007 y 2017. Contiene un conjunto completo de variables meteorológicas, incluyendo temperatura, lluvia, humedad, velocidad del viento, presión atmosférica, y más. El conjunto de datos es ampliamente utilizado por investigadores, meteorólogos y científicos de datos para análisis meteorológico, pronósticos y estudios climáticos.
                """)

                st.title('Características')
                st.markdown("""
                | Característica        | Descripción                                                           |
                |-----------------------|-----------------------------------------------------------------------|
                | **Date**              | Fecha de la observación meteorológica.                                |
                | **Location**          | Nombre o código de la estación meteorológica.                         |
                | **MinTemp**           | Temperatura mínima registrada (en grados Celsius).                    |
                | **MaxTemp**           | Temperatura máxima registrada (en grados Celsius).                    |
                | **Rainfall**          | Cantidad de lluvia registrada (en milímetros).                        |
                | **Evaporation**       | Evaporación del agua (en milímetros).                                 |
                | **Sunshine**          | Horas de sol brillante registradas.                                   |
                | **WindGustDir**       | Dirección de la ráfaga de viento más fuerte.                          |
                | **WindGustSpeed**     | Velocidad de la ráfaga de viento más fuerte (en kilómetros por hora). |
                | **WindDir9am**        | Dirección del viento a las 9 am.                                      |
                | **WindDir3pm**        | Dirección del viento a las 3 pm.                                      |
                | **WindSpeed9am**      | Velocidad del viento a las 9 am (en kilómetros por hora).             |
                | **WindSpeed3pm**      | Velocidad del viento a las 3 pm (en kilómetros por hora).             |
                | **Humidity9am**       | Humedad relativa a las 9 am (en porcentaje).                          |
                | **Humidity3pm**       | Humedad relativa a las 3 pm (en porcentaje).                          |
                | **Pressure9am**       | Presión atmosférica a las 9 am (en hPa).                              |
                | **Pressure3pm**       | Presión atmosférica a las 3 pm (en hPa).                              |
                | **Cloud9am**          | Cobertura de nubes a las 9 am (en octas).                             |
                | **Cloud3pm**          | Cobertura de nubes a las 3 pm (en octas).                             |
                | **Temp9am**           | Temperatura a las 9 am (en grados Celsius).                           |
                | **Temp3pm**           | Temperatura a las 3 pm (en grados Celsius).                           |
                | **RainToday**         | Variable binaria que indica si llovió hoy (1 para "Sí", 0 para "No"). |
                | **RainTomorrow**      | Variable binaria objetivo que indica si lloverá mañana (1 para "Sí", 0 para "No"). |
                """)

                st.title('Uso')
                st.markdown("""
                Siéntete libre de usar este conjunto de datos para investigación, análisis o proyectos de aprendizaje automático. Asegúrate de citar la fuente adecuadamente.
                """)

                st.title('Fuentes de Datos')
                st.markdown("""
                - Oficina de Meteorología de Australia (BOM) - Observaciones Diarias del Clima  
                - Oficina de Meteorología de Australia (BOM) - Datos Climáticos en Línea  
                - API de Archivo de Open Meteo  
                """)

                st.title('Modelos Utilizados')
                st.markdown("""
                - **Regresión Logística:** Un modelo lineal simple utilizado para tareas de clasificación binaria.  
                - **Clasificador Random Forest:** Un método de aprendizaje en conjunto basado en árboles de decisión, conocido por su robustez y precisión.  
                - **LightGBM:** Un marco de boosting de gradiente que utiliza algoritmos de aprendizaje basados en árboles y está optimizado para velocidad y eficiencia.
                """)

                st.dataframe(df.head(), use_container_width=True)
            
            elif section == "Valores Únicos":
                st.write("## Número de Valores Únicos")
                st.dataframe(df.nunique(), use_container_width=True)

            elif section == "Tipos de Datos":
                st.write("## Tipos de Datos")
                st.dataframe(df.dtypes, use_container_width=True)

            elif section == "Estadísticas Resumidas":
                st.write("## Estadísticas Resumidas")
                st.dataframe(df.describe(), use_container_width=True)

            elif section == "Observaciones y Variables":
                st.write("## Número de Observaciones y Variables")
                st.write(f"Número de observaciones: {df.shape[0]}")
                st.write(f"Número de variables: {df.shape[1]}")

            elif section == "Valores Faltantes":
                st.write("## Consulta de Valores Faltantes")
                st.write("### ¿Alguna Columna Tiene Valores Faltantes?")
                st.dataframe(df.isnull().any(), use_container_width=True)

                st.write("### Conteo de Valores Faltantes en Cada Columna")
                st.dataframe(df.isna().sum(), use_container_width=True)
