import streamlit as st
import data_overview, test_model

st.sidebar.title("Aprendizaje de Máquinas 2")

# Mensaje de integrantes del grupo
integrantes = """
### Integrantes del Grupo
- Julio Agustín Donadello
- Diego Braga
- Eduardo Echeverria
- Marco Joel Isidro
- Diego Sarina
"""

st.sidebar.markdown(integrantes)

# Diccionario que mapea los nombres de las páginas con sus respectivos módulos
PAGES = {
    "Data Overview": data_overview,
    "Test Model": test_model
}

def main():

    # Seleccionar la página
    selection = st.sidebar.selectbox("Selecciona la página", list(PAGES.keys()))

    # Llamar a la función show() de la página seleccionada
    page = PAGES[selection]
    page.show()

if __name__ == "__main__":
    main()
