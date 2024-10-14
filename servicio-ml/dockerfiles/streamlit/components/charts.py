import streamlit as st
import matplotlib.pyplot as plt

def plot_chart(data):
    fig, ax = plt.subplots()
    data.plot(ax=ax)
    st.pyplot(fig)
