import streamlit as st
import pandas as pd
from Recomendacao_Filmes import recomendar_filmes, filmes_final, sim

st.set_page_config(page_title="Recomendador de Filmes", layout="centered")

st.title("🎬")
st.title("Recomendador de Filmes")
st.markdown("Insira o título de um filme para receber recomendações similares (Em Inglês).")

# Caixa de seleção com os títulos
opcoes = filmes_final['title'].sort_values().unique()
titulo_escolhido = st.selectbox("Selecione um filme", opcoes)

# Botão para recomendar
if st.button("Recomendar"):
    resultado = recomendar_filmes(titulo_escolhido, filmes_final, sim)

    if 'Erro' in resultado.columns:
        st.error(resultado['Erro'][0])
    else:
        st.success("Filmes recomendados:")
        st.dataframe(resultado)