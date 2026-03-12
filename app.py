import streamlit as st;
from pandas import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


""" 
Regressão Linear > É um modelo de aprendizado supervisionado, o que significa que ele aprendeu a 
relação entre os dados históricos (diâmetro da pizza) e os resultados conhecidos (preço).
"""


df = pd.read_csv('pizzas.csv')

modelo = LinearRegression()
x =  df[['diametro']]
y = df[['preco']]

"""
Para que ele serve?   A Regressão Linear serve para prever um valor numérico contínuo (regressão) 
com base na relação linear entre variáveis.No nosso caso específico, o Objetivo seria: Prever o 
preço (variável dependente y) baseado no tamanho (variável independente x ).

Funcionamento: O algoritmo tenta traçar a "melhor linha reta" que passa entre os pontos de dados do 
seu arquivo pizzas.csv. 
Essa linha minimiza a distância entre ela e todos os pontos reais do gráfico.

"""



modelo.fit(x,y)

st.title('Previsor de Preço de Pizza')
#st.write('### Dados de Treinamento')
st.divider()

diametro = st.number_input('Diametro da Pizza (cm)', min_value=0, value=20, step=1, key='diametro_input')
st.divider()
 
if  diametro > 0:
    preco_previsto = modelo .predict([[diametro]])[0][0]
    st.write(f'O valor da pizza com diametro de {diametro} cm é de R$ {preco_previsto:.2f}')