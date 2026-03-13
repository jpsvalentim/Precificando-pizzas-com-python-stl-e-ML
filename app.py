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



previsoes = modelo.predict(x)
r2 = r2_score(y, previsoes )
st.write(f'precisão do modelo R²: {r2:.2%}')



"""
Como medir se ele está dando certo?Existem métricas específicas para saber  se o seu modelo está "mentindo" 
para você ou se ele realmente aprendeu o padrão.

 As três principais são:
 
 - A. R² (R-Quadrado ou Coeficiente de Determinação)É a métrica mais comum. 
 Ela varia de 0 a 1 (ou 0% a 100%).O que diz: "Quanto da variação do preço é explicada pelo diâmetro?
 "Se o $R^2 = 0.90$, significa que o diâmetro explica 90% do preço da pizza.B. 
 
 - MAE (Erro Médio Absoluto)O que diz: "Em média, quantos Reais o modelo erra para cima ou para baixo?
 "Se o MAE for 2.50, seu modelo erra, em média, R$ 2,50 por previsão.C. 
 
 - Teste Prático (Split Treino/Teste)Atualmente, seu modelo está treinando e testando com os mesmos dados.
Para saber se ele funciona no "mundo real", o ideal é separar uma parte dos dados (ex: 20%) para 
testar o modelo após ele ser treinado com os outros 80%.
"""