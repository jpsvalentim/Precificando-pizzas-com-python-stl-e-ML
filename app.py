import streamlit as st;
from pandas import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('pizzas.csv')

modelo = LinearRegression()
x =  df[['diametro']]
y = df[['preco']]

modelo.fit(x,y)

st.title('Previsor de Preço de Pizza')
#st.write('### Dados de Treinamento')
st.divider()

diametro = st.number_input('Diametro da Pizza (cm)', min_value=0, value=20, step=1, key='diametro_input')
st.divider()
 
if  diametro > 0:
    preco_previsto = modelo .predict([[diametro]])[0][0]
    st.write(f'O valor da pizza com diametro de {diametro} cm é de R$ {preco_previsto:.2f}')