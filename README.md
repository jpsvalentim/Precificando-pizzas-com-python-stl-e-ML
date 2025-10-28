# 🍕 Previsor de Preço de Pizza

Um projeto simples de **Machine Learning com Streamlit**, desenvolvido em **Python 3.13**, que prevê o preço de uma pizza com base no seu diâmetro.  
O modelo utiliza **Regressão Linear** com `scikit-learn` e uma interface interativa em **Streamlit** para a previsão.

---

## 🚀 Tecnologias Utilizadas

- [Python 3.13](https://www.python.org/)
- [Poetry](https://python-poetry.org/) — Gerenciador de dependências e ambiente virtual
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Streamlit](https://streamlit.io/)
- [Jupyter Notebook](https://jupyter.org/) — para experimentos e testes do modelo

---

## 📂 Estrutura do Projeto

```bash
project-ml/
│
├── app.py               # Aplicação principal em Streamlit
├── pizzas.csv           # Base de dados com diâmetro e preço das pizzas
├── testes.ipynb         # Notebook usado para treinar e validar o modelo
├── pyproject.toml       # Arquivo de configuração do Poetry
├── README.md            # Este arquivo
└── .gitignore           # Arquivos ignorados pelo Git
