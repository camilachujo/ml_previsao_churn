# Modelo ML para Previsão de Churn

## Sobre

Utilizando uma base de dados disponível no <a href="https://www.kaggle.com/datasets/shubh0799/churn-modelling">Kaggle</a>, este projeto tem como objetivo testar modelos supervisionados de Machine Learning para o seguinte problema de classificação: prever se um cliente deixará ou não de utilizar os serviços de uma instituição bancária.

Etapas:

1. Análise exploratória dos dados
2. Limpeza, transformação e normalização dos dados
3. Validação de modelos com datatset desbalanceado
4. Validação de modelos com datatset balanceado (técnica SMOTEENN)
5. Selecionar o modelo com melhor desempenho e gerar o arquivo pickle do mesmo
6. Rodar aplicação (Streamlit) para a previsão em novos conjuntos de dados

## Configuração

Para instalar as dependências requeridas:

`pip install -r requirements.txt`

Para executar a aplicação:

`streamlit run app.py`