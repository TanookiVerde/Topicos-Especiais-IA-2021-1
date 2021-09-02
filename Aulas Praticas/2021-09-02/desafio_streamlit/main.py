from numpy.random.mtrand import f
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LinearRegression
from mlxtend.evaluate import bias_variance_decomp
from sklearn.model_selection import train_test_split

st.write(
""" 
# Tópicos Especiais em IA - Aula 02/09/2021

- Este app permite que o usuário selecione um atributo e um alvo para treinamento de um modelo de Regressão Linear
""")

dataset = datasets.load_boston()
boston = pd.DataFrame(dataset.data, columns=dataset.feature_names)
 
st.sidebar.header("Parâmetros")

def input_feature_selection():
    X_name = st.sidebar.selectbox("Atributo", options=list(boston.columns))
    y_name = st.sidebar.selectbox("Alvo", options=list(boston.columns))

    return X_name, y_name
 
def input_features(feature):
    data = dict()

    _min = float(boston.describe()[feature][3])
    _max = float(boston.describe()[feature][7])

    data[feature] = st.sidebar.slider(feature, min_value=_min, max_value=_max)

    return pd.DataFrame(data, index=[0])

X_name, y_name = input_feature_selection()
df = input_features(X_name)
 
st.subheader("Parâmetros inseridos")
st.write(df)

X = boston[[X_name]].to_numpy()
y = boston[y_name].to_numpy()

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state =1)
model_lr = LinearRegression()
mse, bias, variance = bias_variance_decomp(model_lr, X_train, y_train, X_test, y_test, loss = "mse", num_rounds = 200, random_seed=123)

st.subheader("Avaliação do Modelo:")
ava = pd.DataFrame({
    'MSE': mse,
    'Viés': bias,
    'Variância': variance
}, index=['Valor'])

st.write(ava.T)
 
prediction = model_lr.predict(df)
 
st.subheader("Predição do Modelo")
st.write(prediction)