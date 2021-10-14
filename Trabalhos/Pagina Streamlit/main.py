from numpy.random.mtrand import f
import streamlit as st
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from models import MODELS

st.write(
""" 
# Tópicos Especiais em IA - Trabalho 2
- Pedro Marques
- Tomaz Cuber

## Proposta do Trabalho
- O usuário deve ser capaz de escolher os parâmetros do algoritmo (através do uso do componente slider)
- Qual algoritmo de aprendizado de máquina será utilizado para classificar a planta de acordo com a espécie.

## Como usar
1. Seleciona o modelo
1. Configure o modelo
1. Selecione as features para o modelo
1. Crie uma entrada para simular uma planta.
1. Veja o resultado 
""")


dataset = datasets.load_iris()
X_dataset = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y_dataset = pd.DataFrame(dataset.target, columns=['tipo'])


def input_model():
    model = st.sidebar.selectbox("Modelo", options=MODELS.keys())
    return MODELS[model]


def input_feature_selection():
    features = {}
    for column in X_dataset.columns:
        features[column] = st.sidebar.checkbox("Incluir {}".format(column))

    X_selected = []
    for feature_key in features.keys():
        if features[feature_key] == True:
            X_selected.append(feature_key)

    y_name = st.sidebar.selectbox("Alvo", options=list(y_dataset.columns))

    return X_selected, y_name
 
def input_features(features):
    data = dict()

    for feature in features:
        _min = float(X_dataset.describe()[feature][3])
        _max = float(X_dataset.describe()[feature][7])

        data[feature] = st.sidebar.slider(feature, min_value=_min, max_value=_max)

    return pd.DataFrame(data, index=[0])

st.sidebar.header("Selecione o Modelo")

# Usuário seleciona modelo
config = input_model()

# Usuario configura modelo
model = config(st.sidebar)

st.sidebar.header("Parâmetros")

# Features selecionadas
X_selected, y_name = input_feature_selection()

if len(X_selected) > 0:
    st.sidebar.header("Entradas")
    df = input_features(X_selected)

    # Formação dos dataframes com features e target
    X = X_dataset[X_selected]
    y = y_dataset[y_name]

    # Divide entre treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    st.write('## Modelo Selecionado:')
    st.write(model)

    st.write("## Avaliação do Modelo:")
    st.write(
        pd.DataFrame(data={
            'rotulos':dataset.target_names,
            'precision':precision_score(y_test, predictions, average=None),
            'recall':recall_score(y_test, predictions, average=None),
            'accuracy':accuracy_score(y_test, predictions)
        })
    )

    st.write("## Predição do Modelo")
    prediction = model.predict(df)
    pred_class = dataset.target_names[prediction]
    st.write(pred_class)
else:
    st.error("Selecione pelo menos uma feature na aba lateral!")
