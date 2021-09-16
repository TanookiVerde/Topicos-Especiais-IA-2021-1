from numpy.random.mtrand import f
import streamlit as st
import pandas as pd

from sklearn import datasets
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.write(
""" 
# Tópicos Especiais em IA - Aula 14/09/2021
- Pedro Vitor Marques Nascimento

## Algoritmo Naive Bayes aplicado no dataset Iris
""")

dataset = datasets.load_iris()
X_dataset = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y_dataset = pd.DataFrame(dataset.target, columns=['tipo'])

 
st.sidebar.header("Parâmetros")

def input_feature_selection():
    y_name = st.sidebar.selectbox("Alvo", options=list(y_dataset.columns))

    features = {}
    for column in X_dataset.columns:
        features[column] = st.sidebar.checkbox("Incluir {}".format(column))

    X_selected = []
    for feature_key in features.keys():
        if features[feature_key] == True:
            X_selected.append(feature_key)

    return X_selected, y_name
 
def input_features(features):
    data = dict()

    for feature in features:
        _min = float(X_dataset.describe()[feature][3])
        _max = float(X_dataset.describe()[feature][7])

        data[feature] = st.sidebar.slider(feature, min_value=_min, max_value=_max)

    return pd.DataFrame(data, index=[0])

X_selected, y_name = input_feature_selection()

if len(X_selected) == 0:
    st.error("Selecione pelo menos uma feature na aba lateral!")
else:
    df = input_features(X_selected)
    
    st.subheader("Parâmetros inseridos")
    st.write(df)

    X = X_dataset[X_selected]
    y = y_dataset[[y_name]]

    X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state =1)
    model = GaussianNB()

    model.fit(X_train,y_train)
    predictions = model.predict(X_test)

    st.subheader("Avaliação do Modelo:")

    st.write(
        pd.DataFrame(data={
            'rotulos':dataset.target_names,
            'precision':precision_score(y_test, predictions, average=None),
            'recall':recall_score(y_test, predictions, average=None),
            'accuracy':accuracy_score(y_test, predictions)
        })
    )
    
    st.subheader("Predição do Modelo")

    prediction = model.predict(df)
    pred_class = dataset.target_names[prediction]

    st.write(pred_class)