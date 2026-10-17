import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def load_data():
    '''
        Carrega o dataset de câncer de mama a partir de um arquivo CSV.
    '''
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "breast_cancer.csv")
    return pd.read_csv(data_path)


def prepare_data(dataset):
    '''
        Separa o dataset em variáveis independentes (X) e variável alvo (y),
        removendo a coluna de identificação, tratando valores ausentes ('?')
        e garantindo o mapeamento correto das classes.
    '''
    X = dataset.iloc[:, 1:-1]
    
    X = X.replace("?", np.nan).astype(float)
    
    X = X.fillna(X.median()).values

    y = dataset.iloc[:, -1].values
    
    return X, y
    

def train_model(X, y):
    '''
        Treina um modelo Random Forest utilizando os dados fornecidos,
        retornando o modelo ajustado.
    '''
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


def predict(model, input_data):
    '''
        Realiza a predição de uma nova amostra, retornando a 
        classe prevista e a probabilidade exata de ser maligno.
    '''
    if isinstance(input_data, pd.DataFrame):
        input_data = input_data.replace("?", np.nan).astype(float).fillna(0).values

    pred = model.predict(input_data)[0]
    
    classes = model.classes_
    malignant_index = np.where(classes == 4)[0][0]
    
    prob_malignant = model.predict_proba(input_data)[0][malignant_index]
    
    return pred, prob_malignant