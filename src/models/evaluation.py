from components import components
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


PRIMARY_RED = "#DB1A1A"
ACCENT_GREEN = "#8CC7C4"
ACCENT_BLUE = "#2C687B"


@st.cache_data
def perform_benchmarking(X, y):
    '''
        Realiza a validação cruzada dos modelos (5 Folds) e retorna as métricas,
        realizando o tratamento de dados ausentes ('?') para todos os modelos.
    '''

    X_clean = pd.DataFrame(X).replace("?", np.nan).astype(float)
    X_clean = X_clean.fillna(X_clean.median()).values
    y_binary = np.where(y == 4, 1, 0)

    models = {
        "Regressão Logística": LogisticRegression(max_iter=1000),
        "Support Vector Classifier": SVC(probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    scoring = ['accuracy', 'precision', 'recall', 'f1']
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for name, model in models.items():
        cv_results = cross_validate(model, X_clean, y_binary, cv=cv, scoring=scoring)
        results.append({
            "Modelo": name,
            "Acurácia": cv_results['test_accuracy'].mean(),
            "Precisão": cv_results['test_precision'].mean(),
            "Recall": cv_results['test_recall'].mean(),
            "F1-Score": cv_results['test_f1'].mean()
        })

    return pd.DataFrame(results)


def render_evaluation(X, y):
    '''
        Renderiza os resultados do benchmarking na tela do Streamlit.
    '''

    components.show_section_title("🏆 Benchmarking e Avaliação dos Modelos")

    with st.spinner("Executando validação cruzada (5 folds)..."):
        df_results = perform_benchmarking(X, y)


    # Exibe a tabeka destacando os maiores valores por métrica
    st.dataframe(
        df_results.style.highlight_max(
            subset=["Acurácia", "Precisão", "Recall", "F1-Score"], 
            color="#46A849", 
            axis=0
        ).format({
            "Acurácia": "{:.2%}", 
            "Precisão": "{:.2%}", 
            "Recall": "{:.2%}", 
            "F1-Score": "{:.2%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Gráfico comparativo
    df_melted = df_results.melt(id_vars="Modelo", var_name="Métrica", value_name="Pontuação")
    fig = px.bar(
        df_melted, 
        x="Métrica", 
        y="Pontuação", 
        color="Modelo", 
        barmode="group",
        color_discrete_sequence=[ACCENT_BLUE, ACCENT_GREEN, PRIMARY_RED]
    )
    
    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        yaxis=dict(
            range=[0.85, 1.0], 
            title=dict(text="Pontuação", 
            font=dict(color="#2C687B")),
            tickfont=dict(color="#2C687B", size=12),
        ),

        xaxis=dict(
            title="",
            tickfont=dict(color="#2C687B", size=12),
        ),

        legend=dict(
            title=dict(
                text="Modelos",
                font=dict(color="#2C687B", size=12)
            ),
            font=dict(color="#2C687B", size=12),
            bgcolor="#FFFFFF",
            bordercolor="#E0D5D5",
            borderwidth=1
        ),

        margin=dict(t=24, b=24, l=48, r=20),
    )
    
    st.plotly_chart(fig, use_container_width=True)
