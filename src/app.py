from components import components
from models import evaluation
from models import logistic_regression, random_forest, svc
import numpy as np
import plotly.express as px
import streamlit as st
from assets import styles
from sklearn.inspection import permutation_importance


# --------------------------
# Paleta de cores utilizadas
# --------------------------
PRIMARY_RED = "#DB1A1A"
LIGHT_BG = "#FFF6F6"
ACCENT_GREEN = "#8CC7C4"
ACCENT_BLUE = "#2C687B"
DARK_GRAY = "#2C2D2E"

# --------------------------
# Modelos
# --------------------------
MODELS = {
    "Regressão Logística": logistic_regression,
    "Support Vector Classifier": svc,
    "Random Forest": random_forest,
}

# --------------------------
# Setup inicial
# --------------------------
def setup_app():
    st.set_page_config(page_title="Preditor - Câncer de Mama", layout="wide")
    styles.load_styles()


# --------------------------
# Dados e Modelo
# --------------------------
@st.cache_data
def load_data():
    return logistic_regression.load_data()


@st.cache_resource
def load_model(model_name, X, y):
    model_module = MODELS[model_name]
    return model_module.train_model(X, y)


# --------------------------
# Sidebar
# --------------------------
def build_sidebar(dataset, feature_names, feature_labels):
    components.show_sidebar()

    model_options = list(MODELS.keys())

    selected_model_name = st.sidebar.selectbox(
        "Escolha o modelo de classificação",
        options=list(MODELS.keys()),
        index=0,
        key="modelo_classificacao_v3",
        filter_mode=None,
        label_visibility="visible",
    )

    user_input = []

    for feature in feature_names:
        components.show_sidebar_card()

        value = st.sidebar.slider(
            feature_labels[feature],
            float(dataset[feature].min()),
            float(dataset[feature].max()),
            float(dataset[feature].mean()),
            help=f"Nome original: {feature}"
        )
        user_input.append(value)

    input_array = np.array(user_input).reshape(1, -1)

    return selected_model_name, input_array


# --------------------------
# Predição
# --------------------------
def make_prediction(model, model_module, input_array):
    return model_module.predict(model, input_array)


# --------------------------
# Resultados
# --------------------------
def show_results(prediction, prob):
    with st.container():
        components.show_section_title("📊 Resultado")
        
        col1, col2 = st.columns(2)

        with col1:
            components.show_status_card(prediction)
        with col2:
            components.show_result_card("Probabilidade de Malignidade", prob)


# --------------------------
# Gráfico de Probabilidade
# --------------------------
def plot_probability_chart(prob):
    figure = px.bar(
        x=["Benigno", "Maligno"],
        y=[1 - prob, prob],
        labels={"x": "", "y": "Probabilidade"},
    )

    figure.update_traces(
        marker_color=[ACCENT_GREEN, PRIMARY_RED],
        text=[f"{(1-prob)*100:.1f}%", f"{prob*100:.1f}%"],
        textposition="outside"
    )

    figure.update_layout(
        plot_bgcolor=LIGHT_BG,
        paper_bgcolor=LIGHT_BG,
        font=dict(color=ACCENT_BLUE),
        margin=dict(l=20, r=20, t=40, b=20),
        
        xaxis=dict(tickfont=dict(color=ACCENT_BLUE)),
        yaxis=dict(tickfont=dict(color=ACCENT_BLUE)),
        yaxis_title_font=dict(color=ACCENT_BLUE),
    )
    st.plotly_chart(figure,  width="stretch")


# --------------------------
# Gráfico de Importância
# --------------------------
def plot_relevance_chart(model, feature_names, feature_labels, X, y):
    components.show_section_title("📊 Influência das Variáveis")

    if hasattr(model, "coef_"):
        values = np.asarray(model.coef_)
        values = values[0] if values.ndim > 1 else values
        color_values = np.abs(values)
        x_title = "Impacto no modelo (coeficiente)"

    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        color_values = values
        x_title = "Importância da variável"

    else:
        result = permutation_importance(
            model,
            X,
            y,
            n_repeats=5,
            random_state=42,
            scoring="accuracy",
        )
        values = result.importances_mean
        color_values = np.abs(values)
        x_title = "Importância por permutação (queda na acurácia)"

    figure = px.bar(
        x=values,
        y=[feature_labels[f] for f in feature_names],
        orientation="h",
        color=color_values,
        color_continuous_scale=[
            (0.0, ACCENT_GREEN),
            (0.5, ACCENT_BLUE),
            (1.0, PRIMARY_RED),
        ],
    )

    figure.update_layout(
        coloraxis_showscale=False,
        xaxis_title=x_title,
        yaxis_title="",
        plot_bgcolor=LIGHT_BG,
        paper_bgcolor=LIGHT_BG,
        font=dict(color=DARK_GRAY),
        xaxis=dict(tickfont=dict(color=ACCENT_BLUE)),
        yaxis=dict(tickfont=dict(color=ACCENT_BLUE)),
        xaxis_title_font=dict(color=ACCENT_BLUE),
    )

    st.plotly_chart(figure, width="stretch")


# --------------------------
# MAIN
# --------------------------
def main():
    setup_app()

    dataset = load_data()
    X, y = logistic_regression.prepare_data(dataset)

    components.show_title_card(
        "Sistema de Predição de Câncer de Mama",
        "Modelo de Machine Learning para apoio à análise clínica"
    )

    feature_labels = {
        "Clump Thickness": "Espessura do aglomerado celular",
        "Uniformity of Cell Size": "Uniformidade do tamanho das células",
        "Uniformity of Cell Shape": "Uniformidade do formato das células",
        "Marginal Adhesion": "Aderência marginal",
        "Single Epithelial Cell Size": "Tamanho das células epiteliais isoladas",
        "Bare Nuclei": "Núcleos desnudos",
        "Bland Chromatin": "Cromatina homogênea",
        "Normal Nucleoli": "Nucléolos normais",
        "Mitoses": "Taxa de mitoses"
    }

    feature_names = dataset.columns[1:-1]

    selected_model_name, input_array = build_sidebar(
        dataset,
        feature_names,
        feature_labels
    )

    model_module = MODELS[selected_model_name]
    model = load_model(selected_model_name, X, y)

    prediction, prob = make_prediction(
        model,
        model_module,
        input_array
    )

    # Divisão das informações em abas
    tab_pred, tab_eval, tab_info = st.tabs([
            "🩺 Resultado da Análise", 
            "⚖️ Comparativo de Modelos", 
            "ℹ️ Sobre o Projeto"
        ])
    
    # ABA 1: Resultado clínico
    with tab_pred:
        show_results(prediction, prob)
        st.markdown("<br>", unsafe_allow_html=True)
            
        with st.expander("📊 Ver distribuição detalhada de probabilidade", expanded=False):
            plot_probability_chart(prob)
                
        with st.expander("🔬 Entender a influência das variáveis nesta decisão", expanded=False):
            plot_relevance_chart(model, feature_names, feature_labels, X, y)
    
    # ABA 2: Benchmarking
    with tab_eval:
        evaluation.render_evaluation(X, y)
    
    # ABA 3: Informações e créditos
    with tab_info:
        components.show_about_project()


# --------------------------
# ENTRYPOINT
# --------------------------
if __name__ == "__main__":
    main()
