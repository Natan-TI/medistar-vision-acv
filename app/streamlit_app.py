from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from tensorflow.keras.models import load_model
except Exception:
    from keras.models import load_model


# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "outputs" / "models" / "best_model.keras"
SAMPLE_DIR = BASE_DIR / "data" / "sample"

IMG_SIZE = (128, 128)

CLASS_NAMES = [
    "agricultura",
    "sombra",
    "terreno_exposto",
    "urbano",
    "vegetacao_densa",
    "vegetacao_esparsa",
]

MEDISTAR_INTERPRETATION = {
    "urbano": "Área com maior chance de infraestrutura, acesso terrestre e proximidade de serviços.",
    "vegetacao_densa": "Região com possível isolamento territorial, floresta densa ou acesso mais difícil.",
    "vegetacao_esparsa": "Área rural ou de transição ambiental, podendo indicar ocupação menos densa.",
    "agricultura": "Região rural produtiva, possivelmente afastada de grandes centros urbanos.",
    "terreno_exposto": "Solo aberto, estrada, clareira ou área degradada, podendo indicar acesso precário ou ausência de cobertura vegetal.",
    "sombra": "Área com baixa visibilidade na imagem, podendo dificultar a análise territorial automática.",
}


# ============================================================
# FUNÇÕES
# ============================================================

@st.cache_resource
def load_trained_model():
    if not MODEL_PATH.exists():
        st.error(f"Modelo não encontrado em: {MODEL_PATH}")
        st.stop()

    return load_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_image(model, image: Image.Image):
    image_array = preprocess_image(image)

    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    probabilities = pd.DataFrame({
        "Classe": CLASS_NAMES,
        "Probabilidade": predictions
    }).sort_values("Probabilidade", ascending=False)

    return predicted_class, confidence, probabilities


def get_sample_images():
    if not SAMPLE_DIR.exists():
        return []

    image_paths = []
    for extension in ["*.png", "*.jpg", "*.jpeg"]:
        image_paths.extend(SAMPLE_DIR.glob(f"*/*{extension[1:]}"))

    return sorted(image_paths)


# ============================================================
# INTERFACE
# ============================================================

st.set_page_config(
    page_title="Medistar Vision",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Medistar Vision")
st.subheader("Classificação de imagens satelitais para apoio à análise territorial")

st.markdown(
    """
    Esta aplicação demonstra o uso do melhor modelo treinado no projeto **Medistar Vision**.

    O modelo recebe um recorte de imagem de satélite e classifica o tipo de território predominante.
    Essa informação pode ser usada como apoio para a plataforma Medistar, ajudando a interpretar
    o contexto geográfico de comunidades isoladas.
    """
)

model = load_trained_model()

st.divider()

st.sidebar.title("Opções")
input_mode = st.sidebar.radio(
    "Escolha a origem da imagem:",
    ["Enviar imagem", "Usar amostra do dataset"]
)

selected_image = None
selected_image_name = None

if input_mode == "Enviar imagem":
    uploaded_file = st.sidebar.file_uploader(
        "Envie uma imagem",
        type=["png", "jpg", "jpeg", "tif", "tiff"]
    )

    if uploaded_file is not None:
        selected_image = Image.open(uploaded_file)
        selected_image_name = uploaded_file.name

else:
    sample_images = get_sample_images()

    if not sample_images:
        st.warning("Nenhuma imagem encontrada em data/sample/.")
    else:
        sample_options = [str(path.relative_to(BASE_DIR)) for path in sample_images]

        selected_option = st.sidebar.selectbox(
            "Selecione uma imagem de amostra:",
            sample_options
        )

        selected_path = BASE_DIR / selected_option
        selected_image = Image.open(selected_path)
        selected_image_name = selected_option


if selected_image is None:
    st.info("Envie uma imagem ou selecione uma amostra do dataset para iniciar a demonstração.")
    st.stop()


predicted_class, confidence, probabilities = predict_image(model, selected_image)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Imagem analisada")
    st.image(selected_image, caption=selected_image_name, use_container_width=True)

with col2:
    st.subheader("Resultado da predição")

    st.metric("Classe prevista", predicted_class)
    st.metric("Confiança", f"{confidence:.2%}")

    st.markdown("### Interpretação Medistar")
    st.info(MEDISTAR_INTERPRETATION[predicted_class])

st.divider()

st.subheader("Probabilidade por classe")
st.dataframe(probabilities, use_container_width=True)

st.bar_chart(
    probabilities.set_index("Classe")["Probabilidade"]
)

st.divider()

st.markdown(
    """
    ### Observação

    Esta aplicação é uma demonstração acadêmica. A classificação territorial não realiza diagnóstico médico,
    mas pode funcionar como uma variável auxiliar para análise de risco operacional em regiões remotas.
    """
)
