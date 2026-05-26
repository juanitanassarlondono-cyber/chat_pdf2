import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Analyzer",
    page_icon="📄",
    layout="wide",
)

# ─────────────────────────────────────────────
# ESTILOS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(14, 165, 233, 0.10), transparent 32%),
        radial-gradient(circle at top right, rgba(99, 102, 241, 0.10), transparent 28%),
        #f8fafc;
    color: #0f172a;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
    border-right: 1px solid rgba(148, 163, 184, 0.25);
}

[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(226, 232, 240, 0.18);
}

[data-testid="stSidebar"] .stTextInput > div > div > input {
    background-color: rgba(15, 23, 42, 0.95) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(125, 211, 252, 0.45) !important;
    border-radius: 14px !important;
}

[data-testid="stSidebar"] .stFileUploader {
    background: rgba(255, 255, 255, 0.06);
    padding: 16px;
    border-radius: 18px;
    border: 1px dashed rgba(125, 211, 252, 0.45);
}

/* Headers */
h1 {
    color: #0f172a !important;
    letter-spacing: -0.04em;
    font-weight: 800 !important;
}

h2, h3 {
    color: #1e293b !important;
    letter-spacing: -0.02em;
    font-weight: 700 !important;
}

p {
    line-height: 1.65;
}

/* Inputs */
textarea,
input[type="text"] {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 14px !important;
    color: #0f172a !important;
}

textarea:focus,
input[type="text"]:focus {
    border: 1px solid #0ea5e9 !important;
    box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.16) !important;
}

/* Botones */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%) !important;
    color: white !important;
    border-radius: 14px !important;
    border: none !important;
    font-weight: 700 !important;
    width: 100%;
    padding: 0.75rem 1rem;
    box-shadow: 0 10px 25px rgba(37, 99, 235, 0.18);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #0284c7 0%, #1d4ed8 100%) !important;
    color: white !important;
    transform: translateY(-1px);
}

/* Uploader */
[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 18px;
}

[data-testid="stFileUploader"] label {
    color: #334155 !important;
    font-weight: 600 !important;
}

/* Alertas */
[data-testid="stAlert"] {
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.25);
}

/* Cards */
.hero-card {
    background: linear-gradient(135deg, #ffffff 0%, #eef6ff 100%);
    border: 1px solid #dbeafe;
    padding: 36px;
    border-radius: 28px;
    margin-bottom: 22px;
    box-shadow: 0 24px 60px rgba(15, 23, 42, 0.08);
}

.hero-badge {
    display: inline-block;
    background: #dbeafe;
    color: #1d4ed8;
    padding: 7px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    margin-bottom: 14px;
}

.hero-subtitle {
    color: #475569;
    font-size: 1.04rem;
    margin: 8px 0 0 0;
    max-width: 760px;
}

.version-pill {
    display: inline-block;
    margin-top: 18px;
    padding: 8px 12px;
    border-radius: 999px;
    background: #f1f5f9;
    color: #475569;
    font-size: 0.82rem;
    font-family: 'IBM Plex Mono', monospace;
}

.section-card {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid #e2e8f0;
    padding: 24px;
    border-radius: 24px;
    margin-bottom: 18px;
    box-shadow: 0 18px 45px rgba(15, 23, 42, 0.06);
}

.result-card {
    background: linear-gradient(135deg, #ecfeff 0%, #f8fafc 100%);
    border: 1px solid #bae6fd;
    border-left: 6px solid #0ea5e9;
    padding: 24px;
    border-radius: 22px;
    margin-top: 18px;
}

.sidebar-card {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(226, 232, 240, 0.14);
    border-radius: 18px;
    padding: 16px;
    margin-bottom: 16px;
}

.step-list {
    margin: 0;
    padding-left: 18px;
    color: #475569;
}

.step-list li {
    margin-bottom: 7px;
}

.small-muted {
    color: #64748b;
    font-size: 0.88rem;
}

.metric-soft {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 14px 16px;
}

/* Oculta decoración innecesaria */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 PDF Analyzer")
    st.markdown("Analiza documentos PDF con inteligencia artificial.")
    st.markdown("---")

    st.markdown("""
    <div class="sidebar-card">
        <strong>⚙️ Configuración</strong>
        <p style="font-size:0.88rem; margin:8px 0 0 0; color:#cbd5e1 !important;">
            Ingresa tu API Key para activar el análisis del documento.
        </p>
    </div>
    """, unsafe_allow_html=True)

    ke = st.text_input("API Key OpenAI", type="password")

    if ke:
        os.environ['OPENAI_API_KEY'] = ke

    st.markdown("---")

    st.markdown("""
    <div class="sidebar-card">
        <strong>📂 Documento</strong>
        <p style="font-size:0.88rem; margin:8px 0 0 0; color:#cbd5e1 !important;">
            Sube un archivo PDF para procesarlo.
        </p>
    </div>
    """, unsafe_allow_html=True)

    pdf = st.file_uploader("Sube tu PDF", type="pdf")

    st.markdown("---")

    st.markdown("""
    <div class="sidebar-card">
        <strong>📖 ¿Cómo funciona?</strong>
        <p style="font-size:0.86rem; margin:8px 0 0 0; color:#cbd5e1 !important;">
            La app extrae el texto del PDF, lo divide en fragmentos y usa IA para responder preguntas sobre el contenido.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
colH1, colH2 = st.columns([3, 1])

with colH1:
    st.markdown("""
    <div class="hero-card">
        <span class="hero-badge">Asistente documental con IA</span>
        <h1 style="margin-bottom:8px;">📄 PDF Analyzer</h1>
        <p class="hero-subtitle">
            Sube un documento, procesa su contenido y haz preguntas claras sobre la información del PDF.
        </p>
    """, unsafe_allow_html=True)

    st.markdown(
        f'<span class="version-pill">Python {platform.python_version()}</span>',
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

with colH2:
    try:
        image = Image.open('Chat_pdf.png')
        st.image(image, use_container_width=True)
    except:
        st.markdown("""
        <div class="section-card" style="text-align:center;">
            <div style="font-size:4rem;">🤖</div>
            <p class="small-muted" style="margin-bottom:0;">
                Tu asistente de lectura PDF
            </p>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ESTADO INICIAL
# ─────────────────────────────────────────────
status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    st.markdown("""
    <div class="metric-soft">
        <strong>1. Configura</strong>
        <p class="small-muted" style="margin:6px 0 0 0;">Agrega tu API Key.</p>
    </div>
    """, unsafe_allow_html=True)

with status_col2:
    st.markdown("""
    <div class="metric-soft">
        <strong>2. Carga</strong>
        <p class="small-muted" style="margin:6px 0 0 0;">Sube tu PDF.</p>
    </div>
    """, unsafe_allow_html=True)

with status_col3:
    st.markdown("""
    <div class="metric-soft">
        <strong>3. Pregunta</strong>
        <p class="small-muted" style="margin:6px 0 0 0;">Consulta el contenido.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PROCESAMIENTO
# ─────────────────────────────────────────────
if pdf is not None and ke:
    try:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Procesamiento del documento")

        pdf_reader = PdfReader(pdf)
        text = ""

        for page in pdf_reader.pages:
            contenido = page.extract_text()
            if contenido:
                text += contenido

        if not text.strip():
            st.error("No se pudo extraer texto del PDF")
            st.stop()

        st.success(f"{len(text):,} caracteres procesados")

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20
        )

        chunks = text_splitter.split_text(text)

        st.info(f"{len(chunks)} fragmentos creados")

        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        st.markdown("</div>", unsafe_allow_html=True)

        # ────────────────
        # PREGUNTAS
        # ────────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        st.markdown("### 💬 Pregunta al documento")
        st.markdown(
            '<p class="small-muted">Escribe una pregunta concreta para obtener una respuesta basada en el contenido del PDF.</p>',
            unsafe_allow_html=True
        )

        user_question = st.text_area(
            "",
            placeholder="Ej: ¿Cuál es la conclusión del documento?"
        )

        if user_question:
            with st.spinner("Pensando..."):
                docs = knowledge_base.similarity_search(user_question)

                llm = OpenAI(
                    temperature=0,
                    model_name="gpt-4o"
                )

                chain = load_qa_chain(llm, chain_type="stuff")

                response = chain.run(
                    input_documents=docs,
                    question=user_question
                )

                st.markdown("""
                <div class="result-card">
                    <h3 style="margin-top:0;">🧠 Respuesta</h3>
                """, unsafe_allow_html=True)

                st.write(response)

                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")

elif pdf is not None and not ke:
    st.warning("Ingresa tu API Key")

else:
    st.info("Sube un PDF para comenzar")
