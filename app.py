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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #fffde7; color: #333; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #fff9c4 !important;
    border-right: 1px solid #f9a825;
}

/* Headers */
h1 { color: #f57f17 !important; }
h2, h3 { color: #e65100 !important; }

/* Inputs */
textarea, input[type="text"] {
    background-color: #fffff0 !important;
    border: 1px solid #f9a825 !important;
    border-radius: 6px !important;
}

/* Botones */
.stButton > button {
    background: #f9a825 !important;
    color: white !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    width: 100%;
}
.stButton > button:hover {
    background: #f57f17 !important;
}

/* Cards */
.header-card {
    background: #fff8e1;
    border-left: 5px solid #f9a825;
    padding: 28px;
    border-radius: 8px;
    margin-bottom: 20px;
}

.section-card {
    background: #fff8e1;
    border: 1px solid #ffe082;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 PDF Analyzer")
    st.markdown("Analiza documentos con IA")
    st.markdown("---")

    st.markdown("### Configuración")
    ke = st.text_input("API Key OpenAI", type="password")

    if ke:
        os.environ['OPENAI_API_KEY'] = ke

# ─────────────────────────────────────────────
# HEADER (CON IMAGEN)
# ─────────────────────────────────────────────
colH1, colH2 = st.columns([3,1])

with colH1:
    st.markdown("""
    <div class="header-card">
        <h1 style="margin-bottom:6px;">📄 PDF Analyzer</h1>
        <p style="margin:0; color:#f57f17;">
            Explora tus documentos haciendo preguntas sobre su contenido!
        </p>
        <p style="margin-top:8px; font-size:0.8rem; color:#9ca3af;">
    """, unsafe_allow_html=True)

    st.write(f"Versión Python {platform.python_version()}")

with colH2:
    try:
        image = Image.open('Chat_pdf.png')
        st.image(image, use_container_width=True)
    except:
        pass

# ─────────────────────────────────────────────
# CARGA DE PDF
# ─────────────────────────────────────────────

st.markdown("### 📂 Cargar documento")
pdf = st.file_uploader("Sube tu PDF", type="pdf")

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PROCESAMIENTO
# ─────────────────────────────────────────────
if pdf is not None and ke:
    try:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Procesamiento")

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

        st.markdown('</div>', unsafe_allow_html=True)

        # ────────────────
        # PREGUNTAS
        # ────────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        st.markdown("### 💬 Pregunta al documento")

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

                st.markdown("### 🧠 Respuesta")
                st.write(response)

        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")

elif pdf is not None and not ke:
    st.warning("Ingresa tu API Key")

else:
    st.info("Sube un PDF para comenzar")
