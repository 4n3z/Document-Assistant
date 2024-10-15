import os
import streamlit as st
import fitz  # Para manejar PDFs
import spacy  # Importar spaCy
import re  # Para limpieza de texto
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Crear directorio 'data' si no existe
if not os.path.exists('data'):
    os.makedirs('data')

# Inicializar el modelo de embeddings con soporte para español
embeddings = SentenceTransformerEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')

# Inicializar el modelo de lenguaje
model = ChatOpenAI(
    base_url="http://127.0.0.1:5272/v1/",
    api_key="ai-toolkit",
    model="Phi-3-mini-4k-cpu-int4-rtn-block-32-onnx",
    temperature=0.7,
    max_tokens=1000  # Aumentar el límite de tokens para evitar respuestas incompletas
)

# Configurar Chroma con un directorio persistente para la persistencia de datos
if not os.path.exists('ai-toolkit'):
    os.makedirs('ai-toolkit')
# Cargar Chroma
load_db = Chroma(persist_directory='./ai-toolkit', embedding_function=embeddings)

# Cargar el modelo de spaCy para español
nlp = spacy.load('es_core_news_sm')

# Función para actualizar el retriever con un valor de k fijo
def update_retriever():
    global retriever
    retriever = load_db.as_retriever(search_kwargs={'k': 5})  # Valor de k fijo

# Definir el template de prompt
template = """
    Eres un asistente especializado en búsqueda basada en documentos.
    Tu tarea es responder la pregunta del usuario recuperando la información más relevante de los documentos.
    Mantén un tono profesional y asegúrate de que tus respuestas sean precisas y útiles.
    Si el usuario solicita la definición de un término, asegúrate de buscar cualquier párrafo que siga a un título o subtítulo relevante.
    Si no conoces la respuesta, responde "No lo sé".
    {context}
    Pregunta:
    {question}
    """

prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

# Inicializar el retriever
update_retriever()

# Definir el pipeline
setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)
chain = setup_and_retrieval | prompt | model | output_parser

# Función de limpieza para eliminar encabezados, pies de página y números
def clean_text(text):
    # Eliminar números de página y espacios repetidos
    text = re.sub(r'\bPage \d+\b', '', text)  # Elimina el patrón "Page X"
    text = re.sub(r'\s+', ' ', text).strip()  # Elimina múltiples espacios
    return text

# Función para extraer texto de PDF con limpieza adicional
def extract_text_from_pdf(file_path):
    text_with_page_numbers = []
    with fitz.open(file_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            text = clean_text(text)  # Limpiar texto
            text_with_page_numbers.append((text, page_num + 1))  # Las páginas comienzan desde 1
    return text_with_page_numbers

# Función para dividir el texto con ventanas deslizantes para no perder contexto
def split_text(text, max_length=600, overlap=100):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    chunks = []
    current_chunk = ""
    for i, sentence in enumerate(sentences):
        # Identificar si la oración actual parece un título o subtítulo
        if re.match(r'^[a-zA-Z0-9\.-]+', sentence):
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
            if i - overlap >= 0:
                overlap_sentence = sentences[i - overlap]
                current_chunk = overlap_sentence + " " + current_chunk

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


# Función para agregar embeddings a Chroma
def add_embeddings_to_chroma(chunks, metadatas, ids):
    load_db.add_texts(texts=chunks, metadatas=metadatas, ids=ids)

# Procesar el PDF con fragmentación optimizada
def process_pdf(file_path, document_id, progress_bar, doc_progress):
    text_with_pages = extract_text_from_pdf(file_path)
    total_pages = len(text_with_pages)
    processed_pages = 0

    for text, page_number in text_with_pages:
        # Dividir el texto en fragmentos con ventanas deslizantes
        chunks = split_text(text)
        metadatas = [{'document_id': document_id, 'page_number': page_number} for _ in chunks]
        ids = [f'{document_id}_page_{page_number}_chunk_{i}' for i in range(len(chunks))]
        # Agregar embeddings a Chroma
        add_embeddings_to_chroma(chunks, metadatas, ids)
        # Actualizar barra de progreso
        processed_pages += 1
        progress_bar.progress(processed_pages / total_pages)
    # Actualizar barra de progreso del documento
    doc_progress[document_id] = 'Indexado'
    update_retriever()

# Interfaz de usuario de Streamlit
st.title("Asistente basado en documentos")
st.write("Este asistente recupera información relevante de tus documentos para responder a tus preguntas.")

# Indicadores de progreso detallados en la barra lateral
st.sidebar.header("Progreso de Indexación de Documentos")
doc_progress = st.sidebar.empty()  # Espacio para los detalles de indexación
doc_progress_details = {}

# Permitir al usuario subir múltiples archivos PDF
uploaded_files = st.file_uploader("Sube archivos PDF", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        document_id = uploaded_file.name  # Usar el nombre del archivo como ID del documento
        temp_file_path = os.path.join('data', uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"El archivo '{uploaded_file.name}' ha sido subido y guardado.")
        # Crear una barra de progreso para la indexación
        st.write(f"Indexando '{uploaded_file.name}'...")
        progress_bar = st.progress(0)
        # Actualizar detalles del progreso del documento
        doc_progress_details[document_id] = 'Indexando...'
        doc_progress.text(f"{document_id}: {doc_progress_details[document_id]}")
        # Procesar el archivo PDF con fragmentación mejorada
        process_pdf(temp_file_path, document_id, progress_bar, doc_progress_details)
        st.success(f"El documento '{uploaded_file.name}' ha sido procesado e indexado.")
        doc_progress.text(f"{document_id}: {doc_progress_details[document_id]}")

# Inicializar el historial de chat
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes de chat previos
st.sidebar.header("Historial de Búsquedas")
for idx, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        if st.sidebar.button(f"Consulta {idx + 1}: {message['content'][:30]}..."):
            st.session_state.user_input = message['content']  # Reutilizar la consulta previa
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada del usuario para la pregunta
if user_input := st.chat_input("Tu pregunta:"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generar respuesta
    response = chain.invoke(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

# Botón para limpiar el historial de chat
if st.button("Limpiar historial de chat"):
    st.session_state.messages = []