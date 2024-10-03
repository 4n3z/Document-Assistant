import os
import streamlit as st
import fitz  # Para manejar PDFs
import spacy  # Importar spaCy
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
    temperature=0.7
)

# Configurar Chroma con un directorio persistente para la persistencia de datos
persist_directory = './ai-toolkit'  # Directorio donde se guardarán los datos
load_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Cargar el modelo de spaCy para español
nlp = spacy.load('es_core_news_sm')

# Función para actualizar el retriever con el valor de k ajustable
def update_retriever(k_value):
    global retriever
    retriever = load_db.as_retriever(search_kwargs={'k': k_value})

# Definir el template de prompt
template = """
    You are a specialized AI assistant for document-based search.
    Your task is to answer the user's question by retrieving the most relevant information from the documents.
    Maintain a professional tone and ensure your responses are accurate and helpful.
    Strictly adhere to the user's question and provide relevant information.
    If you do not know the answer then respond "I don't know". Do not refer to your knowledge base.
    {context}
    Question:
    {question}
    """

prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

# Inicializar el retriever con un valor por defecto de k
default_k = 3
update_retriever(default_k)

# Definir el pipeline
setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)
chain = setup_and_retrieval | prompt | model | output_parser

# Funciones para procesar el PDF
def extract_text_from_pdf(file_path):
    text_with_page_numbers = []
    with fitz.open(file_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            text_with_page_numbers.append((text, page_num + 1))  # Las páginas comienzan desde 1
    return text_with_page_numbers

def split_text(text, max_length=500):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def add_embeddings_to_chroma(chunks, metadatas, ids):
    load_db.add_texts(texts=chunks, metadatas=metadatas, ids=ids)

def process_pdf(file_path, document_id, progress_bar):
    # Extraer texto con números de página
    text_with_pages = extract_text_from_pdf(file_path)
    total_pages = len(text_with_pages)
    processed_pages = 0

    for text, page_number in text_with_pages:
        # Dividir el texto en fragmentos
        chunks = split_text(text)
        metadatas = [{'document_id': document_id, 'page_number': page_number} for _ in chunks]
        ids = [f'{document_id}_page_{page_number}_chunk_{i}' for i in range(len(chunks))]
        # Agregar embeddings a Chroma
        add_embeddings_to_chroma(chunks, metadatas, ids)
        # Actualizar barra de progreso
        processed_pages += 1
        progress_bar.progress(processed_pages / total_pages)
    # Actualizar retriever después de agregar nuevos embeddings
    update_retriever(st.session_state.get('k_value', default_k))

# Interfaz de usuario de Streamlit
st.title("Asistente basado en documentos")
st.write("Este asistente recupera información relevante de tus documentos para responder a tus preguntas.")

# Permitir al usuario ajustar la exactitud de los párrafos encontrados
st.sidebar.header("Ajustes de búsqueda")
k_value = st.sidebar.slider("Número de fragmentos relevantes a recuperar (k)", min_value=1, max_value=10, value=3)
st.session_state['k_value'] = k_value
update_retriever(k_value)

# Permitir al usuario subir múltiples archivos PDF
uploaded_files = st.file_uploader("Sube archivos PDF", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        document_id = uploaded_file.name  # Usar el nombre del archivo como ID del documento
        # Guardar el archivo subido en una ubicación temporal
        temp_file_path = os.path.join('data', uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"El archivo '{uploaded_file.name}' ha sido subido y guardado.")
        # Crear una barra de progreso para la indexación
        st.write(f"Indexando '{uploaded_file.name}'...")
        progress_bar = st.progress(0)
        # Procesar el archivo PDF
        process_pdf(temp_file_path, document_id, progress_bar)
        st.success(f"El documento '{uploaded_file.name}' ha sido procesado e indexado.")

# Inicializar el historial de chat
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes de chat previos
for message in st.session_state.messages:
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
