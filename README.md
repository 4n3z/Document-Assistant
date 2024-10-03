
# Asistente Documental Basado en Documentos PDF

Este proyecto es una aplicación que permite crear un asistente inteligente capaz de responder preguntas basadas en el contenido de uno o varios documentos PDF. La aplicación utiliza modelos de lenguaje y técnicas de procesamiento de lenguaje natural para analizar el contenido de los documentos, generar embeddings y proporcionar respuestas precisas a las consultas del usuario.

## Tabla de Contenidos

- [Características](#características)
- [Requisitos Previos](#requisitos-previos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Descripción de los Procesos Principales](#descripción-de-los-procesos-principales)
  - [1. Carga de Documentos PDF](#1-carga-de-documentos-pdf)
  - [2. Fragmentación del Contenido del Documento](#2-fragmentación-del-contenido-del-documento)
  - [3. Generación y Almacenamiento de Embeddings](#3-generación-y-almacenamiento-de-embeddings)
  - [4. Generación de Respuestas con AI Toolkit y Modelo ONNX](#4-generación-de-respuestas-con-ai-toolkit-y-modelo-onnx)
- [Antecedentes Relevantes](#antecedentes-relevantes)
- [Créditos](#créditos)
- [Licencia](#licencia)

## Características

- **Carga de múltiples documentos PDF**: Permite al usuario cargar uno o varios archivos PDF para su análisis.
- **Fragmentación precisa del texto**: Utiliza spaCy para segmentar el texto en oraciones de manera eficiente y precisa.
- **Generación de embeddings**: Convierte los fragmentos de texto en embeddings utilizando modelos de lenguaje multilingües.
- **Almacenamiento persistente**: Los embeddings se almacenan de forma persistente utilizando ChromaDB para facilitar su recuperación.
- **Interfaz de usuario interactiva**: Desarrollada con Streamlit, permite una interacción sencilla y amigable.
- **Respuestas generadas localmente**: Utiliza AI Toolkit de Visual Studio Code con modelos ONNX para generar respuestas sin depender de servicios externos.

## Requisitos Previos

Antes de comenzar, asegúrate de tener instalados los siguientes componentes:

- **Python 3.8 o superior**
- **Visual Studio Code** con la extensión **AI Toolkit** instalada
- **Modelo ONNX compatible** (por ejemplo, `Phi-3-mini-4k-cpu-int4-rtn-block-32-onnx`)
- **spaCy** y el modelo de idioma español (`es_core_news_sm`)

## Instalación

Sigue estos pasos para configurar el entorno y ejecutar la aplicación localmente.

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu_usuario/tu_repositorio.git
cd tu_repositorio
```

### 2. Crear y activar un entorno virtual

```bash
python -m venv venv
```

- En Windows:

  ```bash
  venv\Scripts\activate
  ```

- En Unix o MacOS:

  ```bash
  source venv/bin/activate
  ```

### 3. Instalar las dependencias

Asegúrate de tener un archivo `requirements.txt` con las siguientes dependencias:

```plaintext
streamlit
PyMuPDF
langchain
chromadb
sentence-transformers
numpy
openai
spacy
```

Instala las dependencias:

```bash
pip install -r requirements.txt
```

### 4. Instalar spaCy y el modelo de español

```bash
python -m spacy download es_core_news_sm
```

### 5. Configurar AI Toolkit en Visual Studio Code

- Abre **Visual Studio Code**.
- Instala la extensión **AI Toolkit** si aún no lo has hecho.
- Configura el modelo ONNX:
  - Descarga o coloca el modelo `Phi-3-mini-4k-cpu-int4-rtn-block-32-onnx` en una ubicación accesible.
  - Sigue las instrucciones de AI Toolkit para cargar el modelo ONNX.
- Asegúrate de que el servicio esté disponible en `http://127.0.0.1:5272/v1/`.

## Uso

### 1. Ejecutar la aplicación

En el directorio del proyecto, inicia la aplicación de Streamlit:

```bash
streamlit run app.py
```

### 2. Interactuar con la aplicación

- **Carga de documentos**: Utiliza el cargador de archivos para subir uno o varios archivos PDF.
- **Ajustes de búsqueda**: En la barra lateral, puedes ajustar el número de fragmentos relevantes a recuperar (valor de `k`).
- **Realizar consultas**: Escribe tu pregunta en el campo de entrada y presiona Enter. El asistente generará una respuesta basada en el contenido de los documentos.

## Descripción de los Procesos Principales

### 1. Carga de Documentos PDF

Los usuarios pueden cargar uno o varios archivos PDF a través de la interfaz de Streamlit. Los archivos se guardan en el directorio `data/` para su procesamiento. La aplicación maneja la carga y almacenamiento de los archivos de manera segura y eficiente.

### 2. Fragmentación del Contenido del Documento

Para manejar eficientemente el contenido y generar embeddings precisos, los documentos PDF se fragmentan en trozos más pequeños. Se utiliza **spaCy** para dividir el texto en oraciones, lo que mejora la precisión en la segmentación y evita cortar oraciones a la mitad.

### 3. Generación y Almacenamiento de Embeddings

Cada fragmento de texto se convierte en un embedding utilizando el modelo multilingüe `paraphrase-multilingual-MiniLM-L12-v2`. Los embeddings se almacenan en una base de datos vectorial utilizando **Chroma**, lo que permite una rápida recuperación de información relevante.

### 4. Generación de Respuestas con AI Toolkit y Modelo ONNX

Cuando el usuario realiza una pregunta:

1. **Recuperación de fragmentos relevantes**: Utiliza los embeddings almacenados para encontrar los fragmentos más relevantes al contexto de la pregunta.
2. **Generación del prompt**: Crea un prompt que incluye el contexto de los fragmentos recuperados y la pregunta del usuario.
3. **Generación de la respuesta**: Envía el prompt al modelo de lenguaje alojado localmente a través de AI Toolkit. El modelo ONNX procesa el prompt y genera una respuesta.
4. **Presentación al usuario**: La respuesta generada se muestra en la interfaz de Streamlit.

## Antecedentes Relevantes

- **AI Toolkit**: Extensión de Visual Studio Code que permite a los desarrolladores integrar modelos de inteligencia artificial en sus aplicaciones de forma sencilla.
- **Modelos ONNX**: Formato abierto para representar modelos de aprendizaje automático que permite la interoperabilidad entre diferentes frameworks y herramientas.
- **spaCy**: Biblioteca avanzada para el procesamiento de lenguaje natural que ofrece herramientas para la segmentación de texto, etiquetado gramatical y más.
- **LangChain**: Framework para desarrollar aplicaciones basadas en modelos de lenguaje, facilitando la gestión de cadenas de procesamiento, prompts y más.
- **Chroma**: Base de datos vectorial utilizada para almacenar y recuperar embeddings de manera eficiente.

## Créditos

Desarrollado por [Tu Nombre](https://github.com/tu_usuario). Si tienes preguntas o sugerencias, no dudes en contactarme.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para obtener más detalles.
