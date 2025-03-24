
# Manual Técnico: API de Chatbot Laguna

## Índice
0. [Explicaciones Técnicas](#explicaciones-técnicas)  
1. [Introducción](#introducción)  
2. [Requisitos Previos](#requisitos-previos)  
3. [Configuración del Entorno](#configuración-del-entorno)  
4. [Despliegue en Hugging Face Spaces](#despliegue-en-hugging-face-spaces)  
5. [Endpoints de la API](#endpoints-de-la-api)  
6. [Gestión de Documentos](#gestión-de-documentos)  
7. [WebSockets](#websockets)  
8. [Integración con MongoDB](#integración-con-mongodb)  
9. [Solución de Problemas](#solución-de-problemas)  
10. [Apéndices](#apéndices)

---
## 0. Explicaciones Técnicas <a name="explicaciones-técnicas"></a>

### 0.1 ¿Qué es Hugging Face?
Hugging Face es una plataforma que proporciona modelos de lenguaje preentrenados (LLMs) y herramientas para procesamiento de lenguaje natural (NLP). En este proyecto, se utiliza para acceder a modelos como Mistral-7B y generar embeddings.

### 0.2 ¿Qué es Langchain?
Langchain es un framework que facilita la creación de aplicaciones basadas en modelos de lenguaje. Permite integrar diferentes componentes como modelos, bases de datos y herramientas externas en un flujo de trabajo coherente.

### 0.3 ¿Qué es un RAGbot?
Un RAGbot (Retrieval-Augmented Generation bot) es un chatbot que combina la generación de lenguaje con la recuperación de información. En este proyecto, el RAGbot recupera información relevante de los documentos cargados y la utiliza para generar respuestas más precisas.

### 0.4 ¿Qué es Docker?
Docker es una plataforma que permite empaquetar aplicaciones y sus dependencias en contenedores. Esto facilita el despliegue y la ejecución de aplicaciones en diferentes entornos sin problemas de compatibilidad.

### 0.5 ¿Qué es MongoDB?
MongoDB es una base de datos NoSQL que almacena datos en formato JSON-like. En este proyecto, se utiliza para almacenar documentos y conversaciones de manera persistente.
---

## 1. Introducción <a name="introducción"></a>
La API de Chatbot Laguna es una solución multimodal que combina:
- Modelos LLM de Hugging Face/Mistral
- Procesamiento de PDFs con Unstructured
- RAG (Retrieval-Augmented Generation) con Langchain
- Almacenamiento vectorial con ChromaDB
- Integración con MongoDB para persistencia

Funcionalidades clave:
- Carga y análisis de documentos PDF
- Consultas en lenguaje natural con contexto multimodal
- Web scraping y generación de reportes PDF
- Comunicación en tiempo real vía WebSockets

---

## 2. Requisitos Previos <a name="requisitos-previos"></a>

### 2.1 Cuentas Requeridas
1. **Hugging Face**  
   - Registro en [huggingface.co](https://huggingface.co)
   - Obtener API Token desde [Settings → Access Tokens](https://huggingface.co/settings/tokens)

2. **Groq Cloud**  
   - Registro en [console.groq.com](https://console.groq.com)
   - Crear API Key en sección "API Keys"

3. **MongoDB Atlas**  
   - Crear cuenta gratis en [mongodb.com](https://www.mongodb.com)
   - Crear cluster y obtener URI de conexión
4. **Langchain**
    - Crear una cuenta gratis en [langchain.com](https://smith.langchain.com/)
    - Crear una API Key en la seccion de "API Keys"

### 2.2 Herramientas
- Python 3.9+
- Git
- Docker (opcional para despliegue)
- Postman/Insomnia para pruebas

---

## 3. Configuración del Entorno <a name="configuración-del-entorno"></a>

### 3.1 Instalación
```bash
# Clonar repositorio
git clone https://github.com/TecLaguna/ChatBotLaguna.git
cd chatbot-laguna

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### 3.2 Configurar Variables de Entorno
Crear archivo `.env` en la raíz del proyecto:
```ini
GROQ_API_KEY=tu_api_key_groq
MONGODB_URI=mongodb+srv://usuario:contraseña@cluster.mongodb.net/db
HUGGINGFACEHUB_API_TOKEN=tu_token_hf
LANGCHAIN_API_KEY=tu_key_langchain
```

### 3.3 Estructura de Carpetas
```
chatbot-laguna/
├── app.py
├── requirements.txt
├── .env
├── documents/  # PDFs cargados
├── _temp/      # Datos temporales
└── images/     # Imágenes descargadas
```

---

## 4. Despliegue en Hugging Face Spaces <a name="despliegue-en-hugging-face-spaces"></a>

### 4.1 Configurar Dockerfile
```dockerfile
FROM python:3.10

# Define variables de entorno
ENV HOST=0.0.0.0
ENV LISTEN_PORT=7860

# Expone el puerto 7860 para que se pueda acceder desde fuera del contenedor
EXPOSE 7860

# Actualiza el sistema e instala dependencias necesarias (ejecutado como root)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    gcc \
    g++ \
    make \
    wget \
    libsqlite3-dev \
    sqlite3 \
    zlib1g-dev \
    libssl-dev \
    libffi-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libbz2-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Descargar y compilar una versión compatible de SQLite
RUN wget https://www.sqlite.org/2024/sqlite-autoconf-3450100.tar.gz && \
    tar -xzf sqlite-autoconf-3450100.tar.gz && \
    cd sqlite-autoconf-3450100 && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf sqlite-autoconf-3450100 sqlite-autoconf-3450100.tar.gz

# Verificar la versión de SQLite instalada
RUN sqlite3 --version

# Recompilar Python para que use la nueva versión de SQLite
RUN wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz && \
    tar -xzf Python-3.10.13.tgz && \
    cd Python-3.10.13 && \
    ./configure --enable-optimizations --with-ensurepip=install && \
    make && \
    make install && \
    cd .. && \
    rm -rf Python-3.10.13 Python-3.10.13.tgz

# Verificar que Python usa la nueva versión de SQLite
RUN python3 -c "import sqlite3; print('SQLite Version:', sqlite3.sqlite_version)"

# 🔹 Ahora creamos el usuario y cambiamos permisos
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Copia el archivo de dependencias y las instala sin caché para optimizar el tamaño del contenedor
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copia el código fuente de la aplicación al contenedor
COPY --chown=user . .

# Comando por defecto al iniciar el contenedor
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

### 4.2 Pasos de Despliegue
1. Crear nuevo Space en HF
2. Seleccionar "Docker" como backend
3. Subir archivos vía Git:
```bash
git init
git add .
git commit -m "Initial commit"
git push huggingface main
```

### 4.3 Configurar Secrets
En la configuración del Space, agregar las variables de entorno como "Secrets".

---

## 5. Endpoints de la API <a name="endpoints-de-la-api"></a>

### 5.1 Carga de Documentos PDF
**Endpoint**: `POST /embed`  
**Body**: Form-data con archivo PDF

Ejemplo cURL:
```bash
curl -X POST -F "file=@documento.pdf" http://localhost:8000/embed
```

Respuesta exitosa:
```json
{
  "message": "PDF procesado y datos incrustados correctamente"
}
```

### 5.2 Consultas al Chatbot
**Endpoint**: `POST /query`  
**Body**:
```json
{
  "question": "¿Qué requisitos hay para la beca?",
  "conversation_name": "Becas"
}
```

Ejemplo de respuesta:
```json
{
  "response": "Los requisitos principales son... intent:university_related"
}
```

### 5.3 Web Scraping y Generación de PDF
**Endpoint**: `POST /scrape_and_embed`  
**Body**:
```json
{
  "links": ["https://universidad.edu.mx/becas"]
}
```

---

## 6. Gestión de Documentos <a name="gestión-de-documentos"></a>

### 6.1 Listar Documentos
```bash
GET /get_documents
```

### 6.2 Eliminar Documento
```bash
POST /delete_document
Body: { "id": "document_id" }
```

---

## 7. WebSockets <a name="websockets"></a>
Conexión para chat en tiempo real:
```javascript
const socket = new WebSocket('ws://localhost:8000/ws');

socket.onmessage = (event) => {
  console.log('Mensaje recibido:', event.data);
};
```

---

## 8. Integración con MongoDB <a name="integración-con-mongodb"></a>

### 8.1 Estructura de Datos
**Documentos**:
```json
{
  "_id": ObjectId,
  "file_path": "documents/informe.pdf",
  "doc_ids": ["uuid1", "uuid2"],
  "texts": ["texto extraído..."],
  "tables": [["datos tabla"]],
  "upload_date": ISODate()
}
```

**Conversaciones**:
```json
{
  "conversation_name": "Becas",
  "question": "¿Qué requisitos hay?",
  "response": "Los requisitos son...",
  "timestamp": ISODate(),
  "status": "Success"
}
```

---

## 9. Solución de Problemas <a name="solución-de-problemas"></a>

### Error: `SSL_CERT_FILE` no encontrado
**Solución**:
```python
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
```

### Error: Timeout en Hugging Face
**Posibles causas**:
1. Límite de tokens excedido
2. Modelo no disponible
3. Problemas de red

**Solución**:
- Verificar estado del modelo en [HF Status](https://status.huggingface.co)
- Reducir `max_length` en la configuración

---

## 10. Apéndices <a name="apéndices"></a>

### A. Diagrama de Arquitectura
```
[Usuario] → [FastAPI] ↔ [ChromaDB]
                   ↳ [Hugging Face]
                   ↳ [MongoDB]
                   ↳ [WebSocket]
```

### B. Dependencias Clave
```python
langchain==0.1.0
fastapi==0.109.0
pymongo==4.6.1
unstructured==0.12.2
huggingface_hub==0.20.2
```

### C. Información del Contribuidor

**Contribuidor Principal**:  
**Carlos Roberto Rocha Trejo**  
- **GitHub**: [@RobertoRochaT](https://github.com/RobertoRochaT)  
- **LinkedIn**: [Carlos Roberto Rocha Trejo](https://linkedin.com/in/carlosr-rocha)  
- **Rol**: Desarrollador.
- **Contribuciones**:
  - Integración de modelos de lenguaje avanzados (LLMs) y sistemas de recuperación de información (RAG).  
  - Desarrollo de la API RESTful y WebSockets para comunicación en tiempo real.  
  - Implementación de la base de datos vectorial (ChromaDB) y persistencia en MongoDB.  
  - Optimización del contenedor Docker para despliegue en Hugging Face Spaces.  

---

**Agradecimientos Especiales**:  
Agradecemos a todos los colaboradores y a la comunidad de código abierto por su invaluable apoyo en el desarrollo de este proyecto. Su dedicación y expertise han sido fundamentales para el éxito de esta iniciativa.  

---

Este manual y el proyecto son mantenidos activamente por el equipo de desarrollo. ¡Gracias por tu interés en Chatbot Laguna! 🚀
