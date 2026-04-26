# Proyecto-tesis-backend — API RAG EPN

Backend del sistema de Asistente Académico RAG de la Escuela Politécnica Nacional. Implementa el pipeline RAG completo, autenticación JWT, WebSocket para chat en tiempo real y un panel de administración vía API REST.

---

## Tabla de contenidos

1. [Descripción general](#1-descripción-general)
2. [Tecnologías principales](#2-tecnologías-principales)
3. [Estructura del proyecto](#3-estructura-del-proyecto)
4. [Modelos de inteligencia artificial](#4-modelos-de-inteligencia-artificial)
5. [Caché semántico](#5-caché-semántico)
6. [API REST — endpoints](#6-api-rest--endpoints)
7. [Variables de entorno](#7-variables-de-entorno)
8. [Ejecución en desarrollo (sin Docker)](#8-ejecución-en-desarrollo-sin-docker)
9. [Ejecución con Docker (producción)](#9-ejecución-con-docker-producción)
10. [Notas de implementación](#10-notas-de-implementación)

---

## 1. Descripción general

El backend expone una API REST y un endpoint WebSocket construidos con FastAPI. Su responsabilidad principal es el pipeline RAG:

1. Recibe documentos (PDF, DOCX, TXT, MD), los fragmenta y genera embeddings con un modelo sentence-transformers.
2. Almacena los vectores en Qdrant para búsqueda por similitud coseno.
3. Al recibir una consulta por WebSocket, recupera los fragmentos más relevantes, construye un prompt contextualizado y lo envía al LLM activo.
4. Antes de invocar al LLM, consulta un caché semántico en Redis. Si hay una respuesta suficientemente similar ya almacenada, la retorna directamente.
5. El sistema soporta dos modos de operación (local y cloud) que pueden cambiarse en caliente desde el panel de administración sin reiniciar el servicio.

---

## 2. Tecnologías principales

| Componente | Tecnología |
|---|---|
| Framework web | FastAPI 0.136 + Uvicorn |
| ORM / migraciones | SQLAlchemy 2 + Alembic |
| Base de datos relacional | PostgreSQL 16 (psycopg2-binary) |
| Base de datos vectorial | Qdrant (qdrant-client) |
| Caché semántico | Redis 7 (redis-py) |
| Embeddings | sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`) |
| LLM local | Qwen2.5-3B-Instruct-AWQ vía vLLM (API compatible OpenAI) |
| LLM cloud | Google Gemini 2.5 Flash (google-generativeai) |
| Autenticación | JWT (PyJWT + passlib + bcrypt) |
| Métricas | Prometheus (prometheus-fastapi-instrumentator) |
| Extracción de documentos | PyMuPDF + pymupdf4llm + docx2txt |
| Logging | Loguru |

---

## 3. Estructura del proyecto

```
Proyecto-tesis-backend/
├── app/
│   ├── main.py                  # Punto de entrada FastAPI, lifespan, middlewares
│   ├── core/
│   │   ├── config.py            # Variables de entorno (carga .env)
│   │   ├── security.py          # Hashing de contraseñas, generación y verificación de JWT
│   │   ├── constants.py         # Constantes globales del sistema
│   │   ├── prompts.py           # System prompt editable, plantillas de prompt y límites de tokens
│   │   ├── exceptions.py        # Excepciones personalizadas de dominio
│   │   ├── defaults.py          # Valores por defecto de parámetros RAG y NLU
│   │   ├── event_bus.py         # Bus de eventos interno para comunicación entre servicios
│   │   └── singletons.py        # Instancias únicas de clientes (Qdrant, Redis, embedder)
│   ├── db/
│   │   ├── database.py          # Engine y SessionLocal de SQLAlchemy
│   │   ├── models.py            # Modelos ORM (Usuario, Documento, ConfiguracionMotor, etc.)
│   │   └── deps.py              # Dependencias FastAPI (get_db, get_current_user)
│   ├── schemas/
│   │   └── schemas.py           # Esquemas Pydantic de request/response
│   ├── services/
│   │   ├── rag_service.py       # Pipeline RAG principal (recuperación + generación)
│   │   ├── qdrant_service.py    # Operaciones sobre la base de datos vectorial
│   │   ├── cache_service.py     # Caché exacto y semántico en Redis
│   │   ├── documento_service.py # Fragmentación, embedding e indexación de documentos
│   │   ├── config_service.py    # Lectura y escritura de la configuración del motor activo
│   │   ├── rag_params_service.py# Gestión de parámetros RAG (top_k, threshold, etc.)
│   │   ├── nlu_config_service.py# Configuración del módulo NLU (intents, mensajes)
│   │   ├── intent_service.py    # Clasificación de intents antes del pipeline RAG
│   │   ├── providers.py         # Abstracción del proveedor LLM activo (local/cloud)
│   │   └── __init__.py
│   ├── api/
│   │   └── routers/
│   │       ├── auth.py          # Login y refresh de token JWT
│   │       ├── usuarios.py      # CRUD de usuarios administradores
│   │       ├── documents.py     # Subida, listado, eliminación y reprocesado de documentos
│   │       ├── configuracion.py # Cambio de motor activo (local/cloud)
│   │       ├── rag_params.py    # Ajuste de parámetros RAG en caliente
│   │       ├── nlu_config.py    # Configuración de intents y mensajes NLU
│   │       ├── ws_chat.py       # WebSocket para el chat del usuario final
│   │       └── cache_admin.py   # Gestión del caché semántico (listar, editar, eliminar)
│   └── repositories/
│       └── documento_repository.py  # Acceso a datos de documentos en PostgreSQL
├── requirements.txt
├── Dockerfile.backend
├── .env                         # Variables para ejecución local (no versionar)
└── .env.docker                  # Variables para ejecución dentro de Docker (no versionar)
```

---

## 4. Modelos de inteligencia artificial

El sistema implementa dos modos de operación que se pueden cambiar desde el panel de administración del frontend sin reiniciar ningún servicio.

### Modo `local:local` (por defecto)

- **Embeddings**: `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers, 384 dimensiones). Se ejecuta en CPU dentro del contenedor backend.
- **LLM**: `Qwen/Qwen2.5-3B-Instruct-AWQ`, servido por vLLM bajo el nombre de alias `llama3-local`. Requiere GPU NVIDIA.

Parámetros de vLLM configurados en el compose:

| Parámetro | Valor | Descripción |
|---|---|---|
| `--max-model-len` | 2048 | Longitud máxima de contexto en tokens |
| `--gpu-memory-utilization` | 0.90 | Fracción de VRAM reservada para el modelo |
| `--max-num-seqs` | 4 | Máximo de secuencias en paralelo |
| `--quantization` | awq | Cuantización AWQ para reducir uso de VRAM |
| `--enable-prefix-caching` | — | Reutiliza KV-cache de prefijos de prompt repetidos |

El modelo se descarga automáticamente desde Hugging Face en el primer arranque (~2 GB cuantizado) y se guarda en el volumen `tesis_hf_models_cache`.

### Modo `local:cloud`

- **Embeddings**: mismo modelo local (`paraphrase-multilingual-MiniLM-L12-v2`).
- **LLM**: Google Gemini 2.5 Flash, invocado mediante la API de Google AI Studio.

Este modo está completamente implementado y funcional. Para activarlo se requiere una `GOOGLE_API_KEY` válida en el `.env.docker` del backend y cambiar el motor desde el panel de administración. El servicio `tesis-vllm` sigue corriendo pero no recibe solicitudes de generación.

### Límites del prompt

Los límites de tokens están definidos en `app/core/prompts.py` y se ajustan automáticamente al presupuesto del modelo:

| Concepto | Tokens |
|---|---|
| Ventana total del modelo | 2048 |
| Reservados para respuesta | 512 |
| Reservados para contexto RAG | 630 |
| Disponibles para system prompt editable | ~906 (≈ 3171 caracteres) |

El system prompt base es editable desde el panel de administración. El sufijo con el contexto y la pregunta se concatena automáticamente y no es editable.

---

## 5. Caché semántico

Redis implementa un caché de dos niveles para las respuestas RAG:

- **Caché exacto**: si la pregunta (hasheada) ya existe en Redis, retorna la respuesta almacenada sin invocar al LLM ni a Qdrant.
- **Caché semántico**: si no hay coincidencia exacta, compara el embedding de la nueva pregunta contra los embeddings almacenados. Si la similitud coseno supera el umbral configurado, retorna la respuesta semánticamente más cercana.

Las respuestas no se almacenan en caché si son respuestas de error, respuestas de rechazo (el LLM no encontró información relevante), o si el texto es demasiado breve o incompleto.

Desde el frontend, el panel de administración permite listar todas las entradas del caché, buscar por texto de pregunta, ver la respuesta completa de cada entrada, corregir manualmente una respuesta almacenada y eliminar entradas individuales.

---

## 6. API REST — endpoints

Todos los endpoints protegidos requieren el header `Authorization: Bearer <token>` obtenido en `/auth/login`.

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | Health check del servidor |
| `GET` | `/health` | Health check interno (usado por Docker) |
| `POST` | `/auth/login` | Autenticación. Devuelve token JWT |
| `GET` | `/usuarios/` | Listar administradores |
| `POST` | `/usuarios/` | Crear nuevo administrador |
| `PUT` | `/usuarios/{id}` | Editar administrador |
| `DELETE` | `/usuarios/{id}` | Eliminar administrador |
| `GET` | `/documents/` | Listar documentos indexados |
| `POST` | `/documents/upload` | Subir y procesar nuevo documento |
| `DELETE` | `/documents/{id}` | Eliminar documento y sus vectores |
| `POST` | `/documents/{id}/reprocess` | Reprocesar fragmentación y embedding |
| `GET` | `/configuracion/` | Obtener motor activo (motor_vectores + motor_llm) |
| `PUT` | `/configuracion/` | Cambiar motor activo en caliente |
| `GET` | `/rag-params/` | Obtener parámetros RAG actuales |
| `PUT` | `/rag-params/` | Actualizar parámetros RAG (top_k, threshold, etc.) |
| `GET` | `/nlu-config/` | Obtener configuración NLU (intents, mensajes) |
| `PUT` | `/nlu-config/` | Actualizar configuración NLU |
| `GET` | `/cache/` | Listar entradas del caché semántico |
| `PUT` | `/cache/{key}` | Corregir respuesta almacenada en caché |
| `DELETE` | `/cache/{key}` | Eliminar entrada del caché |
| `WS` | `/ws/chat` | WebSocket para consultas RAG del usuario final |
| `GET` | `/metrics` | Métricas Prometheus |

---

## 7. Variables de entorno

El backend usa dos archivos de entorno según el contexto de ejecución:

### `.env` — ejecución local

```env
DATABASE_URL=postgresql://postgres:<password>@localhost:5432/db_tesis_cc
API_HOST=0.0.0.0
API_PORT=8000
FRONTEND_URL=http://localhost:5173
SECRET_KEY=<clave_hex_larga>
ACCESS_TOKEN_EXPIRE_MINUTES=480
GOOGLE_API_KEY=<tu_api_key>
VLLM_BASE_URL=http://localhost:8001/v1
QDRANT_URL=http://localhost:6334
REDIS_URL=redis://localhost:6380
QDRANT_COLLECTION_LOCAL=documentos_local
LLM_MODEL_LOCAL=llama3-local
EMBED_MODEL_LOCAL=paraphrase-multilingual-MiniLM-L12-v2
EMBED_DIMENSION_LOCAL=384
LLM_MODEL_CLOUD=gemini-2.5-flash
DOCUMENTS_DIR_LOCAL=./documents_local
```

### `.env.docker` — ejecución dentro de Docker

Idéntico al anterior excepto por las URLs de los servicios dependientes, que usan los nombres de contenedor de la red interna `tesis-net`:

```env
DATABASE_URL=postgresql://postgres:<password>@postgres:5432/db_tesis_cc
VLLM_BASE_URL=http://vllm:8001/v1
QDRANT_URL=http://qdrant:6333
REDIS_URL=redis://redis:6379
# El resto de variables son iguales al .env local
```

> **Seguridad**: nunca versionar ninguno de estos archivos. Ambos contienen credenciales y la `SECRET_KEY` de JWT.

Para generar una `SECRET_KEY` segura:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

---

## 8. Ejecución en desarrollo (sin Docker)

Requiere PostgreSQL, Qdrant y Redis corriendo localmente (o sus puertos mapeados desde Docker).

```bash
# Crear entorno virtual e instalar dependencias
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con los valores correctos

# Iniciar el servidor
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

El flag `--reload` activa recarga automática ante cambios en el código fuente. No usar en producción.

Al iniciar, el backend crea automáticamente todas las tablas de la base de datos si no existen y siembra un usuario `admin` / `admin123` si no hay ningún administrador registrado. **Cambiar esta contraseña inmediatamente desde el panel de administración.**

---

## 9. Ejecución con Docker (producción)

El backend no se levanta de forma independiente. Forma parte del stack orquestado desde el repositorio `tesis-deploy`. Consultar el README de ese repositorio para las instrucciones completas de despliegue.

El `Dockerfile.backend` instala PyTorch CPU (suficiente para los embeddings) y luego el resto de dependencias de `requirements.txt`. El modelo de embeddings se descarga automáticamente de Hugging Face en el primer uso.

---

## 10. Notas de implementación

**Precalentamiento al arrancar.** Al iniciar, el backend envía internamente una consulta de prueba al pipeline RAG para que el modelo de embeddings se cargue en memoria antes de la primera solicitud real del usuario. Si vLLM aún no está disponible en ese momento, el precalentamiento falla silenciosamente y el sistema sigue operativo.

**Migración de configuración.** Si existe un archivo `config_ia.json` en el directorio de trabajo (legado de versiones anteriores), el backend lo migra automáticamente a la tabla `ConfiguracionMotor` de PostgreSQL al arrancar y elimina el archivo.

**Cambio de motor en caliente.** Cambiar entre `local:local` y `local:cloud` mediante el endpoint `PUT /configuracion/` no requiere reiniciar el proceso. La configuración se persiste en PostgreSQL y sobrevive reinicios.

**CORS.** El backend acepta peticiones desde cualquier origen (`allow_origins=["*"]`). Esto es deliberado porque el frontend se conecta desde la IP de la máquina detectada dinámicamente en el navegador. Restringir el origen en entornos donde la IP es estática y conocida.

**WebSocket y HTTPS.** El frontend se conecta al WebSocket usando `wss://` cuando accede vía HTTPS. El backend en sí escucha HTTP puro; el cifrado TLS lo aporta nginx en el contenedor del frontend, que actúa como proxy para los endpoints `/api/` y `/ws/`.