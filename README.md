# Backend — Asistente Académico RAG EPN

API REST y WebSocket que implementa el pipeline de Recuperación Aumentada por Generación (RAG) para consultas académicas sobre documentos de la Escuela Politécnica Nacional. Soporta dos modos de operación: inferencia completamente local mediante vLLM, e inferencia híbrida con embeddings locales y LLM en la nube (Google Gemini).

La descripción general del sistema, la arquitectura de servicios, los modelos de inteligencia artificial utilizados y los comandos de orquestación completa están documentados en el README del repositorio de despliegue (`tesis-deploy`).

---

## Tabla de contenidos

1. [Requisitos](#1-requisitos)
2. [Estructura del proyecto](#2-estructura-del-proyecto)
3. [Variables de entorno](#3-variables-de-entorno)
4. [Inicialización automática](#4-inicialización-automática)
5. [API — Endpoints](#5-api--endpoints)
6. [WebSocket — Protocolo de mensajes](#6-websocket--protocolo-de-mensajes)
7. [Caché semántico](#7-caché-semántico)
8. [Patrones de diseño implementados](#8-patrones-de-diseño-implementados)
9. [Gestión del contenedor backend](#9-gestión-del-contenedor-backend)
10. [Verificación del servicio](#10-verificación-del-servicio)

---

## 1. Requisitos

El backend está diseñado para correr como contenedor Docker. Los servicios de los que depende (PostgreSQL, Qdrant, Redis, vLLM) son gestionados por el `docker-compose.yml` del repositorio de despliegue.

Para desarrollo local sin Docker:

- Python 3.11
- Las dependencias de `requirements.txt`
- Los servicios de infraestructura accesibles en los puertos del host definidos en `.env`

---

## 2. Estructura del proyecto

```
Proyecto-tesis-backend/
├── app/
│   ├── main.py                    # Punto de entrada FastAPI, lifespan, middleware
│   ├── api/
│   │   └── routers/
│   │       ├── auth.py            # Login y emisión de tokens JWT
│   │       ├── documents.py       # Gestión de documentos (subida, listado, eliminación)
│   │       ├── usuarios.py        # CRUD de usuarios
│   │       ├── configuracion.py   # Motor activo (local / cloud)
│   │       ├── rag_params.py      # Parámetros RAG ajustables (umbral, k, prompts)
│   │       ├── ws_chat.py         # WebSocket de chat y monitor
│   │       ├── nlu_config.py      # Configuración NLU (saludos, mensajes, etc.)
│   │       └── cache_admin.py     # Administración del caché Redis
│   ├── core/
│   │   ├── config.py              # Lectura de variables de entorno
│   │   ├── constants.py           # Motores válidos, combinaciones, etiquetas
│   │   ├── defaults.py            # Fuente única de valores de seed (NLU, RAG, prompts)
│   │   ├── event_bus.py           # Implementación del patrón Observer
│   │   ├── exceptions.py          # Jerarquía de excepciones del dominio
│   │   ├── prompts.py             # Prompts del sistema y plantillas de usuario
│   │   ├── security.py            # Hashing bcrypt, JWT, dependencias de autenticación
│   │   └── singletons.py          # Clientes singleton (Qdrant, Redis, Embed, Httpx)
│   ├── db/
│   │   ├── database.py            # Engine SQLAlchemy y SessionLocal
│   │   ├── deps.py                # Dependencia get_db para inyección en routers
│   │   └── models.py              # Modelos ORM (Usuario, Documento, ConfiguracionRAG,
│   │                              #   ConfiguracionNLU, ConfiguracionMotor)
│   ├── domain/
│   │   └── documento_state.py     # Patrón State para el ciclo de vida del documento
│   ├── repositories/
│   │   └── documento_repository.py # Patrón Repository para acceso a datos de documentos
│   ├── schemas/
│   │   └── schemas.py             # Modelos Pydantic de entrada y salida
│   └── services/
│       ├── cache_service.py       # Caché semántico en Redis con circuit breaker
│       ├── config_service.py      # Lectura y escritura del motor activo en BD
│       ├── documento_service.py   # Lógica de negocio de documentos
│       ├── intent_service.py      # Clasificación de intención NLU
│       ├── nlu_config_service.py  # Acceso y caché en memoria de ConfiguracionNLU
│       ├── providers.py           # Adaptadores LLM (VLLMAdapter, GeminiAdapter)
│       ├── qdrant_service.py      # Operaciones sobre la colección vectorial
│       ├── rag_params_service.py  # Acceso y caché en memoria de ConfiguracionRAG
│       └── rag_service.py         # Pipeline RAG: chunking, embedding, retrieval, generación
├── .env                           # Variables para ejecución local (usa puertos del host)
├── .env.docker                    # Variables para ejecución en Docker (usa nombres de servicio)
├── .env.example                   # Plantilla de referencia sin valores sensibles
├── Dockerfile.backend
└── requirements.txt
```

---

## 3. Variables de entorno

El backend utiliza dos archivos de entorno según el contexto de ejecución.

### `.env` — ejecución local

Se usa cuando el backend corre directamente con `uvicorn` fuera de Docker. Las URLs apuntan a `localhost` con los puertos mapeados al host.

```env
DATABASE_URL=postgresql://postgres:PASSWORD@localhost:5432/db_tesis_cc
API_HOST=0.0.0.0
API_PORT=8000
FRONTEND_URL=http://IP_LOCAL:5173
SECRET_KEY=clave_hex_de_64_caracteres
ACCESS_TOKEN_EXPIRE_MINUTES=480
GOOGLE_API_KEY=clave_de_google_ai_studio
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

### `.env.docker` — ejecución en contenedor

Se usa cuando el backend corre dentro de Docker Compose. Las URLs utilizan los nombres de servicio de la red interna `tesis-net`.

```env
DATABASE_URL=postgresql://postgres:PASSWORD@postgres:5432/db_tesis_cc
API_HOST=0.0.0.0
API_PORT=8000
FRONTEND_URL=http://IP_LOCAL:5173
SECRET_KEY=clave_hex_de_64_caracteres
ACCESS_TOKEN_EXPIRE_MINUTES=480
GOOGLE_API_KEY=clave_de_google_ai_studio
VLLM_BASE_URL=http://vllm:8001/v1
QDRANT_URL=http://qdrant:6333
REDIS_URL=redis://redis:6379
QDRANT_COLLECTION_LOCAL=documentos_local
LLM_MODEL_LOCAL=llama3-local
EMBED_MODEL_LOCAL=paraphrase-multilingual-MiniLM-L12-v2
EMBED_DIMENSION_LOCAL=384
LLM_MODEL_CLOUD=gemini-2.5-flash
DOCUMENTS_DIR_LOCAL=./documents_local
```

Para generar un `SECRET_KEY` seguro:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

### Descripción de variables

| Variable | Descripción |
|---|---|
| `DATABASE_URL` | Cadena de conexión PostgreSQL completa |
| `SECRET_KEY` | Clave de firma JWT. Debe ser aleatoria y mantenerse secreta |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Duración del token de sesión en minutos |
| `GOOGLE_API_KEY` | Clave de Google AI Studio para el modo cloud (Gemini) |
| `VLLM_BASE_URL` | URL base del servidor vLLM (compatible con OpenAI API) |
| `QDRANT_URL` | URL HTTP de la instancia Qdrant |
| `REDIS_URL` | URL de conexión Redis |
| `QDRANT_COLLECTION_LOCAL` | Nombre de la colección vectorial para documentos locales |
| `LLM_MODEL_LOCAL` | Nombre del modelo servido por vLLM (`--served-model-name`) |
| `EMBED_MODEL_LOCAL` | Modelo sentence-transformers para generación de embeddings |
| `EMBED_DIMENSION_LOCAL` | Dimensión del vector de embedding (debe coincidir con el modelo) |
| `LLM_MODEL_CLOUD` | Modelo Gemini a usar en modo cloud |
| `DOCUMENTS_DIR_LOCAL` | Directorio donde se almacenan los archivos subidos |

---

## 4. Inicialización automática

Al arrancar, el backend ejecuta las siguientes operaciones mediante el `lifespan` de FastAPI:

1. Crea todas las tablas en PostgreSQL si no existen (`create_all`).
2. Crea el usuario administrador por defecto (`admin` / `admin123`) si no existe. Se recomienda cambiar esta contraseña tras el primer acceso.
3. Crea el registro de `ConfiguracionRAG` con los valores de `DEFAULTS_RAG` si no existe.
4. Crea el registro de `ConfiguracionNLU` con los valores de `DEFAULTS_NLU` si no existe.
5. Si existe el archivo `config_ia.json` de versiones anteriores, migra su contenido a la tabla `ConfiguracionMotor` y elimina el archivo.
6. Si el motor LLM activo es `local`, ejecuta una consulta de precalentamiento para inicializar el modelo de embeddings en memoria antes de la primera solicitud real.

Todos los valores de seed están centralizados en `app/core/defaults.py`.

---

## 5. API — Endpoints

Todos los endpoints, excepto `/` y `/api/auth/login`, requieren autenticación mediante el header `Authorization: Bearer <token>`.

### Health

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | Estado del servidor y versión |
| GET | `/metrics` | Métricas en formato Prometheus (text/plain) |

### Autenticación

| Método | Ruta | Descripción |
|---|---|---|
| POST | `/api/auth/login` | Obtener token JWT. Body: `{"username": "", "password": ""}` |

### Documentos

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/documentos/` | Listar todos los documentos |
| POST | `/api/documentos/subir` | Subir y procesar un documento. Formatos: PDF, DOCX, TXT, MD. Máximo 10 documentos, 5 MB por archivo |
| DELETE | `/api/documentos/{id}` | Eliminar documento, sus vectores en Qdrant y su caché en Redis |
| POST | `/api/documentos/reset-todo` | Eliminar todos los documentos y vaciar la colección vectorial |

### Usuarios

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/usuarios/` | Listar usuarios |
| POST | `/api/usuarios/` | Crear usuario |
| PUT | `/api/usuarios/{id}` | Actualizar usuario |
| DELETE | `/api/usuarios/{id}` | Eliminar usuario |

### Configuración del motor

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/config/motor` | Motor activo actual |
| PUT | `/api/config/motor` | Cambiar motor. Body: `{"motor_vectores": "local", "motor_llm": "local\|cloud"}` |

Combinaciones válidas: `local:local` (vLLM + embeddings locales) y `local:cloud` (embeddings locales + Gemini).

### Parámetros RAG

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/rag-params` | Parámetros actuales, defaults y límites |
| PUT | `/api/rag-params` | Actualizar parámetros con limpieza automática de caché |
| POST | `/api/rag-params/reset` | Restaurar parámetros a valores por defecto |
| GET | `/api/rag-params/defaults` | Consultar defaults y límites sin modificar nada |

Parámetros ajustables:

| Parámetro | Tipo | Rango | Descripción |
|---|---|---|---|
| `umbral_relevancia_local` | float | [0.05, 0.50] | Score coseno mínimo para incluir un fragmento en el contexto RAG |
| `rag_k_local` | int | [2, 20] | Número de fragmentos a recuperar de Qdrant |
| `prompt_principal` | string | — | Plantilla del prompt de usuario (`{contexto}` y `{pregunta}`) |
| `system_prompt` | string | — | Prompt del sistema enviado al LLM |

### Configuración NLU

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/nlu-config` | Configuración NLU actual |
| PUT | `/api/nlu-config` | Actualizar campos NLU (actualización parcial) |
| POST | `/api/nlu-config/reset` | Restaurar a valores por defecto |
| GET | `/api/nlu-config/defaults` | Consultar defaults sin modificar |

### Administración de caché

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/cache-admin/entradas` | Listar entradas del caché Redis con preview de respuesta |
| GET | `/api/cache-admin/entradas/buscar?q=` | Buscar entradas por texto de pregunta |
| GET | `/api/cache-admin/entradas/{key}` | Obtener entrada completa por clave Redis |
| PATCH | `/api/cache-admin/entradas/{key}` | Corregir manualmente la respuesta de una entrada |
| DELETE | `/api/cache-admin/entradas/{key}` | Eliminar una entrada del caché |

---

## 6. WebSocket — Protocolo de mensajes

### Chat — `ws://host:8000/ws/chat`

No requiere autenticación. El servidor asigna un `client_id` único a cada conexión.

**Mensajes que envía el cliente:**

```json
{ "tipo": "pregunta", "pregunta": "¿Cuántos créditos necesito para graduarme?" }
```

```json
{ "tipo": "ping" }
```

**Mensajes que envía el servidor:**

| `tipo` | Descripción |
|---|---|
| `status` | Mensaje de estado durante el procesamiento |
| `token` | Fragmento de texto en streaming (solo modo `local:local`) |
| `respuesta` | Respuesta completa con campos `pregunta`, `respuesta`, `motor`, `timestamp` |
| `complete` | Señal de fin de consulta |
| `error` | Descripción del error |
| `pong` | Respuesta a ping |
| `info` | Notificación broadcast cuando el motor activo cambia |

El servidor aplica un rate limit de 60 consultas por minuto por cliente, implementado con Redis.

### Monitor — `ws://host:8000/ws/monitor?token=JWT`

Requiere token JWT válido. Emite en tiempo real el estado de conexiones activas, consultas en curso, historial de consultas, latencia promedio y tasa de acierto del caché.

---

## 7. Caché semántico

El servicio `cache_service.py` implementa un caché de respuestas RAG en Redis con dos niveles de búsqueda y un circuit breaker para tolerancia a fallos.

### Niveles de búsqueda

**Nivel 1 — Coincidencia exacta.** La pregunta se hashea y se busca la clave correspondiente en Redis. Si existe, retorna la respuesta sin ningún procesamiento adicional.

**Nivel 2 — Coincidencia semántica.** Si no hay coincidencia exacta y el embedding de la pregunta está disponible, se compara mediante similitud coseno contra los embeddings almacenados en Redis. Si alguno supera el umbral configurado, se retorna esa respuesta. El scan semántico está limitado a un número máximo de claves para no bloquear el servidor.

### Criterios de almacenamiento

Una respuesta no se almacena en caché si:
- El texto tiene menos de 20 caracteres
- Contiene frases de rechazo configuradas en el NLU (el LLM no encontró información)
- Es una respuesta de error interno
- El texto no termina con un signo de puntuación final (respuesta truncada)

### Circuit breaker

Si Redis falla tres veces consecutivas, el circuit breaker suspende el uso del caché durante 30 segundos. El sistema continúa funcionando sin caché durante ese período.

### Administración desde el frontend

El panel de administración expone una vista completa del caché donde es posible buscar entradas, ver la respuesta almacenada junto con su motor de origen y timestamp, corregir manualmente respuestas incorrectas sin necesidad de reprocesar el documento, y eliminar entradas individuales.

---

## 8. Patrones de diseño implementados

| Patrón | Ubicación | Aplicación |
|---|---|---|
| Adapter | `app/services/providers.py` | `VLLMAdapter` y `GeminiAdapter` implementan `LLMAdapter`. Permiten intercambiar el LLM sin modificar el servicio RAG |
| Factory | `app/services/providers.py` | `crear_proveedor_llm(motor)` instancia el proveedor adecuado según el motor activo |
| Singleton | `app/core/singletons.py` | Una instancia compartida por proceso para los clientes de Qdrant, Redis, modelo de embeddings y httpx |
| Repository | `app/repositories/documento_repository.py` | `DocumentoRepository` encapsula todas las queries SQLAlchemy sobre la entidad Documento |
| Builder | `app/services/rag_service.py` | `PipelineRAGBuilder` construye el prompt final con selección y truncamiento de contexto según límite de tokens |
| State | `app/domain/documento_state.py` | `EstadoNoSubido`, `EstadoProcesado`, `EstadoError` controlan las transiciones del ciclo de vida de un documento |
| Observer | `app/core/event_bus.py` | `EventBus` notifica a los clientes WebSocket conectados cuando el motor activo cambia |

---

## 9. Gestión del contenedor backend

Estos comandos operan exclusivamente sobre el contenedor `tesis-backend`. Se ejecutan desde el directorio raíz del proyecto de despliegue (donde se encuentra el `docker-compose.yml`).

### Detener el backend

```bash
docker compose stop backend
```

### Iniciar el backend detenido

```bash
docker compose start backend
```

### Reiniciar el backend

```bash
docker compose restart backend
```

### Reconstruir y reiniciar el backend

Necesario únicamente cuando se modifican `requirements.txt` o el `Dockerfile.backend`. Para cambios en código Python no es necesario porque el directorio del proyecto está montado como volumen en `/app`.

```bash
docker compose up -d --build --no-deps backend
```

### Ver logs en tiempo real

```bash
docker logs tesis-backend -f --tail 100
```

### Ejecutar un comando dentro del contenedor

```bash
docker exec -it tesis-backend bash
```

---

## 10. Verificación del servicio

### Estado HTTP

```bash
curl -s http://localhost:8000/ | python3 -m json.tool
```

Respuesta esperada:

```json
{
    "status": "ok",
    "message": "Servidor Backend RAG (Arquitectura Dual) en línea.",
    "version": "4.0.0",
    "modos_activos": ["local:local", "local:cloud"]
}
```

### Autenticación

```bash
curl -s -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}' \
  | python3 -m json.tool
```

### Motor activo

```bash
TOKEN="<access_token_del_paso_anterior>"

curl -s http://localhost:8000/api/config/motor \
  -H "Authorization: Bearer $TOKEN" \
  | python3 -m json.tool
```

### Logs de arranque esperados

```
[STARTUP] admin creado              # solo en el primer arranque
[STARTUP] precalentando modelo (local:local)
Application startup complete.
```

Si el motor activo es `local:cloud`:

```
[STARTUP] motor cloud activo, precalentamiento omitido
Application startup complete.
```