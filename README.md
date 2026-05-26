# Asistente Académico EPN — Local - Prototipo 3

Sistema de consulta académica desarrollado como prototipo de tesis para la Escuela Politécnica Nacional. Permite consultar documentos institucionales mediante un modelo de lenguaje ejecutado íntegramente de forma local, sin dependencia de servicios externos. El backend está construido con FastAPI, el frontend con Vue 3, y el almacenamiento vectorial utiliza ChromaDB sobre PostgreSQL.

---

## Especificaciones del entorno de desarrollo

| Componente | Detalle |
|---|---|
| Equipo | Lenovo IdeaPad Gaming 3 |
| Sistema operativo | Windows 11 |
| RAM | 16 GB |
| Almacenamiento | 1 TB SSD |
| GPU | NVIDIA GeForce RTX 3050 (4 GB GDDR6) |

---

## Requisitos de software

| Componente | Versión |
|---|---|
| Python | 3.11 |
| Node.js | 20 LTS |
| PostgreSQL | 15 |
| Ollama | 0.6.x |
| CUDA Toolkit | 12.x |

---

## 1. Configuración de Ollama

### 1.1 Instalación

Descargar el instalador desde `https://ollama.com/download` y ejecutarlo. Ollama se instala como servicio en segundo plano.

Verificar la instalación en PowerShell:

```powershell
ollama --version
```

### 1.2 Forzar uso de la RTX 3050

El equipo tiene iGPU AMD e dGPU NVIDIA. Sin esta configuración, Ollama carga el modelo en CPU. Ejecutar PowerShell como Administrador:

```powershell
[System.Environment]::SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", "0", "Machine")
[System.Environment]::SetEnvironmentVariable("OLLAMA_GPU_DRIVER", "cuda", "Machine")
```

Reiniciar el servicio de Ollama tras aplicar los cambios.

### 1.3 Descargar los modelos

```powershell
ollama pull qwen3:4b
ollama pull nomic-embed-text:v1.5
```

### 1.4 Crear el modelo optimizado para el sistema

En la raíz del Backend, crear un archivo llamado `Modelfile` con el siguiente contenido:

```
FROM qwen3:4b
PARAMETER num_gpu 99
PARAMETER num_ctx 4096
PARAMETER num_predict 512
PARAMETER temperature 0
PARAMETER top_k 10
PARAMETER top_p 0.5
PARAMETER repeat_penalty 1.3
```

Registrar el modelo:

```powershell
ollama create qwen3-rag -f Modelfile
```

Verificar que carga en GPU:

```powershell
ollama ps
```

La columna `PROCESSOR` debe mostrar `100% GPU`. El consumo esperado en VRAM es aproximadamente 3.3 GB. Si `nvidia-smi` reporta uso sostenido de 4096 MiB, reducir `num_ctx` a `2048` en el Modelfile y ejecutar `ollama create` nuevamente.

---

## 2. Base de datos

Crear la base de datos en PostgreSQL:

```sql
CREATE DATABASE db_tesis_cc;
```

Las tablas se crean automáticamente al primer arranque del backend. Para insertar el usuario administrador inicial, generar el hash de la contraseña desde Python:

```python
import bcrypt
print(bcrypt.hashpw(b"admin", bcrypt.gensalt()).decode())
```

Luego insertar el registro en psql o pgAdmin:

```sql
INSERT INTO usuarios (username, hashed_password, rol)
VALUES ('admin', '<hash_generado>', 'admin');
```

---

## 3. Backend

### 3.1 Entorno virtual e instalación

```powershell
cd Backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3.2 Archivo `.env`

Editar `Backend/.env` con los valores del entorno local:

```env
DATABASE_URL=postgresql://postgres:<contraseña>@localhost:5432/db_tesis_cc
FRONTEND_URL=http://localhost:5173
SECRET_KEY=<clave_aleatoria_256bits_hex>
ACCESS_TOKEN_EXPIRE_MINUTES=480
LLM_MODEL=qwen3-rag
EMBED_MODEL=nomic-ai/nomic-embed-text-v1.5
DOCUMENTS_DIR=./documents_local
VECTOR_STORE_DIR=./vector_store_local
OLLAMA_KEEP_ALIVE=60m
```

### 3.3 Levantar el servidor

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

El backend queda disponible en `http://localhost:8000`. La documentación de la API se encuentra en `http://localhost:8000/docs`.

Al iniciar, el servicio precalienta el modelo LLM para reducir la latencia de la primera consulta.

---

## 4. Frontend

### 4.1 Instalación

```powershell
cd Frontend
npm install
```

### 4.2 Archivo `.env`

```env
VITE_BACKEND_URL=http://localhost:8000
```

### 4.3 Levantar el servidor de desarrollo

```powershell
npm run dev
```

El frontend queda disponible en `http://localhost:5173`.

---

## 5. Verificación del sistema

Con backend y frontend activos, ejecutar:

```powershell
curl http://localhost:8000/
```

Respuesta esperada:

```json
{"status": "ok", "version": "4.0.0", "motor": "ollama-gpu + hf-cpu-embeddings"}
```

---

## 6. Pruebas de rendimiento

Los scripts se encuentran en `Backend/scripts/` y requieren que el backend esté en ejecución y que exista el archivo `banco_preguntas.json` en el mismo directorio.

Instalar la dependencia de métricas de GPU:

```powershell
pip install pynvml
```

### Prueba de comportamiento real

Simula 5 usuarios concurrentes con tiempos de espera entre preguntas (12–30 s). Cada pregunta del banco se consume una sola vez.

```powershell
cd Backend/scripts
python prueba_real.py
```

Genera `datos_real_<timestamp>.json`.

### Prueba de estrés

Ejecuta dos escenarios en secuencia: 5 usuarios durante 90 s con pausa aleatoria, seguido de 15 usuarios durante 60 s sin pausa.

```powershell
python prueba_estres.py
```

Genera `datos_estres_<timestamp>.json`.

### Generación de gráficas

```powershell
python generar_graficas.py datos_estres_<timestamp>.json datos_real_<timestamp>.json
```

Produce imágenes PNG y un reporte HTML con los resultados. Las métricas de GPU solo se registran si `pynvml` está instalado correctamente.