import json
import os
import random
import threading
import time
from datetime import datetime

import psutil
import requests

BACKEND_URL    = "http://localhost:8000"
AUTH_ENDPOINT  = f"{BACKEND_URL}/api/auth/login"
CHAT_ENDPOINT  = f"{BACKEND_URL}/api/chat/consultar"
CACHE_ENDPOINT = f"{BACKEND_URL}/api/documents/cache/all"
TIMEOUT_SEG    = 120
USUARIO_ADMIN  = "admin"
CLAVE_ADMIN    = "admin"

NUM_USUARIOS = 5
THINK_MIN    = 12
THINK_MAX    = 30

INTERVALO_METRICAS_S = 1.0

_lock_cola  = threading.Lock()
_lock_res   = threading.Lock()
resultados  = []
cola_global = []
ts_inicio   = None
ts_fin      = None


def obtener_token():
    r = requests.post(
        AUTH_ENDPOINT,
        json={"username": USUARIO_ADMIN, "password": CLAVE_ADMIN},
        timeout=10
    )
    if r.status_code == 200:
        return r.json().get("access_token")
    print(f"Login fallido: {r.status_code}")
    exit(1)


def limpiar_cache(headers):
    try:
        r = requests.delete(CACHE_ENDPOINT, headers=headers, timeout=15)
        if r.status_code not in (200, 204):
            print(f"Advertencia cache: {r.status_code}")
    except Exception as e:
        print(f"Error limpiando cache: {e}")


def _leer_gpu():
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util   = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem    = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp   = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        return {
            "gpu_util_pct":     util.gpu,
            "gpu_mem_used_mb":  round(mem.used / 1024 / 1024, 1),
            "gpu_mem_total_mb": round(mem.total / 1024 / 1024, 1),
            "gpu_temp_c":       temp,
        }
    except Exception:
        return {}


def colector_metricas(fin_evento, buffer, t_referencia):
    psutil.cpu_percent(interval=None)
    while not fin_evento.is_set():
        t  = time.time()
        vm = psutil.virtual_memory()
        muestra = {
            "t":            round(t - t_referencia, 2),
            "timestamp":    datetime.now().isoformat(),
            "cpu_pct":      psutil.cpu_percent(interval=None),
            "ram_used_mb":  round(vm.used / 1024 / 1024, 1),
            "ram_total_mb": round(vm.total / 1024 / 1024, 1),
            "ram_pct":      vm.percent,
        }
        muestra.update(_leer_gpu())
        buffer.append(muestra)
        fin_evento.wait(INTERVALO_METRICAS_S)


def hacer_pregunta(usuario_id, pregunta_dict, num_pregunta, think_time, headers):
    pregunta = pregunta_dict["pregunta"]
    tipo     = pregunta_dict["tipo"]
    t0       = time.time()
    ts       = datetime.now().isoformat()
    try:
        resp = requests.post(
            CHAT_ENDPOINT,
            json={"pregunta": pregunta},
            headers=headers,
            timeout=TIMEOUT_SEG
        )
        latencia_ms = round((time.time() - t0) * 1000)
        if resp.status_code == 200:
            datos = resp.json()
            return {
                "usuario_id": usuario_id, "num_pregunta": num_pregunta,
                "tipo": tipo, "timestamp": ts, "pregunta": pregunta,
                "respuesta": datos.get("respuesta", ""), "latencia_ms": latencia_ms,
                "think_time_s": round(think_time, 1), "estado": "ok",
                "t_inicio": t0, "t_fin": time.time()
            }
        return {
            "usuario_id": usuario_id, "num_pregunta": num_pregunta,
            "tipo": tipo, "timestamp": ts, "pregunta": pregunta,
            "respuesta": "", "latencia_ms": latencia_ms,
            "think_time_s": round(think_time, 1), "estado": f"error_{resp.status_code}",
            "t_inicio": t0, "t_fin": time.time()
        }
    except requests.exceptions.Timeout:
        return {
            "usuario_id": usuario_id, "num_pregunta": num_pregunta,
            "tipo": tipo, "timestamp": ts, "pregunta": pregunta,
            "respuesta": "", "latencia_ms": TIMEOUT_SEG * 1000,
            "think_time_s": round(think_time, 1), "estado": "timeout",
            "t_inicio": t0, "t_fin": time.time()
        }
    except Exception:
        return {
            "usuario_id": usuario_id, "num_pregunta": num_pregunta,
            "tipo": tipo, "timestamp": ts, "pregunta": pregunta,
            "respuesta": "", "latencia_ms": TIMEOUT_SEG * 1000,
            "think_time_s": round(think_time, 1), "estado": "error_excepcion",
            "t_inicio": t0, "t_fin": time.time()
        }


def worker_usuario(usuario_id, headers, fin_evento):
    num_local = 0
    while not fin_evento.is_set():
        with _lock_cola:
            if not cola_global:
                break
            idx, pregunta_dict = cola_global.pop(0)

        think = random.uniform(THINK_MIN, THINK_MAX) if num_local > 0 else random.uniform(1, 3)
        if think > 0:
            fin_evento.wait(think)
            if fin_evento.is_set():
                break

        num_local += 1
        resultado = hacer_pregunta(usuario_id, pregunta_dict, idx + 1, think, headers)

        with _lock_res:
            resultados.append(resultado)
            print(
                f"U{usuario_id} | [{resultado['tipo']:<18}] | "
                f"{resultado['latencia_ms']:>6} ms | "
                f"{resultado['estado']:<10} | "
                f"{resultado['pregunta'][:45]}"
            )


if __name__ == "__main__":
    ruta_banco = os.path.join(os.path.dirname(__file__), "banco_preguntas.json")
    try:
        with open(ruta_banco, "r", encoding="utf-8") as f:
            banco = json.load(f)
        preguntas = banco["real"]["preguntas"]
        print(f"{len(preguntas)} preguntas cargadas.")
    except Exception as e:
        print(f"No se pudo cargar el banco: {e}")
        exit(1)

    preguntas_mezcladas = preguntas.copy()
    random.shuffle(preguntas_mezcladas)
    for i, p in enumerate(preguntas_mezcladas):
        cola_global.append((i, p))

    token   = obtener_token()
    headers = {"Authorization": f"Bearer {token}"}

    limpiar_cache(headers)
    time.sleep(2)

    metricas_sistema = []
    fin_metricas     = threading.Event()
    ts_inicio        = time.time()

    hilo_metricas = threading.Thread(
        target=colector_metricas,
        args=(fin_metricas, metricas_sistema, ts_inicio),
        daemon=True
    )
    hilo_metricas.start()

    fin_evento = threading.Event()
    hilos = [
        threading.Thread(
            target=worker_usuario,
            args=(i, headers, fin_evento),
            daemon=True
        )
        for i in range(1, NUM_USUARIOS + 1)
    ]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join(timeout=TIMEOUT_SEG * len(preguntas))

    fin_evento.set()
    fin_metricas.set()
    hilo_metricas.join(timeout=5)
    ts_fin = time.time()

    ts_str      = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_json = f"datos_real_{ts_str}.json"

    ok      = sum(1 for r in resultados if r["estado"] == "ok")
    errores = sum(1 for r in resultados if r["estado"] != "ok")

    salida = {
        "tipo_prueba":           "comportamiento_real",
        "generado":              datetime.now().isoformat(),
        "num_usuarios":          NUM_USUARIOS,
        "total_preguntas_banco": len(preguntas),
        "duracion_real_s":       round(ts_fin - ts_inicio, 2),
        "resumen": {
            "total": len(resultados), "ok": ok, "errores": errores
        },
        "preguntas":        resultados,
        "metricas_sistema": metricas_sistema,
    }

    with open(nombre_json, "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)

    print(f"{len(resultados)} preguntas — OK: {ok}, Errores: {errores} — {round(ts_fin - ts_inicio, 1)}s")
    print(f"Archivo: {nombre_json}")