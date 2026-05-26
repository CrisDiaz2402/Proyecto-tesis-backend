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

ESCENARIO_A_USUARIOS  = 5
ESCENARIO_A_DURACION  = 90
ESCENARIO_A_PAUSA_MIN = 0.5
ESCENARIO_A_PAUSA_MAX = 1.5

ESCENARIO_B_USUARIOS  = 15
ESCENARIO_B_DURACION  = 60
ESCENARIO_B_PAUSA_MIN = 0.0
ESCENARIO_B_PAUSA_MAX = 0.0

INTERVALO_METRICAS_S = 1.0


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
            "gpu_util_pct":  util.gpu,
            "gpu_mem_used_mb": round(mem.used / 1024 / 1024, 1),
            "gpu_mem_total_mb": round(mem.total / 1024 / 1024, 1),
            "gpu_temp_c":    temp,
        }
    except Exception:
        return {}


def colector_metricas(fin_evento, buffer, t_referencia):
    psutil.cpu_percent(interval=None)
    while not fin_evento.is_set():
        t = time.time()
        vm  = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)
        muestra = {
            "t":            round(t - t_referencia, 2),
            "timestamp":    datetime.now().isoformat(),
            "cpu_pct":      cpu,
            "ram_used_mb":  round(vm.used / 1024 / 1024, 1),
            "ram_total_mb": round(vm.total / 1024 / 1024, 1),
            "ram_pct":      vm.percent,
        }
        muestra.update(_leer_gpu())
        buffer.append(muestra)
        fin_evento.wait(INTERVALO_METRICAS_S)


def hacer_pregunta(usuario_id, pregunta, num_request, headers, escenario):
    t0 = time.time()
    ts = datetime.now().isoformat()
    try:
        resp = requests.post(
            CHAT_ENDPOINT,
            json={"pregunta": pregunta},
            headers=headers,
            timeout=TIMEOUT_SEG
        )
        latencia_ms = round((time.time() - t0) * 1000)
        estado = "ok" if resp.status_code == 200 else f"error_{resp.status_code}"
        respuesta = resp.json().get("respuesta", "") if resp.status_code == 200 else ""
        return {
            "num": num_request, "escenario": escenario, "usuario_id": usuario_id,
            "timestamp": ts, "pregunta": pregunta, "respuesta": respuesta,
            "latencia_ms": latencia_ms, "estado": estado,
            "t_inicio": t0, "t_fin": time.time()
        }
    except requests.exceptions.Timeout:
        return {
            "num": num_request, "escenario": escenario, "usuario_id": usuario_id,
            "timestamp": ts, "pregunta": pregunta, "respuesta": "",
            "latencia_ms": TIMEOUT_SEG * 1000, "estado": "timeout",
            "t_inicio": t0, "t_fin": time.time()
        }
    except Exception:
        return {
            "num": num_request, "escenario": escenario, "usuario_id": usuario_id,
            "timestamp": ts, "pregunta": pregunta, "respuesta": "",
            "latencia_ms": TIMEOUT_SEG * 1000, "estado": "error_excepcion",
            "t_inicio": t0, "t_fin": time.time()
        }


def ejecutar_escenario(nombre, num_usuarios, duracion, pausa_min, pausa_max, pool, headers):
    limpiar_cache(headers)
    time.sleep(2)

    resultados_escenario = []
    metricas_escenario   = []
    lock_local = threading.Lock()
    fin_evento = threading.Event()

    t_inicio = time.time()

    hilo_metricas = threading.Thread(
        target=colector_metricas,
        args=(fin_evento, metricas_escenario, t_inicio),
        daemon=True
    )
    hilo_metricas.start()

    def worker_local(uid):
        pool_local = pool.copy()
        random.shuffle(pool_local)
        indice = 0
        while not fin_evento.is_set():
            pregunta = pool_local[indice % len(pool_local)]
            indice += 1
            with lock_local:
                num = len(resultados_escenario) + 1
            resultado = hacer_pregunta(uid, pregunta, num, headers, nombre)
            with lock_local:
                resultados_escenario.append(resultado)
                print(
                    f"[{nombre}] U{uid:02d} | "
                    f"{resultado['latencia_ms']:>6} ms | "
                    f"{resultado['estado']:<12} | "
                    f"{pregunta[:50]}"
                )
            if fin_evento.is_set():
                break
            if pausa_max > 0:
                time.sleep(random.uniform(pausa_min, pausa_max))

    hilos = [
        threading.Thread(target=worker_local, args=(i,), daemon=True)
        for i in range(1, num_usuarios + 1)
    ]
    for h in hilos:
        h.start()

    time.sleep(duracion)
    fin_evento.set()

    for h in hilos:
        h.join(timeout=TIMEOUT_SEG + 10)
    hilo_metricas.join(timeout=5)

    t_fin = time.time()
    ok      = sum(1 for r in resultados_escenario if r["estado"] == "ok")
    errores = sum(1 for r in resultados_escenario if r["estado"] != "ok")
    print(f"[{nombre}] {len(resultados_escenario)} solicitudes | OK: {ok} | Errores: {errores} | {round(t_fin - t_inicio, 2)}s")

    return resultados_escenario, metricas_escenario, t_inicio, t_fin


if __name__ == "__main__":
    ruta_banco = os.path.join(os.path.dirname(__file__), "banco_preguntas.json")
    try:
        with open(ruta_banco, "r", encoding="utf-8") as f:
            banco = json.load(f)
        POOL = banco["estres"]["preguntas"]
        print(f"{len(POOL)} preguntas cargadas.")
    except Exception as e:
        print(f"No se pudo cargar el banco: {e}")
        exit(1)

    token   = obtener_token()
    headers = {"Authorization": f"Bearer {token}"}

    res_a, met_a, t0_a, t1_a = ejecutar_escenario(
        nombre       = "CargaSostenida",
        num_usuarios = ESCENARIO_A_USUARIOS,
        duracion     = ESCENARIO_A_DURACION,
        pausa_min    = ESCENARIO_A_PAUSA_MIN,
        pausa_max    = ESCENARIO_A_PAUSA_MAX,
        pool         = POOL,
        headers      = headers
    )

    time.sleep(10)

    res_b, met_b, t0_b, t1_b = ejecutar_escenario(
        nombre       = "EstresReal",
        num_usuarios = ESCENARIO_B_USUARIOS,
        duracion     = ESCENARIO_B_DURACION,
        pausa_min    = ESCENARIO_B_PAUSA_MIN,
        pausa_max    = ESCENARIO_B_PAUSA_MAX,
        pool         = POOL,
        headers      = headers
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_json = f"datos_estres_{ts}.json"

    salida = {
        "tipo_prueba": "estres_dos_escenarios",
        "generado":    datetime.now().isoformat(),
        "escenario_A": {
            "nombre":            "CargaSostenida",
            "num_usuarios":      ESCENARIO_A_USUARIOS,
            "duracion_config_s": ESCENARIO_A_DURACION,
            "duracion_real_s":   round(t1_a - t0_a, 2),
            "pausa_min_s":       ESCENARIO_A_PAUSA_MIN,
            "pausa_max_s":       ESCENARIO_A_PAUSA_MAX,
            "solicitudes":       res_a,
            "metricas_sistema":  met_a,
        },
        "escenario_B": {
            "nombre":            "EstresReal",
            "num_usuarios":      ESCENARIO_B_USUARIOS,
            "duracion_config_s": ESCENARIO_B_DURACION,
            "duracion_real_s":   round(t1_b - t0_b, 2),
            "pausa_min_s":       ESCENARIO_B_PAUSA_MIN,
            "pausa_max_s":       ESCENARIO_B_PAUSA_MAX,
            "solicitudes":       res_b,
            "metricas_sistema":  met_b,
        }
    }

    with open(nombre_json, "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)

    total = len(res_a) + len(res_b)
    print(f"Total: {total} solicitudes. Archivo: {nombre_json}")