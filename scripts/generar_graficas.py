"""
generar_graficas.py
Uso:
    python generar_graficas.py datos_estres_XXX.json [datos_real_XXX.json]

Produce graficas PNG y un reporte HTML en el directorio de salida configurado.
"""

import sys
import json
import os
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "figure.dpi":        150,
    "figure.autolayout": True,
})

UMBRAL_CACHE_MS = 2500
OUT_DIR         = "reportes_avatar"


# ---------------------------------------------------------------------------
# Carga y normalización
# ---------------------------------------------------------------------------

def cargar_json(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


def es_cache_hit(row):
    return (row["estado"] == "ok") and (row["latencia_ms"] < UMBRAL_CACHE_MS)


def cargar_datos_estres(datos):
    frames = {}
    meta   = {}
    for key, label in [("escenario_A", "A"), ("escenario_B", "B")]:
        if key not in datos or "solicitudes" not in datos[key]:
            continue
        df = pd.DataFrame(datos[key]["solicitudes"])
        if df.empty:
            continue
        df["origen"]       = label
        df["nombre"]       = datos[key].get("nombre", key)
        df["es_cache_hit"] = df.apply(es_cache_hit, axis=1)
        df["es_ok"]        = df["estado"] == "ok"
        df["es_timeout"]   = df["estado"] == "timeout"
        frames[label] = df
        meta[label] = {
            "num_usuarios":      datos[key].get("num_usuarios", "?"),
            "duracion_config_s": datos[key].get("duracion_config_s", "?"),
            "duracion_real_s":   datos[key].get("duracion_real_s", "?"),
            "pausa_min_s":       datos[key].get("pausa_min_s", 0),
            "pausa_max_s":       datos[key].get("pausa_max_s", 0),
        }
    return frames, meta


def cargar_datos_real(datos):
    if "preguntas" not in datos:
        return None, {}
    df = pd.DataFrame(datos["preguntas"])
    if df.empty:
        return None, {}
    df["es_cache_hit"] = df.apply(es_cache_hit, axis=1)
    df["es_ok"]        = df["estado"] == "ok"
    df["es_timeout"]   = df["estado"] == "timeout"
    meta = {
        "num_usuarios":          datos.get("num_usuarios", "?"),
        "total_preguntas_banco": datos.get("total_preguntas_banco", "?"),
        "duracion_real_s":       datos.get("duracion_real_s", "?"),
    }
    return df, meta


def cargar_metricas_sistema(datos_estres, datos_real):
    """
    Devuelve un dict con DataFrames de metricas de sistema por fuente.
    Claves: 'A', 'B', 'real'
    """
    resultado = {}
    for key, label in [("escenario_A", "A"), ("escenario_B", "B")]:
        mlist = datos_estres.get(key, {}).get("metricas_sistema", [])
        if mlist:
            resultado[label] = pd.DataFrame(mlist)
    if datos_real:
        mlist = datos_real.get("metricas_sistema", [])
        if mlist:
            resultado["real"] = pd.DataFrame(mlist)
    return resultado


def calcular_metricas(df):
    total    = len(df)
    ok       = int(df["es_ok"].sum())
    timeouts = int(df["es_timeout"].sum())
    errores  = total - ok - timeouts
    tasa_error = round((timeouts + errores) / total * 100, 1) if total else 0
    df_ok    = df[df["es_ok"]]
    hit_rate = round(df_ok["es_cache_hit"].mean() * 100, 1) if not df_ok.empty else 0.0
    p50 = int(df["latencia_ms"].median())
    p95 = int(df["latencia_ms"].quantile(0.95))
    p99 = int(df["latencia_ms"].quantile(0.99))
    return {
        "total": total, "ok": ok, "timeouts": timeouts,
        "errores": errores, "tasa_error": tasa_error,
        "hit_rate": hit_rate, "p50": p50, "p95": p95, "p99": p99,
    }


# ---------------------------------------------------------------------------
# Graficas de rendimiento de solicitudes
# ---------------------------------------------------------------------------

def g1_cdf_comparativa(frames_estres, df_real, out_dir):
    """
    CDF acumulada de latencia para todos los escenarios.
    Incluye timeouts. Permite comparar distribucion global de tiempos de respuesta.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    config = {
        "A":    ("steelblue", "Carga Sostenida (5 usuarios)"),
        "B":    ("crimson",   "Estres Real (15 usuarios)"),
        "real": ("seagreen",  "Comportamiento Real (5 usuarios)"),
    }

    datasets = dict(frames_estres)
    if df_real is not None:
        datasets["real"] = df_real

    for key, df in datasets.items():
        lat = np.sort(df["latencia_ms"].values)
        pct = np.linspace(0, 100, len(lat))
        col, etiqueta = config.get(key, ("gray", key))
        ax.plot(lat, pct, linewidth=2.5, color=col, label=etiqueta)
        p50 = np.percentile(lat, 50)
        p95 = np.percentile(lat, 95)
        ax.axvline(p50, color=col, linestyle="--", alpha=0.4, linewidth=1)
        ax.axvline(p95, color=col, linestyle=":",  alpha=0.4, linewidth=1)

    ax.axvline(120000, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Limite timeout (120 s)")
    ax.set_title("Funcion de Distribucion Acumulada de Latencia")
    ax.set_xlabel("Latencia (ms)")
    ax.set_ylabel("Solicitudes acumuladas (%)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 103)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}s"))

    ruta = os.path.join(out_dir, "1_cdf_latencia.png")
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


def g2_throughput_tiempo(frames_estres, out_dir):
    """
    Solicitudes completadas por ventana de 10 s. Barras apiladas OK / Timeout.
    Revela el momento exacto en que el sistema empieza a saturarse.
    """
    VENTANA_S = 10
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    colores = {"A": "steelblue", "B": "crimson"}
    nombres = {"A": "Carga Sostenida", "B": "Estres Real"}

    for idx, (key, df) in enumerate(frames_estres.items()):
        ax  = axes[idx]
        col = colores.get(key, "gray")
        t0  = df["t_inicio"].min()
        df  = df.copy()
        df["t_rel"] = df["t_fin"] - t0
        t_max    = df["t_rel"].max()
        ventanas = np.arange(0, t_max + VENTANA_S, VENTANA_S)
        ok_counts, to_counts, midpoints = [], [], []
        for i in range(len(ventanas) - 1):
            m = (df["t_rel"] >= ventanas[i]) & (df["t_rel"] < ventanas[i+1])
            ok_counts.append(int(df[m]["es_ok"].sum()))
            to_counts.append(int(df[m]["es_timeout"].sum()))
            midpoints.append((ventanas[i] + ventanas[i+1]) / 2)
        ax.bar(midpoints, ok_counts, width=VENTANA_S*0.8, color=col, alpha=0.8, label="OK")
        ax.bar(midpoints, to_counts, width=VENTANA_S*0.8, bottom=ok_counts,
               color="#cc0000", alpha=0.85, label="Timeout")
        ax.set_title(f"{nombres.get(key, key)}\n(OK: {sum(ok_counts)} | Timeouts: {sum(to_counts)})")
        ax.set_xlabel("Tiempo transcurrido (s)")
        ax.set_ylabel("Solicitudes por ventana de 10 s")
        ax.legend(fontsize=9)

    fig.suptitle("Throughput por Ventana de Tiempo — OK vs Timeouts", fontsize=13, fontweight="bold")
    ruta = os.path.join(out_dir, "2_throughput.png")
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


def g3_latencia_scatter_tiempo(frames_estres, out_dir):
    """
    Scatter latencia vs tiempo relativo con mediana movil.
    Detecta degradacion progresiva del sistema bajo carga.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=False)
    colores = {"A": "steelblue", "B": "crimson"}
    nombres = {"A": "Carga Sostenida (5 usuarios)", "B": "Estres Real (15 usuarios)"}

    for idx, (key, df) in enumerate(frames_estres.items()):
        ax  = axes[idx]
        t0  = df["t_inicio"].min()
        df  = df.copy()
        df["t_rel_s"] = df["t_fin"] - t0
        col = colores.get(key, "gray")

        df_ok    = df[df["es_ok"]]
        df_to    = df[df["es_timeout"]]
        df_cache = df_ok[df_ok["es_cache_hit"]]
        df_llm   = df_ok[~df_ok["es_cache_hit"]]

        ax.scatter(df_llm["t_rel_s"],   df_llm["latencia_ms"],   c=col,       alpha=0.7, s=28, label="LLM", zorder=3)
        ax.scatter(df_cache["t_rel_s"], df_cache["latencia_ms"], c="#27ae60", alpha=0.9, s=38, marker="*", label="Cache hit", zorder=4)
        if not df_to.empty:
            ax.scatter(df_to["t_rel_s"], df_to["latencia_ms"],   c="#cc0000", alpha=0.9, s=55, marker="X", label="Timeout", zorder=5)

        df_ok_sorted = df_ok.sort_values("t_rel_s")
        if len(df_ok_sorted) >= 5:
            roll = df_ok_sorted["latencia_ms"].rolling(5, center=True).median()
            ax.plot(df_ok_sorted["t_rel_s"], roll, color="black", linewidth=1.5, alpha=0.5, label="Mediana movil (n=5)")

        ax.axhline(UMBRAL_CACHE_MS, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_title(nombres.get(key, key))
        ax.set_xlabel("Tiempo relativo de finalizacion (s)")
        ax.set_ylabel("Latencia (ms)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v/1000)}s"))
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Evolucion de Latencia durante la Prueba", fontsize=13, fontweight="bold")
    ruta = os.path.join(out_dir, "3_latencia_tiempo.png")
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


def g4_latencia_por_tipo(df_real, out_dir):
    """
    Latencia mediana por tipo de pregunta con IQR. Solo prueba real.
    """
    if df_real is None or "tipo" not in df_real.columns:
        return None

    colores_tipo = {
        "Factual":          "#4472C4",
        "Correccion":       "#ED7D31",
        "Anti_alucinacion": "#70AD47",
        "Fuera_dominio":    "#E04040",
    }

    resumen = (
        df_real.groupby("tipo")["latencia_ms"]
        .agg(mediana="median",
             p25=lambda x: x.quantile(0.25),
             p75=lambda x: x.quantile(0.75),
             n="count")
        .reset_index()
        .sort_values("mediana", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (_, row) in enumerate(resumen.iterrows()):
        color = colores_tipo.get(row["tipo"], "#888")
        ax.barh(i, row["mediana"], color=color, alpha=0.85, edgecolor="white")
        ax.errorbar(row["mediana"], i,
                    xerr=[[row["mediana"] - row["p25"]], [row["p75"] - row["mediana"]]],
                    fmt="none", color="black", capsize=5, linewidth=1.5)
        ax.text(row["mediana"] + 500, i,
                f"{int(row['mediana']/1000)}s  (n={int(row['n'])})",
                va="center", fontsize=10)

    ax.set_yticks(range(len(resumen)))
    ax.set_yticklabels(resumen["tipo"].tolist())
    ax.set_title("Latencia Mediana por Tipo de Pregunta (IQR p25-p75)")
    ax.set_xlabel("Latencia mediana (ms)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}s"))

    ruta = os.path.join(out_dir, "4_latencia_por_tipo.png")
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


# ---------------------------------------------------------------------------
# Graficas de metricas de sistema
# ---------------------------------------------------------------------------

def _tiene_gpu(df_met):
    return "gpu_util_pct" in df_met.columns and df_met["gpu_util_pct"].notna().any()


def g5_cpu_ram_tiempo(metricas_sistema, frames_estres, df_real, out_dir):
    """
    CPU (%) y RAM (%) vs tiempo para cada escenario.
    Superpone marcas de timeout para correlacionar saturacion de recursos con fallos.
    """
    fuentes = []
    for key in ["A", "B"]:
        if key in metricas_sistema and key in frames_estres:
            fuentes.append((key, metricas_sistema[key], frames_estres[key]))
    if df_real is not None and "real" in metricas_sistema:
        fuentes.append(("real", metricas_sistema["real"], df_real))

    if not fuentes:
        return None

    colores_escenario = {"A": "steelblue", "B": "crimson", "real": "seagreen"}
    nombres_escenario = {"A": "Carga Sostenida", "B": "Estres Real", "real": "Comportamiento Real"}

    fig, axes = plt.subplots(len(fuentes), 2, figsize=(13, 4 * len(fuentes)), sharex=False)
    if len(fuentes) == 1:
        axes = [axes]

    for row_idx, (key, df_met, df_sol) in enumerate(fuentes):
        col    = colores_escenario.get(key, "gray")
        nombre = nombres_escenario.get(key, key)
        ax_cpu = axes[row_idx][0]
        ax_ram = axes[row_idx][1]

        ax_cpu.plot(df_met["t"], df_met["cpu_pct"], color=col, linewidth=1.5, label="CPU %")
        ax_cpu.set_ylabel("CPU (%)")
        ax_cpu.set_ylim(0, 105)
        ax_cpu.set_title(f"{nombre} — CPU")

        ax_ram.plot(df_met["t"], df_met["ram_pct"], color="darkorange", linewidth=1.5, label="RAM %")
        ax_ram.set_ylabel("RAM (%)")
        ax_ram.set_ylim(0, 105)
        ax_ram.set_title(f"{nombre} — RAM")

        t0_sol = df_sol["t_inicio"].min()
        df_to  = df_sol[df_sol["es_timeout"]]
        for _, r in df_to.iterrows():
            t_rel = r["t_fin"] - t0_sol
            ax_cpu.axvline(t_rel, color="red", alpha=0.35, linewidth=0.8)
            ax_ram.axvline(t_rel, color="red", alpha=0.35, linewidth=0.8)

        for ax in [ax_cpu, ax_ram]:
            ax.set_xlabel("Tiempo (s)")
            if not df_to.empty:
                ax.axvline(-9999, color="red", alpha=0.5, linewidth=0.8, label="Timeout")
            ax.legend(fontsize=8)

    fig.suptitle("Uso de CPU y RAM durante las Pruebas", fontsize=13, fontweight="bold")
    ruta = os.path.join(out_dir, "5_cpu_ram.png")
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


def g6_gpu_tiempo(metricas_sistema, frames_estres, df_real, out_dir):
    """
    Utilizacion GPU (%), memoria GPU y temperatura por escenario.
    Solo se genera si el servidor tiene GPU con soporte pynvml.
    """
    fuentes = []
    for key in ["A", "B"]:
        if key in metricas_sistema and _tiene_gpu(metricas_sistema[key]):
            df_sol = frames_estres.get(key)
            if df_sol is not None:
                fuentes.append((key, metricas_sistema[key], df_sol))
    if df_real is not None and "real" in metricas_sistema and _tiene_gpu(metricas_sistema["real"]):
        fuentes.append(("real", metricas_sistema["real"], df_real))

    if not fuentes:
        return None

    colores_escenario = {"A": "steelblue", "B": "crimson", "real": "seagreen"}
    nombres_escenario = {"A": "Carga Sostenida", "B": "Estres Real", "real": "Comportamiento Real"}

    fig, axes = plt.subplots(len(fuentes), 3, figsize=(15, 4 * len(fuentes)), sharex=False)
    if len(fuentes) == 1:
        axes = [axes]

    for row_idx, (key, df_met, df_sol) in enumerate(fuentes):
        col    = colores_escenario.get(key, "gray")
        nombre = nombres_escenario.get(key, key)
        ax_u, ax_m, ax_t = axes[row_idx]

        ax_u.plot(df_met["t"], df_met["gpu_util_pct"],    color=col,          linewidth=1.5)
        ax_m.plot(df_met["t"], df_met["gpu_mem_used_mb"], color="darkorchid",  linewidth=1.5)
        ax_t.plot(df_met["t"], df_met["gpu_temp_c"],      color="darkorange",  linewidth=1.5)

        ax_u.set_title(f"{nombre} — GPU Utilizacion (%)")
        ax_m.set_title(f"{nombre} — GPU Memoria (MB)")
        ax_t.set_title(f"{nombre} — GPU Temperatura (C)")

        t0_sol = df_sol["t_inicio"].min()
        df_to  = df_sol[df_sol["es_timeout"]]
        for _, r in df_to.iterrows():
            t_rel = r["t_fin"] - t0_sol
            for ax in [ax_u, ax_m, ax_t]:
                ax.axvline(t_rel, color="red", alpha=0.35, linewidth=0.8)

        for ax in [ax_u, ax_m, ax_t]:
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("")

    fig.suptitle("Metricas de GPU durante las Pruebas (lineas rojas = timeouts)", fontsize=13, fontweight="bold")
    ruta = os.path.join(out_dir, "6_gpu.png")
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


def g7_latencia_vs_cpu(metricas_sistema, frames_estres, out_dir):
    """
    Scatter latencia de cada solicitud vs CPU % promedio en la ventana de esa solicitud.
    Evidencia si la saturacion de CPU explica los picos de latencia.
    """
    fuentes = [(k, frames_estres[k], metricas_sistema[k])
               for k in ["A", "B"]
               if k in frames_estres and k in metricas_sistema]
    if not fuentes:
        return None

    colores = {"A": "steelblue", "B": "crimson"}
    nombres = {"A": "Carga Sostenida", "B": "Estres Real"}

    fig, axes = plt.subplots(1, len(fuentes), figsize=(7 * len(fuentes), 5), sharey=False)
    if len(fuentes) == 1:
        axes = [axes]

    for idx, (key, df_sol, df_met) in enumerate(fuentes):
        ax = axes[idx]
        t0 = df_sol["t_inicio"].min()

        filas = []
        for _, r in df_sol.iterrows():
            t_rel_ini = r["t_inicio"] - t0
            t_rel_fin = r["t_fin"]    - t0
            mascara = (df_met["t"] >= t_rel_ini) & (df_met["t"] <= t_rel_fin)
            cpu_mean = df_met[mascara]["cpu_pct"].mean() if mascara.any() else np.nan
            filas.append({"latencia_ms": r["latencia_ms"], "cpu_mean": cpu_mean, "estado": r["estado"]})

        df_scatter = pd.DataFrame(filas).dropna()
        df_ok  = df_scatter[df_scatter["estado"] == "ok"]
        df_to  = df_scatter[df_scatter["estado"] == "timeout"]

        ax.scatter(df_ok["cpu_mean"],  df_ok["latencia_ms"],  c=colores[key], alpha=0.6, s=30, label="OK")
        if not df_to.empty:
            ax.scatter(df_to["cpu_mean"], df_to["latencia_ms"], c="#cc0000", alpha=0.9, s=55, marker="X", label="Timeout")

        if len(df_ok) > 3:
            z = np.polyfit(df_ok["cpu_mean"].dropna(), df_ok["latencia_ms"].dropna(), 1)
            xr = np.linspace(df_ok["cpu_mean"].min(), df_ok["cpu_mean"].max(), 100)
            ax.plot(xr, np.polyval(z, xr), color="black", linewidth=1.2, linestyle="--", alpha=0.6, label="Tendencia")

        ax.set_title(nombres.get(key, key))
        ax.set_xlabel("CPU promedio durante solicitud (%)")
        ax.set_ylabel("Latencia (ms)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v/1000)}s"))
        ax.legend(fontsize=9)

    fig.suptitle("Latencia vs Uso de CPU por Solicitud", fontsize=13, fontweight="bold")
    ruta = os.path.join(out_dir, "7_latencia_vs_cpu.png")
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


# ---------------------------------------------------------------------------
# Reporte HTML
# ---------------------------------------------------------------------------

def construir_html(graficas, frames_estres, meta_estres, df_real, meta_real, out_dir):
    estilos = """
    body  { font-family:'Times New Roman',serif; background:#fff; color:#111;
            max-width:1150px; margin:auto; padding:40px; line-height:1.7; }
    h1    { border-bottom:3px solid #000; padding-bottom:12px; text-align:center; }
    h2    { border-bottom:1px solid #aaa; margin-top:45px; }
    h3    { margin-top:28px; color:#222; }
    .metrics { display:flex; flex-wrap:wrap; gap:10px; background:#f4f4f4;
               border:1px solid #ddd; padding:20px; border-radius:6px; margin:16px 0; }
    .m    { text-align:center; flex:1; min-width:110px; }
    .m h3 { margin:0; font-size:22px; color:#333; }
    .m p  { margin:4px 0 0; font-size:12px; text-transform:uppercase; color:#777; }
    .m.bad  h3 { color:#c0392b; }
    .m.warn h3 { color:#e67e22; }
    .m.good h3 { color:#27ae60; }
    .img  { text-align:center; margin:28px 0; }
    .img img { max-width:100%; border:1px solid #eee; box-shadow:0 4px 12px rgba(0,0,0,.07); }
    .caption { font-size:12px; color:#666; margin-top:6px; font-style:italic; max-width:800px; margin:6px auto 0; }
    footer { margin-top:50px; text-align:center; font-size:11px; color:#aaa;
             border-top:1px solid #eee; padding-top:16px; }
    table  { border-collapse:collapse; width:100%; margin:16px 0; font-size:13px; }
    th, td { border:1px solid #ddd; padding:8px 12px; text-align:center; }
    th     { background:#f0f0f0; }
    tr:nth-child(even) { background:#fafafa; }
    """

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Analisis de Rendimiento — Sistema Avatar EPN</title>
  <style>{estilos}</style>
</head>
<body>
<h1>Analisis Consolidado de Rendimiento — Sistema Avatar EPN</h1>
<p style="text-align:center;color:#555;">Generado: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

<h2>Indicadores Globales por Escenario</h2>
<table>
  <thead>
    <tr>
      <th>Escenario</th><th>Usuarios</th><th>Total</th>
      <th>OK</th><th>Timeouts</th><th>Tasa fallo</th>
      <th>Cache hit rate*</th><th>p50</th><th>p95</th>
    </tr>
  </thead>
  <tbody>
"""
    for key, df in frames_estres.items():
        m  = calcular_metricas(df)
        mu = meta_estres.get(key, {})
        color_to   = "#c0392b" if m["timeouts"] > 0 else "#27ae60"
        color_err  = "#c0392b" if m["tasa_error"] > 10 else ("#e67e22" if m["tasa_error"] > 0 else "#27ae60")
        html += f"""
    <tr>
      <td><strong>{df['nombre'].iloc[0]}</strong></td>
      <td>{mu.get('num_usuarios','?')}</td>
      <td>{m['total']}</td>
      <td style="color:#27ae60"><strong>{m['ok']}</strong></td>
      <td style="color:{color_to}"><strong>{m['timeouts']}</strong></td>
      <td style="color:{color_err}">{m['tasa_error']}%</td>
      <td>{m['hit_rate']}%</td>
      <td>{int(m['p50']/1000)}s ({m['p50']} ms)</td>
      <td>{int(m['p95']/1000)}s ({m['p95']} ms)</td>
    </tr>"""

    if df_real is not None:
        m  = calcular_metricas(df_real)
        mu = meta_real
        html += f"""
    <tr>
      <td><strong>Comportamiento Real</strong></td>
      <td>{mu.get('num_usuarios','?')}</td>
      <td>{m['total']}</td>
      <td style="color:#27ae60"><strong>{m['ok']}</strong></td>
      <td style="color:{'#c0392b' if m['timeouts']>0 else '#27ae60'}"><strong>{m['timeouts']}</strong></td>
      <td style="color:#27ae60">{m['tasa_error']}%</td>
      <td>{m['hit_rate']}%</td>
      <td>{int(m['p50']/1000)}s ({m['p50']} ms)</td>
      <td>{int(m['p95']/1000)}s ({m['p95']} ms)</td>
    </tr>"""

    html += """
  </tbody>
</table>
<p style="font-size:12px;color:#888;">* Cache hit rate calculado sobre solicitudes OK. p50/p95 incluyen timeouts.</p>
"""

    descripciones = {
        "1_cdf_latencia.png": (
            "CDF de Latencia — Comparativa de Escenarios",
            "Funcion de distribucion acumulada de latencias. Una curva mas hacia la izquierda "
            "indica menor latencia. El escalon brusco al final de la curva de Estres Real "
            "corresponde a los timeouts que alcanzan los 120 s. Las lineas discontinuas marcan p50 y p95."
        ),
        "2_throughput.png": (
            "Throughput por Ventana de Tiempo — OK vs Timeouts",
            "Solicitudes completadas cada 10 segundos. Permite identificar en que momento "
            "de la prueba el sistema empieza a saturarse y los timeouts comienzan a aparecer."
        ),
        "3_latencia_tiempo.png": (
            "Evolucion de Latencia a lo Largo del Tiempo",
            "Scatter de latencia por tiempo relativo de finalizacion. "
            "La linea negra es la mediana movil (ventana=5). Permite observar si la "
            "latencia se degrada progresivamente bajo carga sostenida. "
            "Verde = cache hit. Rojo X = timeout."
        ),
        "4_latencia_por_tipo.png": (
            "Latencia Mediana por Tipo de Pregunta — Comportamiento Real",
            "Comparacion de latencias medianas por categoria de consulta. "
            "Las barras de error representan el rango intercuartilico p25-p75."
        ),
        "5_cpu_ram.png": (
            "Uso de CPU y RAM durante las Pruebas",
            "Series de tiempo de CPU y RAM. Las lineas rojas verticales indican "
            "el momento de finalizacion de cada timeout, permitiendo correlacionar "
            "saturacion de recursos con degradacion del servicio."
        ),
        "6_gpu.png": (
            "Metricas de GPU durante las Pruebas",
            "Utilizacion (%), memoria usada (MB) y temperatura (C) de la GPU. "
            "Critico para evaluar la capacidad del modelo de lenguaje bajo carga concurrente."
        ),
        "7_latencia_vs_cpu.png": (
            "Latencia de Solicitud vs CPU Promedio durante la Solicitud",
            "Cada punto representa una solicitud. El eje X es el CPU promedio del servidor "
            "mientras esa solicitud estaba activa. La linea de tendencia indica si "
            "existe correlacion entre uso de CPU y tiempo de respuesta."
        ),
    }

    html += "<h2>Graficas de Analisis</h2>\n"
    basenames = {os.path.basename(g) for g in graficas if g}
    for nombre_img, (titulo, caption) in descripciones.items():
        if nombre_img in basenames:
            html += f"""
<h3>{titulo}</h3>
<div class="img">
  <img src="{nombre_img}" alt="{titulo}">
  <p class="caption">{caption}</p>
</div>
"""

    html += f"""
<footer>
  Sistema Avatar Multimodal — EPN &nbsp;|&nbsp; {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
</footer>
</body>
</html>"""

    ruta = os.path.join(out_dir, "reporte_rendimiento.html")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(html)
    return ruta


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Uso: python generar_graficas.py datos_estres_XXX.json [datos_real_XXX.json]")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    datos_estres = cargar_json(sys.argv[1])
    frames_estres, meta_estres = cargar_datos_estres(datos_estres)
    if not frames_estres:
        print("No se encontraron solicitudes en el archivo de estres.")
        sys.exit(1)

    datos_real = None
    df_real, meta_real = None, {}
    if len(sys.argv) > 2:
        datos_real = cargar_json(sys.argv[2])
        if datos_real:
            df_real, meta_real = cargar_datos_real(datos_real)

    metricas_sistema = cargar_metricas_sistema(datos_estres, datos_real)

    graficas = []
    graficas.append(g1_cdf_comparativa(frames_estres, df_real, OUT_DIR))
    graficas.append(g2_throughput_tiempo(frames_estres, OUT_DIR))
    graficas.append(g3_latencia_scatter_tiempo(frames_estres, OUT_DIR))

    if df_real is not None:
        graficas.append(g4_latencia_por_tipo(df_real, OUT_DIR))

    graficas.append(g5_cpu_ram_tiempo(metricas_sistema, frames_estres, df_real, OUT_DIR))
    graficas.append(g6_gpu_tiempo(metricas_sistema, frames_estres, df_real, OUT_DIR))
    graficas.append(g7_latencia_vs_cpu(metricas_sistema, frames_estres, OUT_DIR))

    ruta_html = construir_html(
        graficas, frames_estres, meta_estres,
        df_real, meta_real, OUT_DIR
    )

    generadas = sum(1 for g in graficas if g)
    print(f"{generadas} graficas + reporte HTML generados en '{OUT_DIR}/'")
    print(f"Reporte: {ruta_html}")


if __name__ == "__main__":
    main()