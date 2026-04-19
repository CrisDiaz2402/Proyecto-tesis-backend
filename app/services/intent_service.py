# app/services/intent_service.py

from __future__ import annotations


def _normalizar(texto: str) -> str:
    texto = texto.lower().strip()
    for src, dst in {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n", "ü": "u"}.items():
        texto = texto.replace(src, dst)
    return texto


def detectar_intencion(
    pregunta: str,
    palabras_saludo: list[str],
    frases_despedida: list[str],
    frases_agradecimiento: list[str],
) -> dict:
    normalizada = _normalizar(pregunta)

    for frase in frases_agradecimiento:
        if _normalizar(frase) in normalizada:
            return {"intencion": "agradecimiento", "confianza": 0.95}

    for frase in frases_despedida:
        if _normalizar(frase) in normalizada:
            return {"intencion": "despedida", "confianza": 0.95}

    for palabra in palabras_saludo:
        palabra_norm = _normalizar(palabra)
        if normalizada == palabra_norm or normalizada.startswith(palabra_norm + " ") or normalizada.startswith(palabra_norm + ","):
            resto = normalizada.replace(palabra_norm, "", 1).strip(" ,.:;!?¡¿")
            if len(resto) > 5:
                return {"intencion": "consulta_academica", "confianza": 0.8}
            return {"intencion": "saludo", "confianza": 0.95}

    # Por defecto: consulta académica
    return {"intencion": "consulta_academica", "confianza": 0.7}
