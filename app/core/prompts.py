SYSTEM_PROMPT_EDITABLE = """Eres el Asistente Académico de la EPN. Responde SIEMPRE en español.

REGLAS (sin excepción):
1. Usa SOLO la información del CONTEXTO. Nunca inventes datos.
2. Sin datos: responde exactamente "No encontré información sobre [tema específico] en los documentos académicos disponibles." No uses "Lo siento" ni variantes.
3. Dato numérico: incluye su contexto explicativo si el CONTEXTO lo tiene.
4. Afirmación incorrecta del usuario: empieza con "No es correcto." y da el dato real del CONTEXTO.
5. Entidad inexistente: si no existe pero hay entidades similares en el CONTEXTO, menciónalas.
6. Nunca escribas "Según el contexto", "El documento indica" ni referencias explícitas al contexto."""

PROMPT_SUFFIX_FIJO = "\n\nCONTEXTO:\n{contexto}\n\nPregunta: {pregunta}\nRespuesta:"

PROMPT_SUFFIX_FIJO_CHARS: int = len(PROMPT_SUFFIX_FIJO)

MAX_TOKENS_MODELO:            int = 2048
TOKENS_RESERVADOS_RESPUESTA:  int = 512
TOKENS_RESERVADOS_CONTEXTO:   int = 630  

MAX_TOKENS_SYSTEM_PROMPT:     int = (
    MAX_TOKENS_MODELO - TOKENS_RESERVADOS_RESPUESTA - TOKENS_RESERVADOS_CONTEXTO
)  

MAX_CHARS_SYSTEM_PROMPT:      int = int(MAX_TOKENS_SYSTEM_PROMPT * 3.5)  # 3171
MIN_CHARS_SYSTEM_PROMPT:      int = 50

MAX_CHARS_SYSTEM_PROMPT_EDITABLE: int = MAX_CHARS_SYSTEM_PROMPT - PROMPT_SUFFIX_FIJO_CHARS

USER_TEMPLATE = """CONTEXTO:
{contexto}

Pregunta: {pregunta}"""

PROMPT_ERROR_FALLBACK = """En este momento no puedo procesar tu consulta. Por favor intenta de nuevo en unos segundos."""