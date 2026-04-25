from app.core.prompts import SYSTEM_PROMPT_EDITABLE, USER_TEMPLATE

DEFAULTS_NLU: dict = {
    "palabras_saludo": [
        "hola", "buenos días", "buenas tardes", "buenas noches",
        "buen día", "buenas", "hey", "saludos", "hi", "hello",
    ],
    "frases_despedida": [
        "adiós", "adios", "hasta luego", "chao", "chau",
        "nos vemos", "hasta pronto", "bye",
    ],
    "frases_agradecimiento": [
        "gracias", "muchas gracias", "te agradezco", "gracias por tu ayuda",
        "muy amable", "perfecto gracias",
    ],
    "palabras_lista_larga": [
        "todas las materias", "todos los niveles", "lista completa",
        "enumera todas", "todos los semestres",
        "qué materias hay en", "materias del nivel", "cuáles son todas",
        "prerrequisitos transitivos", "debería haber aprobado antes",
        "sin ningún prerrequisito", "no tienen prerrequisito",
        "qué necesito para graduarme", "requisitos para graduarme",
        "qué requisitos", "cuáles son los requisitos",
    ],
    "frases_rechazo": [
        "no encontré información",
        "no está disponible",
        "lo siento",
        "no tengo informacion",
        "no hay informacion",
        "no se encuentra",
        "no consta",
        "no puedo",
        "no dispongo",
        "no cuento con esa información",
    ],
    "mensaje_saludo": "¡Hola! Soy el Asistente Académico de la EPN. ¿En qué puedo ayudarte hoy?",
    "mensaje_despedida": "¡Hasta luego! Si tienes más consultas académicas, aquí estaré.",
    "mensaje_agradecimiento": "Con gusto. ¿Hay algo más en lo que pueda ayudarte?",
    "mensaje_fuera_de_tema": "Solo puedo ayudarte con consultas académicas de la EPN. ¿Tienes alguna pregunta sobre materias, créditos o requisitos de graduación?",
    "mensaje_sin_resultados": "No encontré información sobre eso en los documentos académicos disponibles. Intenta reformular tu pregunta.",
}

DEFAULTS_RAG: dict = {
    "umbral_relevancia_local": 0.05,
    "rag_k_local": 10,
    "prompt_principal": SYSTEM_PROMPT_EDITABLE,
    "system_prompt": SYSTEM_PROMPT_EDITABLE,
}

FRASES_NORMALIZACION_VACIA: list[str] = [
    "lo siento",
    "no existe en mi base",
    "no tengo esa información",
    "no poseo información",
    "no cuento con esa",
    "esa información no existe",
    "no hay información",
    "no se encontró información",
    "no dispongo de",
]

MARCADORES_IDIOMA_INCORRECTO: list[str] = [
    "você", "voce", "matérias", "seguintes", "tem como",
    "são ", "está ", "precisa ", "todas as ", "pode ",
]
