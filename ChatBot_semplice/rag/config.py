"""Parametri del sistema RAG, raccolti in un solo punto."""

# --- Embedding e chunking (come nel notebook di partenza) ---
EMBEDDING_MODEL = "BAAI/bge-m3"
MAX_TOKEN = 450  # lunghezza massima di un chunk, in token di BGE-M3
OVERLAP = 75  # token condivisi tra due chunk consecutivi della stessa pagina

# --- Retrieval ibrido ---
TOP_K = 4  # passaggi passati all'LLM
RRF_K = 60  # costante della Reciprocal Rank Fusion
PESO_SEMANTICO = 0.8
PESO_LESSICALE = 0.2

NON_SO = "Non ho informazioni sufficienti nei documenti forniti."

# --- LLM via API gratuita (Groq) ---
GROQ_ENV = "GROQ_API_KEY"
GROQ_URL_CHIAVE = "https://console.groq.com/keys"
# Il primo modello disponibile di questo elenco diventa quello predefinito.
GROQ_MODELLI_PREFERITI = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b"]
