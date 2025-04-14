import os
from dotenv import load_dotenv

# Загружаем переменные окружения из файла .env
load_dotenv()

class Config_LLM:
    model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", 'https://bothub.chat/api/v2/openai/v1')
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.5"))

# Путь к директории с документами
DOCS_DIR = os.getenv("DOCS_DIR", os.path.join("C:", "Users", "Lenovo", "Desktop", "ИФ", "писянина", "ФМ", "1 глава - математика в целом"))

# Размерность эмбеддингов модели LaBSE-ru-turbo
EMBEDDING_DIMENSION = 768

# Конфигурация RAG
RAG_CONFIG = {
    'similarity_threshold': float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.5")),
    'search_kwargs': {
        'k': int(os.getenv("RAG_SEARCH_K", "20")),
        'score_threshold': float(os.getenv("RAG_SCORE_THRESHOLD", "0.5"))
    },
    'text_splitter': {
        'chunk_size': int(os.getenv("RAG_CHUNK_SIZE", "384")),
        'chunk_overlap': int(os.getenv("RAG_CHUNK_OVERLAP", "128")),
        'length_function': len,
        'is_separator_regex': False,
        'separators': ["\n\n", "\n", ". ", " ", ""]
    },
    'max_context_length': int(os.getenv("RAG_MAX_CONTEXT_LENGTH", "16000"))
}

MAX_HISTORY = int(os.getenv("MAX_HISTORY", "50"))

BOT_TOKEN = os.getenv("BOT_TOKEN")
BOT_NAME = os.getenv("BOT_NAME")
BOT_DESCRIPTION = os.getenv("BOT_DESCRIPTION")

# Параметры подключения к PostgreSQL в Docker
conn_params = {
    'dbname': 'vector_db',
    'host': 'localhost',
    'port': '5432',
    'user': 'postgres',
    'password': 'vector'
}