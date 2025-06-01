from setting.setting_rag import docs_loader
from utils.mylogger import Logger
import nest_asyncio
from config import DOCS_DIR

logger = Logger("Documents_loader", "logs/rag.log")

def load_documents():
    """
    Функция для загрузки документов в базу данных.
    Эта функция должна быть запущена отдельно перед запуском бота.
    """
    try:
        logger.info("Начало загрузки документов")
        nest_asyncio.apply()
        docs_loader(DOCS_DIR)
        logger.info("Документы успешно загружены")
    except Exception as e:
        logger.error(f"Ошибка при загрузке документов: {str(e)}")
        raise

if __name__ == "__main__":
    load_documents() 