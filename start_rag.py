from setting.setting_rag import docs_loader
from utils.mylogger import Logger
import nest_asyncio
from config import DOCS_DIR

logger = Logger("Documents_loader", "logs/rag.log")

logger.info("Начало загрузки документов")
nest_asyncio.apply()
docs_loader(DOCS_DIR)
logger.info("Документы успешно загружены")