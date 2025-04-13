from langchain.embeddings.base import Embeddings
from utils.mylogger import Logger
import asyncio

# Инициализация логгера для отслеживания работы с эмбеддингами
logger = Logger('CustomEmbeddings', 'logs/rag.log')

class CustomEmbeddings(Embeddings):
    """
    Пользовательский класс для создания эмбеддингов текста.
    
    Этот класс является оберткой для моделей sentence-transformers, 
    обеспечивающей совместимость с интерфейсом LangChain Embeddings.
    
    Особенности:
    - Использует предобученные модели sentence-transformers
    - Поддерживает нормализацию эмбеддингов
    - Оптимизирован для работы с русскоязычными текстами
    - Обеспечивает единый интерфейс для батч-обработки и одиночных запросов
    
    Attributes:
        model: Модель sentence-transformers для генерации эмбеддингов
            Должна поддерживать методы encode() и normalize_embeddings
        
    Methods:
        embed_documents: Создает эмбеддинги для списка документов
        embed_query: Создает эмбеддинг для одного запроса
        embed_documents_async: Асинхронная версия embed_documents
        embed_query_async: Асинхронная версия embed_query
    """
    def __init__(self, model):
        """
        Инициализация класса CustomEmbeddings.
        
        Args:
            model: Модель sentence-transformers для генерации эмбеддингов
                Должна быть экземпляром класса SentenceTransformer
                и поддерживать русскоязычные тексты
        """
        self.model = model
        logger.info(f"Инициализация класса CustomEmbeddings. Модель: {model}")
        
    async def embed_documents_async(self, texts):
        """
        Асинхронно создает эмбеддинги для списка текстовых документов.
        
        Процесс:
        1. Принимает список текстовых документов
        2. Преобразует их в векторные представления
        3. Нормализует векторы для улучшения качества сравнения
        
        Args:
            texts (List[str]): Список текстовых документов
                Каждый документ должен быть строкой
                Поддерживаются документы на русском языке
            
        Returns:
            List[List[float]]: Список векторов эмбеддингов для каждого документа
                Каждый вектор - список чисел с плавающей точкой
                Все векторы нормализованы (длина = 1)
        """
        try:
            logger.debug(f"Асинхронное создание эмбеддингов для {len(texts)} документов")
            # Нормализуем эмбеддинги для улучшения качества сравнения
            embeddings = await asyncio.to_thread(
                self.model.encode,
                texts,
                normalize_embeddings=True
            )
            logger.info(f"Успешно созданы эмбеддинги для {len(texts)} документов")
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддингов: {str(e)}")
            raise

    def embed_documents(self, texts):
        """
        Синхронная обертка для создания эмбеддингов документов
        """
        return asyncio.run(self.embed_documents_async(texts))
        
    async def embed_query_async(self, text):
        """
        Асинхронно создает эмбеддинг для одного текстового запроса.
        
        Процесс:
        1. Принимает один текстовый запрос
        2. Преобразует его в векторное представление
        3. Нормализует вектор для согласованности с embed_documents
        
        Args:
            text (str): Текстовый запрос
                Должен быть строкой
                Поддерживаются запросы на русском языке
            
        Returns:
            List[float]: Вектор эмбеддинга для запроса
                Список чисел с плавающей точкой
                Вектор нормализован (длина = 1)
        """
        try:
            logger.debug(f"Асинхронное создание эмбеддинга для запроса: {text[:100]}...")
            # Нормализуем эмбеддинг для согласованности с embed_documents
            embedding = await asyncio.to_thread(
                self.model.encode,
                text,
                normalize_embeddings=True
            )
            logger.info("Успешно создан эмбеддинг для запроса")
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддинга для запроса: {str(e)}")
            raise

    def embed_query(self, text):
        """
        Синхронная обертка для создания эмбеддинга запроса
        """
        return asyncio.run(self.embed_query_async(text)) 