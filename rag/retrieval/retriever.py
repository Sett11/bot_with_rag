from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from rag.embedded.custom_embeddings import CustomEmbeddings
from utils.mylogger import Logger
from config import RAG_CONFIG
import asyncio
from typing import List, Dict, Any
from langchain.schema import Document

logger = Logger('Retriever', 'logs/rag.log')

class Retriever:
    """
    Класс для настройки и использования системы ретриверов для поиска документов.
    
    Этот класс предоставляет функциональность для:
    - Настройки базового ретривера на основе векторного хранилища
    - Настройки фильтра по эмбеддингам для улучшения релевантности
    - Создания компресионного ретривера для финального поиска
    
    Attributes:
        llm: Объект класса LLM, содержащий векторное хранилище
        vectorstore: Векторное хранилище для поиска документов
        embedding_model: Модель для создания эмбеддингов
    """
    def __init__(self, llm) -> None:
        """
        Инициализация класса Retriever.
        
        Args:
            llm: Объект класса LLM, содержащий векторное хранилище и другие компоненты
        """
        logger.info("Инициализация класса Retriever")
        self.llm = llm
        self.vectorstore = llm.vectorstore
        self.embedding_model = CustomEmbeddings(llm.sentence_transformer)
        logger.debug("Компоненты Retriever успешно инициализированы")

    async def get_relevant_documents_async(self, query: str, max_retries: int = 3) -> List[Document]:
        """
        Асинхронно получает релевантные документы для запроса.
        
        Args:
            query (str): Текстовый запрос пользователя
            max_retries (int): Максимальное количество попыток при ошибке
            
        Returns:
            List[Document]: Список релевантных документов
        """
        for attempt in range(max_retries):
            try:
                logger.debug(f"Попытка {attempt + 1}/{max_retries} поиска документов для запроса: {query[:100]}...")
                
                # Получаем эмбеддинг запроса
                query_embedding = self.embedding_model.embed_query(query)
                
                # Используем PostgreSQL для поиска похожих документов
                if self.vectorstore and hasattr(self.vectorstore, 'postgres_db'):
                    results = await self.vectorstore.postgres_db.find_similar_async(
                        query_embedding,
                        limit=min(RAG_CONFIG["search_kwargs"]["k"], 10)
                    )
                    
                    # Преобразуем результаты в формат Document
                    documents = []
                    for content, metadata, similarity in results:
                        doc = Document(
                            page_content=content,
                            metadata=metadata
                        )
                        documents.append(doc)
                else:
                    raise ValueError("Векторное хранилище не инициализировано или не содержит PostgreSQL")
                
                logger.info(f"Найдено {len(documents)} релевантных документов")
                return documents
                
            except Exception as e:
                logger.error(f"Ошибка при поиске документов (попытка {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    # Добавляем задержку перед следующей попыткой
                    await asyncio.sleep(1)
                else:
                    raise

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Синхронная обертка для получения релевантных документов
        """
        return asyncio.run(self.get_relevant_documents_async(query))

    async def setup_retrievers_async(self) -> None:
        """
        Настройка системы ретриверов для поиска документов.
        
        Процесс настройки:
        1. Проверка инициализации векторного хранилища
        2. Настройка базового ретривера с порогом схожести
        3. Настройка фильтра по эмбеддингам
        4. Создание компресионного ретривера
        
        Raises:
            ValueError: Если векторное хранилище не инициализировано
            Exception: При ошибках настройки ретриверов
        """
        logger.info("Начало настройки системы ретриверов")
        
        if not self.vectorstore:
            error_msg = "Векторное хранилище не инициализировано"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        try:
            # Настраиваем базовый ретривер
            try:
                logger.debug("Настройка базового ретривера")
                # Используем векторное хранилище из llm, а не из vectorstore
                self.llm.base_retriever = self.llm.vectorstore.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={
                        **RAG_CONFIG["search_kwargs"],
                        "k": min(RAG_CONFIG["search_kwargs"]["k"], 10)  # Ограничиваем количество результатов
                    }
                )
                logger.info("Базовый ретривер успешно настроен")
            except Exception as e:
                logger.error(f"Ошибка при настройке базового ретривера: {str(e)}")
                raise
                
            # Настраиваем фильтр по эмбеддингам для LLM
            try:
                logger.debug("Настройка фильтра по эмбеддингам")
                embeddings_filter = await asyncio.to_thread(
                    EmbeddingsFilter,
                    embeddings=self.embedding_model,
                    similarity_threshold=RAG_CONFIG["similarity_threshold"]
                )
                logger.info("Фильтр по эмбеддингам успешно настроен")
            except Exception as e:
                logger.warning(f"Не удалось настроить фильтр по эмбеддингам: {str(e)}")
                logger.info("Продолжаем работу без фильтра по эмбеддингам")
                embeddings_filter = None
                
            # Создаем компресионный ретривер для LLM
            try:
                logger.debug("Создание компресионного ретривера")
                if embeddings_filter:
                    self.llm.retriever = await asyncio.to_thread(
                        ContextualCompressionRetriever,
                        base_compressor=embeddings_filter,
                        base_retriever=self.llm.base_retriever
                    )
                    logger.info("Компресионный ретривер успешно настроен")
                else:
                    self.llm.retriever = self.llm.base_retriever
                    logger.info("Используется базовый ретривер без фильтрации")
            except Exception as e:
                logger.warning(f"Не удалось создать компресионный ретривер: {str(e)}")
                logger.info("Используем базовый ретривер")
                self.llm.retriever = self.llm.base_retriever
                
            logger.info("Настройка системы ретриверов успешно завершена")
            
        except Exception as e:
            logger.error(f"Ошибка при настройке системы ретриверов: {str(e)}")
            raise

    def setup_retrievers(self) -> None:
        """
        Синхронная обертка для настройки системы ретриверов
        """
        asyncio.run(self.setup_retrievers_async())