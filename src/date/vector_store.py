from langchain_core.documents import Document
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import asyncio

from utils.mylogger import Logger
from src.embedded.custom_embeddings import CustomEmbeddings
from config import RAG_CONFIG

# Инициализация логгера для отслеживания работы векторного хранилища
logger = Logger('VectorStore', 'logs/rag.log')

class VectorStore:
    """
    Класс для создания и управления векторным хранилищем на основе FAISS.
    
    Основные функции:
    1. Разбиение документов на чанки с помощью RecursiveCharacterTextSplitter
    2. Создание векторных представлений документов
    3. Индексация документов в FAISS для быстрого поиска
    
    Особенности:
    - Использует кастомную модель эмбеддингов
    - Поддерживает метаданные документов
    - Имеет механизм fallback при ошибках создания хранилища
    - Оптимизирован для работы с русскоязычными текстами
    """
    def __init__(self, llm) -> None:
        """
        Инициализация векторного хранилища.

        Args:
            llm: Экземпляр класса AdvancedRAG, содержащий модель для эмбеддингов
        """
        self.llm = llm
        # Инициализация сплиттера для разбиения документов на чанки
        # Параметры сплиттера берутся из конфигурации RAG_CONFIG
        self.text_splitter = RecursiveCharacterTextSplitter(**RAG_CONFIG["text_splitter"])
        # Создание модели для генерации эмбеддингов
        self.embedding_model = CustomEmbeddings(llm.sentence_transformer)

    async def create_vector_store_async(self, documents: List[Document]) -> None:
        """
        Асинхронное создание векторного хранилища из документов.

        Процесс создания:
        1. Разбиение документов на чанки с помощью text_splitter
        2. Попытка создания хранилища стандартным методом FAISS
        3. При неудаче - создание хранилища вручную:
           - Извлечение текстов и метаданных
           - Генерация эмбеддингов
           - Создание индекса FAISS
           - Инициализация векторного хранилища

        Args:
            documents (List[Document]): Список документов для индексации
                Каждый документ должен содержать:
                - page_content: текст документа
                - metadata: метаданные (источник, страница и т.д.)

        Raises:
            ValueError: Если список документов пуст
            Exception: При ошибках создания векторного хранилища
        """
        if not documents:
            raise ValueError("Список документов не может быть пустым")
        try:
            # Разбиваем документы на чанки для оптимизации поиска
            try:
                chunks = self.text_splitter.split_documents(documents)
                logger.info(f"Документы разбиты на {len(chunks)} чанков")
            except Exception as e:
                logger.error(f"Ошибка при разбиении документов на чанки: {str(e)}")
                raise
                
            # Создаем векторное хранилище
            try:
                # Пробуем стандартный метод создания FAISS
                self.llm.vectorstore = await asyncio.to_thread(
                    FAISS.from_documents,
                    documents=chunks,
                    embedding=self.embedding_model
                )
                logger.info("Векторное хранилище успешно создано стандартным методом")
            except Exception as e:
                logger.warning(f"Не удалось создать векторное хранилище стандартным методом: {str(e)}")
                logger.info("Пробуем создать векторное хранилище вручную")
                try:
                    # Извлекаем тексты и метаданные из чанков
                    texts = [doc.page_content for doc in chunks]
                    metadatas = [doc.metadata for doc in chunks]
                    
                    # Получаем векторные представления для всех текстов
                    embeddings = await asyncio.to_thread(
                        self.embedding_model.embed_documents,
                        texts
                    )
                    
                    # Создаем индекс FAISS с размерностью векторов
                    dimension = len(embeddings[0])
                    index = FAISS.IndexFlatL2(dimension)
                    index.add(np.array(embeddings).astype('float32'))
                    
                    # Создаем векторное хранилище с кастомной моделью эмбеддингов
                    self.llm.vectorstore = FAISS(
                        self.embedding_model.embed_query,
                        index,
                        texts,
                        metadatas
                    )
                    logger.info("Векторное хранилище успешно создано вручную")
                except Exception as e:
                    logger.error(f"Ошибка при создании векторного хранилища вручную: {str(e)}")
                    raise
        except Exception as e:
            logger.error(f"Критическая ошибка при создании векторного хранилища: {str(e)}")
            raise

    def create_vector_store(self, documents: List[Document]) -> None:
        """
        Синхронное создание векторного хранилища из документов.
        """
        asyncio.run(self.create_vector_store_async(documents))