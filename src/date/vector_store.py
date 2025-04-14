from langchain_core.documents import Document
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import asyncio

from utils.mylogger import Logger
from src.embedded.custom_embeddings import CustomEmbeddings
from src.date.postgres_db import PostgresDB
from config import RAG_CONFIG

# Инициализация логгера для отслеживания работы векторного хранилища
logger = Logger('VectorStore', 'logs/rag.log')

class VectorStore:
    """
    Класс для создания и управления векторным хранилищем на основе FAISS.
    
    Основные функции:
    1. Разбиение документов на чанки с помощью RecursiveCharacterTextSplitter
    2. Создание векторных представлений документов
    3. Сохранение эмбеддингов в PostgreSQL
    4. Индексация документов в FAISS для быстрого поиска
    
    Особенности:
    - Использует кастомную модель эмбеддингов
    - Поддерживает метаданные документов
    - Имеет механизм fallback при ошибках создания хранилища
    - Оптимизирован для работы с русскоязычными текстами
    - Использует PostgreSQL для хранения эмбеддингов
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
        # Инициализация PostgreSQL
        self.postgres_db = PostgresDB()

    async def create_vector_store_async(self, documents: List[Document]) -> None:
        """
        Асинхронное создание векторного хранилища из документов.

        Args:
            documents: Список документов для индексации
        """
        try:
            # Разбиваем документы на чанки для оптимизации поиска
            try:
                chunks = self.text_splitter.split_documents(documents)
                logger.info(f"Документы разбиты на {len(chunks)} чанков")
            except Exception as e:
                logger.error(f"Ошибка при разбиении документов на чанки: {str(e)}")
                raise
                
            # Извлекаем тексты и метаданные из чанков
            texts = [doc.page_content for doc in chunks]
            metadatas = [doc.metadata for doc in chunks]
            
            # Получаем векторные представления для всех текстов
            embeddings = await asyncio.to_thread(
                self.embedding_model.embed_documents,
                texts
            )
            
            # Сохраняем эмбеддинги в PostgreSQL
            try:
                await self.postgres_db.save_embeddings_async(texts, embeddings, metadatas)
                logger.info("Эмбеддинги успешно сохранены в PostgreSQL")
            except Exception as e:
                logger.error(f"Ошибка при сохранении эмбеддингов в PostgreSQL: {str(e)}")
                raise
            
            # Создаем FAISS индекс из сохраненных эмбеддингов
            try:
                # Получаем общее количество документов
                total_docs = await self.postgres_db.get_total_documents()
                logger.info(f"Всего документов в базе: {total_docs}")
                
                # Определяем размер пакета и количество пакетов
                batch_size = min(100, total_docs)  # Ограничиваем размер пакета
                num_batches = (total_docs + batch_size - 1) // batch_size
                
                all_texts = []
                all_embeddings = []
                all_metadatas = []
                
                # Загружаем данные пакетами
                for i in range(num_batches):
                    offset = i * batch_size
                    batch_texts, batch_embeddings, batch_metadatas = await self.postgres_db.get_embeddings_async(
                        limit=batch_size,
                        offset=offset
                    )
                    
                    if batch_texts:  # Проверяем, что пакет не пустой
                        all_texts.extend(batch_texts)
                        all_embeddings.extend(batch_embeddings)
                        all_metadatas.extend(batch_metadatas)
                        logger.info(f"Загружен пакет {i + 1}/{num_batches} ({len(batch_texts)} документов)")
                    
                    # Добавляем небольшую задержку между пакетами
                    await asyncio.sleep(0.1)
                
                # Создаем векторное хранилище FAISS
                if all_texts:  # Проверяем, что есть данные для создания индекса
                    self.llm.vectorstore = FAISS.from_embeddings(
                        text_embeddings=list(zip(all_texts, all_embeddings)),
                        embedding=self.embedding_model,
                        metadatas=all_metadatas
                    )
                    logger.info("Векторное хранилище FAISS успешно создано")
                else:
                    raise ValueError("Нет данных для создания векторного хранилища")
                    
            except Exception as e:
                logger.error(f"Ошибка при создании FAISS индекса: {str(e)}")
                raise
                
        except Exception as e:
            logger.error(f"Критическая ошибка при создании векторного хранилища: {str(e)}")
            raise

    def create_vector_store(self, documents: List[Document]) -> None:
        """
        Синхронное создание векторного хранилища из документов.
        """
        asyncio.run(self.create_vector_store_async(documents))