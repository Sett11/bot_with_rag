from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import execute_values
import asyncio
import json
import ast
from utils.mylogger import Logger
from config import EMBEDDING_DIMENSION, conn_params

logger = Logger('PostgresDB', 'logs/rag.log')

class PostgresDB:
    """
    Класс для работы с PostgreSQL и pgvector.
    
    Основные функции:
    1. Создание и управление таблицами для хранения документов и их эмбеддингов
    2. Сохранение и получение эмбеддингов
    3. Поиск похожих документов с использованием pgvector
    
    Особенности:
    - Использует расширение pgvector для векторных операций
    - Поддерживает метаданные документов
    - Асинхронные операции через asyncio
    """
    
    def __init__(self):
        """
        Инициализация подключения к PostgreSQL.
        """
        self.conn_params = conn_params
        # Увеличиваем размер буфера для сокета
        self.conn_params['options'] = '-c statement_timeout=0 -c tcp_keepalives_idle=60 -c tcp_keepalives_interval=10 -c tcp_keepalives_count=6'
        self._init_db()
        
    def _init_db(self):
        """
        Инициализация базы данных и создание необходимых таблиц.
        """
        try:
            with psycopg2.connect(**self.conn_params) as conn:
                with conn.cursor() as cur:
                    # Создаем расширение pgvector, если оно еще не создано
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    
                    # Создаем таблицу для документов и их эмбеддингов
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS documents (
                            id SERIAL PRIMARY KEY,
                            content TEXT NOT NULL,
                            metadata JSONB,
                            embedding vector({EMBEDDING_DIMENSION}),
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    
                    # Создаем индекс для векторного поиска
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                        ON documents 
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                    """)
                    
                    conn.commit()
                    logger.info("База данных успешно инициализирована")
        except Exception as e:
            logger.error(f"Ошибка при инициализации базы данных: {str(e)}")
            raise

    async def save_embeddings_async(self, texts: List[str], embeddings: List[List[float]], 
                                  metadatas: List[Dict[str, Any]] = None) -> None:
        """
        Асинхронное сохранение текстов и их эмбеддингов в базу данных.
        
        Args:
            texts: Список текстов документов
            embeddings: Список эмбеддингов
            metadatas: Список метаданных документов (опционально)
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
            
        try:
            # Разбиваем данные на пакеты по 1000 записей
            batch_size = 1000
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                
                with psycopg2.connect(**self.conn_params) as conn:
                    with conn.cursor() as cur:
                        # Подготавливаем данные для вставки, преобразуя метаданные в JSON
                        data = [(text, json.dumps(metadata), embedding) 
                               for text, metadata, embedding in zip(batch_texts, batch_metadatas, batch_embeddings)]
                        
                        # Вставляем данные
                        execute_values(
                            cur,
                            """
                            INSERT INTO documents (content, metadata, embedding)
                            VALUES %s
                            ON CONFLICT DO NOTHING
                            """,
                            data,
                            template="(%s, %s::jsonb, %s::vector)"
                        )
                        
                        conn.commit()
                        logger.info(f"Успешно сохранен пакет {i//batch_size + 1} ({len(batch_texts)} документов)")
        except Exception as e:
            logger.error(f"Ошибка при сохранении эмбеддингов: {str(e)}")
            raise

    def save_embeddings(self, texts: List[str], embeddings: List[List[float]], 
                       metadatas: List[Dict[str, Any]] = None) -> None:
        """
        Синхронная обертка для сохранения эмбеддингов
        """
        asyncio.run(self.save_embeddings_async(texts, embeddings, metadatas))

    async def get_total_documents(self) -> int:
        """
        Получение общего количества документов в базе данных.
        
        Returns:
            int: Общее количество документов
        """
        try:
            with psycopg2.connect(**self.conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM documents")
                    count = cur.fetchone()[0]
                    return count
        except Exception as e:
            logger.error(f"Ошибка при получении количества документов: {str(e)}")
            raise

    async def get_embeddings_async(self, limit: int = None, offset: int = 0) -> tuple:
        """
        Асинхронное получение текстов и их эмбеддингов из базы данных.
        
        Args:
            limit: Максимальное количество документов для получения
            offset: Смещение для пагинации
            
        Returns:
            tuple: (texts, embeddings, metadatas)
        """
        try:
            with psycopg2.connect(**self.conn_params) as conn:
                with conn.cursor() as cur:
                    # Используем более эффективный запрос с сортировкой по id
                    query = """
                        SELECT content, embedding, metadata
                        FROM documents
                        ORDER BY id
                    """
                    if limit:
                        query += f" LIMIT {limit} OFFSET {offset}"
                        
                    cur.execute(query)
                    results = cur.fetchall()
                    
                    texts = [row[0] for row in results]
                    # Преобразуем строковое представление эмбеддингов в список чисел
                    embeddings = [ast.literal_eval(str(row[1])) for row in results]
                    metadatas = [row[2] for row in results]
                    
                    logger.info(f"Успешно получено {len(texts)} документов")
                    return texts, embeddings, metadatas
        except Exception as e:
            logger.error(f"Ошибка при получении эмбеддингов: {str(e)}")
            raise

    def get_embeddings(self, limit: int = None) -> tuple:
        """
        Синхронная обертка для получения эмбеддингов
        """
        return asyncio.run(self.get_embeddings_async(limit))

    async def find_similar_async(self, query_embedding: List[float], limit: int = 5) -> List[tuple]:
        """
        Асинхронный поиск похожих документов по эмбеддингу запроса.
        
        Args:
            query_embedding: Эмбеддинг запроса
            limit: Максимальное количество результатов
            
        Returns:
            List[tuple]: Список кортежей (текст, метаданные, расстояние)
        """
        try:
            with psycopg2.connect(**self.conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT content, metadata, 1 - (embedding <=> %s::vector) as similarity
                        FROM documents
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """, (query_embedding, query_embedding, limit))
                    
                    results = cur.fetchall()
                    logger.info(f"Найдено {len(results)} похожих документов")
                    return results
        except Exception as e:
            logger.error(f"Ошибка при поиске похожих документов: {str(e)}")
            raise

    def find_similar(self, query_embedding: List[float], limit: int = 5) -> List[tuple]:
        """
        Синхронная обертка для поиска похожих документов
        """
        return asyncio.run(self.find_similar_async(query_embedding, limit))
