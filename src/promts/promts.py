from typing import List
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder
from utils.mylogger import Logger
import asyncio

logger = Logger('Promts', 'logs/rag.log')

class Promts:
    """
    Класс для управления промптами в RAG системе.
    
    Основные функции:
    1. Настройка промптов для LLM
    2. Реранжирование документов с помощью cross-encoder
    
    Промпты:
    - main_prompt: используется для генерации ответов на вопросы
    - verification_prompt: используется для проверки качества ответов
    
    После инициализации и вызова setup_prompts() объект класса AdvancedRAG
    получает два основных промпта:
    1. main_prompt - определяет структуру и формат ответа LLM
    2. verification_prompt - оценивает качество сгенерированного ответа
    """
    def __init__(self, llm) -> None:
        """
        Инициализация менеджера промптов.

        Args:
            llm: Экземпляр класса AdvancedRAG, которому будут присвоены промпты
        """
        self.llm = llm
        # Инициализируем cross-encoder для реранжирования документов
        # Эта модель помогает определить наиболее релевантные документы
        # путем оценки их соответствия вопросу пользователя
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        logger.info("Cross-encoder успешно инициализирован")

    async def setup_prompts_async(self) -> None:
        """
        Настройка промптов для LLM.
        
        Создает два основных промпта:
        1. main_prompt - для генерации ответов
           - Анализирует контекст и вопрос
           - Формирует структурированный ответ
           - Включает источники информации
        2. verification_prompt - для проверки качества ответов
           - Проверяет соответствие контексту
           - Оценивает полноту ответа
           - Предлагает улучшения при необходимости
        
        Оба промпта используют унифицированные теги <answer> и </answer>
        для выделения точного ответа.
        """
        try:
            # Основной промпт для генерации ответов
            self.llm.main_prompt = await asyncio.to_thread(
                ChatPromptTemplate.from_messages,
                [
                    ("system", """Ты - экспертный русскоязычный AI-ассистент. Анализируй вопрос и контекст, следуя шагам:
1. Определи ключевые аспекты вопроса
2. Выдели релевантные части контекста
3. Сформулируй точный ответ и обязательно помести его между тегами <answer> и </answer>
4. Если нужно, укажи источник информации

Контекст:
{context}

Вопрос: {question}"""),
                    ("human", "{question}")
                ]
            )
            
            # Промпт для верификации качества ответа
            self.llm.verification_prompt = await asyncio.to_thread(
                ChatPromptTemplate.from_messages,
                [
                    ("system", """Проверь соответствие ответа контексту и вопросу. Ответ должен:
1. Быть основан только на контексте
2. Не содержать вымышленных фактов
3. Быть точным и конкретным
4. Полностью отвечать на заданный вопрос

Контекст:
{context}

Вопрос пользователя:
{question}

Ответ для проверки:
{response}"""),
                    ("human", "Требуется ли корректировка ответа? Если да, предложи улучшенную версию ответа и обязательно помести эту версию между тегами <perfect_answer> и </perfect_ answer>.")
                ]
            )
            logger.info("Промпты успешно настроены")
        except Exception as e:
            logger.error(f"Ошибка настройки промптов: {str(e)}")
            raise

    def setup_prompts(self) -> None:
        """
        Синхронная обертка для настройки промптов
        """
        asyncio.run(self.setup_prompts_async())

    async def rerank_documents_async(self, question: str, documents: List[Document]) -> List[Document]:
        """
        Реранжирование документов с использованием cross-encoder.

        Процесс реранжирования:
        1. Создание пар вопрос-документ
        2. Получение оценок релевантности
        3. Сортировка документов по оценкам

        Args:
            question (str): Вопрос пользователя
            documents (List[Document]): Список документов для реранжирования

        Returns:
            List[Document]: Отсортированный список документов

        Raises:
            ValueError: Если список документов пуст
            Exception: При ошибках реранжирования
        """
        if not documents:
            raise ValueError("Список документов не может быть пустым")
        try:
            # Создаем пары вопрос-документ для оценки
            pairs = [(question, doc.page_content) for doc in documents]
            
            # Получаем оценки релевантности
            scores = await asyncio.to_thread(self.cross_encoder.predict, pairs)
            
            # Сортируем документы по оценкам
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Возвращаем отсортированные документы
            return [doc for doc, _ in scored_docs]
            
        except Exception as e:
            logger.error(f"Ошибка при реранжировании документов: {str(e)}")
            raise

    def rerank_documents(self, question: str, documents: List[Document]) -> List[Document]:
        """
        Синхронная обертка для реранжирования документов
        """
        return asyncio.run(self.rerank_documents_async(question, documents))