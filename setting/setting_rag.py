import asyncio
from typing import List
import os
from src.rag import AdvancedRAG
from src.handle_dir_and_files.load_documents import LoadDocuments
from src.handle_dir_and_files.process_documents import ProcessDocuments
from utils.mylogger import Logger
from config import Config_LLM

LLM = None
# Инициализация логгера для отслеживания работы приложения
logger = Logger('Start RAG', 'logs/rag.log')

def create_LLM(config):
    """
    Создает и настраивает экземпляр класса AdvancedRAG.

    Процесс создания:
    1. Получение конфигурации из config
    2. Инициализация AdvancedRAG с параметрами из конфигурации
    3. Настройка компонентов:
        - промптов
    4. Возврат готового экземпляра

    Args:
        config: Объект конфигурации с параметрами для LLM
            - model_name: название модели
            - api_key: ключ API
            - base_url: базовый URL
            - temperature: температура генерации
    """
    try:
        logger.info("Начало создания экземпляра AdvancedRAG")
        global LLM
        
        # Проверка наличия необходимых параметров
        required_params = ['model_name', 'api_key', 'base_url', 'temperature']
        for param in required_params:
            if not hasattr(config, param):
                raise ValueError(f"Отсутствует обязательный параметр конфигурации: {param}")
        
        logger.debug(f"Инициализация AdvancedRAG с параметрами: model_name={config.model_name}, base_url={config.base_url}, temperature={config.temperature}")
        
        LLM = AdvancedRAG(
            config.model_name,
            config.api_key,
            config.base_url,
            config.temperature
        )
        logger.info("Экземпляр AdvancedRAG успешно создан")
        # Настройка компонентов
        
        logger.debug("Настройка промптов")
        LLM.promts.setup_prompts()
        logger.info("Промпты успешно настроены")
        
        logger.info("Настройка LLM успешно завершена")
        return LLM
    except Exception as e:
        error_msg = f"Ошибка при создании экземпляра AdvancedRAG: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

async def setting_up_LLM(documents: List[str]):
    """
    Асинхронно настраивает LLM для работы с документами.

    Процесс настройки:
    1. Асинхронная загрузка документов из указанных путей
    2. Асинхронная обработка документов (разбивка на чанки)
    3. Создание векторного хранилища
    4. Настройка компонентов:
        - ретриверов

    Args:
        documents: Список путей к документам для обработки
            Поддерживаемые форматы: PDF, TXT, DOCX
    """
    try:
        global LLM
        if not documents:
            raise ValueError("Список документов пуст")
        
        logger.info(f"Начало настройки LLM с документами: {documents}")
        
        # Проверка существования файлов
        for doc in documents:
            if not os.path.exists(doc):
                raise FileNotFoundError(f"Документ не найден: {doc}")
        
        # Асинхронная загрузка документов
        logger.debug("Начало загрузки документов")
        loaded_documents = await LoadDocuments(documents).load_documents_async()
        logger.info(f"Успешно загружено {len(loaded_documents)} документов")
        
        # Асинхронная обработка документов
        logger.debug("Начало обработки документов")
        processed_documents = await ProcessDocuments(loaded_documents).process_documents_async()
        logger.info(f"Документы успешно обработаны, получено {len(processed_documents)} чанков")
        
        # Создание векторного хранилища
        logger.debug("Создание векторного хранилища")
        LLM.vectorstore.create_vector_store(processed_documents)
        logger.info("Векторное хранилище успешно создано")
        # Настройка ретриверов
        logger.debug("Настройка ретриверов")
        LLM.retriever.setup_retrievers()
        logger.info("Ретриверы успешно настроены")
        
    except Exception as e:
        error_msg = f"Ошибка при настройке LLM: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

async def process_question(question: str) -> str:
    """
    Асинхронно обрабатывает вопрос пользователя.
    """
    global LLM
    try:
        if not question or not question.strip():
            raise ValueError("Вопрос не может быть пустым")
            
        logger.info(f"Обработка вопроса: {question[:100]}...")
        
        if not LLM:
            LLM = create_LLM(Config_LLM)
        response = await LLM.query_async(question)
        logger.info("Вопрос успешно обработан")
        return response
    except Exception as e:
        error_msg = f"Ошибка при обработке вопроса: {str(e)}"
        logger.error(error_msg)
        return f"Произошла ошибка: {str(e)}"

async def async_docs_loader(docs_dir: str):
    """
    Асинхронная основная функция приложения.
    
    Процесс работы:
    1. Создание экземпляра AdvancedRAG
    2. Проверка наличия директории с документами
    3. Настройка путей к документам
    4. Инициализация LLM с документами
    """
    try:
        logger.info(f"Начало загрузки документов из директории: {docs_dir}")
        
        if not os.path.exists(docs_dir):
            raise FileNotFoundError(f"Директория не найдена: {docs_dir}")
            
        global LLM
        # Создание и настройка LLM
        logger.debug("Создание экземпляра LLM")
        LLM = create_LLM(Config_LLM)
        
        # Асинхронная настройка LLM для работы с документами
        logger.debug("Настройка LLM для работы с документами")
        await setting_up_LLM([docs_dir])
        
        logger.info("Загрузка документов успешно завершена")
    except Exception as e:
        error_msg = f"Ошибка при загрузке документов: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

def docs_loader(docs_dir: str):
    """
    Загружает документы и обрабатывает вопрос пользователя.
    """
    try:
        logger.info("Запуск синхронной загрузки документов")
        asyncio.run(async_docs_loader(docs_dir))
        logger.info("Синхронная загрузка документов завершена")
    except Exception as e:
        error_msg = f"Ошибка при синхронной загрузке документов: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

def query_llm(question: str) -> str:
    """
    Обрабатывает вопрос пользователя.
    """
    try:
        logger.info("Запуск обработки вопроса через синхронный интерфейс")
        response = asyncio.run(process_question(question))
        logger.info("Синхронная обработка вопроса завершена")
        return response
    except Exception as e:
        error_msg = f"Ошибка при синхронной обработке вопроса: {str(e)}"
        logger.error(error_msg)
        return f"Произошла ошибка: {str(e)}"