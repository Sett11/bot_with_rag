from typing import List
import os
import asyncio
import aiofiles
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

from utils.mylogger import Logger
from src.handle_dir_and_files.check_dir import CheckDirExists
from src.handle_dir_and_files.check_file import CheckFile

logger = Logger('LoadDocuments', 'logs/rag.log')

class LoadDocuments:
    """
    Класс для загрузки документов из различных форматов файлов.
    
    Этот класс предоставляет функциональность для:
    - Загрузки документов из PDF, TXT и DOCX файлов
    - Проверки существования и доступа к файлам и директориям
    - Фильтрации неподдерживаемых форматов
    - Обработки ошибок при загрузке документов
    
    Attributes:
        file_patterns (List[str]): Список паттернов для поиска файлов
        check_dir (CheckDirExists): Объект для проверки директорий
        check_file (CheckFile): Объект для проверки файлов
    """
    def __init__(self, file_patterns: List[str]) -> None:
        """
        Инициализация класса LoadDocuments.
        
        Args:
            file_patterns (List[str]): Список паттернов для поиска файлов
        """
        logger.info("Инициализация класса LoadDocuments")
        self.file_patterns = file_patterns
        logger.debug(f"Получены паттерны для поиска файлов: {file_patterns}")
        self.check_dir = CheckDirExists()
        self.check_file = CheckFile()

    def _get_supported_formats(self) -> List[str]:
        """
        Возвращает список поддерживаемых форматов файлов.

        Returns:
            List[str]: Список расширений файлов
        """
        logger.debug("Получение списка поддерживаемых форматов")
        return ['.pdf', '.txt', '.docx']

    def _is_supported_format(self, file_path: str) -> bool:
        """
        Проверяет, поддерживается ли формат файла.

        Args:
            file_path (str): Путь к файлу

        Returns:
            bool: True, если формат поддерживается, иначе False
        """
        # Если путь указывает на директорию, считаем его поддерживаемым
        if os.path.isdir(file_path):
            return True
            
        supported_formats = self._get_supported_formats()
        is_supported = any(file_path.lower().endswith(fmt) for fmt in supported_formats)
        logger.debug(f"Проверка формата файла {file_path}: {'поддерживается' if is_supported else 'не поддерживается'}")
        return is_supported

    async def _load_single_document(self, file_path: str) -> List[Document]:
        """
        Асинхронно загружает один документ.
        """
        try:
            if file_path.lower().endswith('.pdf'):
                logger.debug(f"Загрузка PDF файла: {file_path}")
                loader = PyPDFLoader(file_path)
            elif file_path.lower().endswith('.txt'):
                logger.debug(f"Загрузка TXT файла: {file_path}")
                loader = TextLoader(file_path)
            elif file_path.lower().endswith('.docx'):
                logger.debug(f"Загрузка DOCX файла: {file_path}")
                loader = Docx2txtLoader(file_path)
            else:
                logger.warning(f"Неподдерживаемый формат файла: {file_path}")
                return []

            # Загружаем документ
            docs = loader.load()
            if docs:
                logger.info(f"Успешно загружен файл: {file_path}")
                return docs
            else:
                logger.warning(f"Файл не содержит текста: {file_path}")
                return []

        except Exception as e:
            logger.error(f"Ошибка при загрузке файла {file_path}: {str(e)}")
            return []

    async def load_documents_async(self) -> List[Document]:
        """
        Асинхронная загрузка документов из указанных файловых паттернов.

        Процесс загрузки:
        1. Проверка существования и доступа к директории
        2. Рекурсивный поиск файлов с поддерживаемыми расширениями
        3. Проверка прав доступа к каждому файлу
        4. Загрузка документов в зависимости от формата
        5. Обработка ошибок при загрузке

        Returns:
            List[Document]: Список загруженных документов

        Raises:
            ValueError: Если список паттернов пуст
            FileNotFoundError: Если не найдено ни одного файла
            Exception: При ошибках загрузки документов
        """
        logger.info("Начало асинхронной загрузки документов")
        
        if not self.file_patterns:
            error_msg = "Список паттернов файлов не может быть пустым"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        documents = []
        loaded_files = 0
        skipped_files = 0
        
        try:
            # Получаем базовую директорию из первого паттерна
            base_dir = self.file_patterns[0]
            logger.debug(f"Базовая директория для поиска: {base_dir}")
            
            # Проверяем существование и доступ к директории
            if not self.check_dir.check_dir_exists(base_dir):
                logger.warning(f"Пропускаем директорию {base_dir}: директория не существует")
                return documents
                
            if not self.check_dir.check_dir_access(base_dir):
                logger.warning(f"Пропускаем директорию {base_dir}: нет доступа к директории")
                return documents
                
            # Получаем список поддерживаемых расширений
            supported_formats = self._get_supported_formats()
            
            # Собираем все файлы для загрузки
            files_to_load = []
            for root, _, files in os.walk(base_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_ext = os.path.splitext(file_path)[1].lower()
                    
                    if file_ext in supported_formats:
                        if self.check_file.check_file_access(file_path):
                            files_to_load.append(file_path)
                        else:
                            logger.warning(f"Пропускаем файл без прав доступа: {file_path}")
                            skipped_files += 1

            if not files_to_load:
                error_msg = "Не найдено файлов для загрузки"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            # Асинхронно загружаем все документы
            tasks = [self._load_single_document(file_path) for file_path in files_to_load]
            results = await asyncio.gather(*tasks)
            
            # Объединяем результаты
            for docs in results:
                if docs:
                    documents.extend(docs)
                    loaded_files += 1
                else:
                    skipped_files += 1

            if not documents:
                error_msg = "Не удалось загрузить ни одного документа"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            logger.info(f"Асинхронная загрузка завершена. Загружено: {loaded_files}, пропущено: {skipped_files}")
            return documents

        except Exception as e:
            logger.error(f"Критическая ошибка при асинхронной загрузке документов: {str(e)}")
            raise

    def load_documents(self) -> List[Document]:
        """
        Синхронная загрузка документов (для обратной совместимости).
        """
        return asyncio.run(self.load_documents_async())