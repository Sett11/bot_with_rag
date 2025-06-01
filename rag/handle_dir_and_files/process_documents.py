from langchain_core.documents import Document
from typing import List
import asyncio
from utils.mylogger import Logger

logger = Logger('ProcessDocuments', 'logs/rag.log')

class ProcessDocuments:
    """
    Класс для обработки и подготовки документов перед индексацией.
    
    Этот класс предоставляет функциональность для:
    - Валидации содержимого документов
    - Очистки текста от лишних пробелов и переносов строк
    - Фильтрации документов с недостаточным количеством текста
    - Сохранения метаданных документов
    
    Attributes:
        documents (List[Document]): Список документов для обработки
    """
    def __init__(self, documents: List[Document]) -> None:
        """
        Инициализация класса ProcessDocuments.
        
        Args:
            documents (List[Document]): Список документов для обработки
        """
        logger.info("Инициализация класса ProcessDocuments")
        self.documents = documents
        logger.debug(f"Получено документов для обработки: {len(documents)}")

    async def _process_single_document(self, doc: Document) -> Document:
        """
        Асинхронно обрабатывает один документ.
        """
        try:
            # Проверяем наличие текста
            if not doc.page_content or not doc.page_content.strip():
                logger.warning("Пропускаем документ без текста")
                return None
                
            # Очищаем текст от лишних пробелов и переносов строк
            cleaned_text = ' '.join(doc.page_content.split())
            
            # Проверяем длину текста после очистки
            if len(cleaned_text) < 10:  # Минимальная длина текста
                logger.warning("Пропускаем документ с недостаточным количеством текста")
                return None
                
            # Создаем новый документ с очищенным текстом
            processed_doc = Document(
                page_content=cleaned_text,
                metadata=doc.metadata
            )
            logger.debug(f"Документ успешно обработан: {doc.metadata.get('source', 'unknown')}")
            return processed_doc
            
        except Exception as e:
            logger.error(f"Ошибка при обработке документа: {str(e)}")
            return None

    async def process_documents_async(self) -> List[Document]:
        """
        Асинхронная обработка и подготовка документов для индексации.
        """
        logger.info("Начало асинхронной обработки документов")
        
        if not self.documents:
            error_msg = "Список документов не может быть пустым"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        try:
            # Создаем задачи для асинхронной обработки каждого документа
            tasks = [self._process_single_document(doc) for doc in self.documents]
            results = await asyncio.gather(*tasks)
            
            # Фильтруем None значения и подсчитываем статистику
            processed_docs = [doc for doc in results if doc is not None]
            skipped_docs = len(results) - len(processed_docs)
            
            if not processed_docs:
                error_msg = "Не удалось обработать ни одного документа"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            logger.info(f"Асинхронная обработка завершена. Обработано: {len(processed_docs)}, пропущено: {skipped_docs}")
            return processed_docs
            
        except Exception as e:
            logger.error(f"Критическая ошибка при асинхронной обработке документов: {str(e)}")
            raise

    def process_documents(self) -> List[Document]:
        """
        Синхронная обработка документов (для обратной совместимости).
        """
        return asyncio.run(self.process_documents_async())