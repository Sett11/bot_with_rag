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
        llm: Экземпляр класса AdvancedRAG
    """
    def __init__(self, documents: List[Document], llm=None) -> None:
        """
        Инициализация класса ProcessDocuments.
        
        Args:
            documents (List[Document]): Список документов для обработки
            llm: Экземпляр класса AdvancedRAG
        """
        logger.info("Инициализация класса ProcessDocuments")
        self.documents = documents
        self.llm = llm
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

    async def process_documents_async(self, documents: List[Document]) -> List[Document]:
        """
        Асинхронная обработка списка документов.

        Args:
            documents (List[Document]): Список документов для обработки

        Returns:
            List[Document]: Список обработанных документов

        Raises:
            ValueError: Если список документов пуст при загрузке документов
            Exception: При ошибках обработки
        """
        if not documents:
            return []
            
        try:
            processed_docs = []
            for doc in documents:
                try:
                    # Очищаем текст от лишних пробелов и переносов
                    cleaned_text = await asyncio.to_thread(
                        lambda: " ".join(doc.page_content.split())
                    )
                    
                    # Создаем новый документ с очищенным текстом
                    processed_doc = Document(
                        page_content=cleaned_text,
                        metadata=doc.metadata
                    )
                    processed_docs.append(processed_doc)
                    
                except Exception as e:
                    logger.error(f"Ошибка при обработке документа: {str(e)}")
                    continue
                    
            if not processed_docs and self.llm.is_loading_documents:
                raise ValueError("Не удалось обработать ни один документ при загрузке")
                
            return processed_docs
            
        except Exception as e:
            logger.error(f"Ошибка при обработке документов: {str(e)}")
            raise

    def process_documents(self) -> List[Document]:
        """
        Синхронная обработка документов (для обратной совместимости).
        """
        return asyncio.run(self.process_documents_async(self.documents))