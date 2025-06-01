import os
from utils.mylogger import Logger
import asyncio

logger = Logger('CheckDirExists', 'logs/rag.log')

class CheckDirExists:
    """
    Класс для проверки существования и доступа к директории.
    
    Этот класс предоставляет методы для проверки:
    - Существования директории по указанному пути
    - Прав доступа к директории для чтения
    
    Attributes:
        None
    """
    def __init__(self):
        """
        Инициализация класса CheckDirExists.
        """
        logger.info("Инициализация класса CheckDirExists")

    async def check_dir_exists_async(self, dir_path: str) -> bool:
        """
        Асинхронно проверяет существование директории по указанному пути.

        Args:
            dir_path (str): Путь к директории для проверки

        Returns:
            bool: True если директория существует, False в противном случае
        """
        try:
            logger.debug(f"Проверка существования директории: {dir_path}")
            exists = await asyncio.to_thread(os.path.exists, dir_path)
            if not exists:
                logger.error(f"Директория не существует: {dir_path}")
                return False
            logger.info(f"Директория существует: {dir_path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при проверке директории {dir_path}: {str(e)}")
            return False

    def check_dir_exists(self, dir_path: str) -> bool:
        """
        Синхронная обертка для проверки существования директории
        """
        return asyncio.run(self.check_dir_exists_async(dir_path))
        
    async def check_dir_access_async(self, dir_path: str) -> bool:
        """
        Асинхронно проверяет права доступа к директории для чтения.

        Args:
            dir_path (str): Путь к директории для проверки прав доступа

        Returns:
            bool: True если есть права на чтение, False в противном случае
        """
        try:
            logger.debug(f"Проверка прав доступа к директории: {dir_path}")
            has_access = await asyncio.to_thread(os.access, dir_path, os.R_OK)
            if not has_access:
                logger.error(f"Нет прав на чтение директории: {dir_path}")
                return False
            logger.info(f"Есть права на чтение директории: {dir_path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при проверке прав доступа к директории {dir_path}: {str(e)}")
            return False

    def check_dir_access(self, dir_path: str) -> bool:
        """
        Синхронная обертка для проверки прав доступа к директории
        """
        return asyncio.run(self.check_dir_access_async(dir_path))